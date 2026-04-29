"""
quantum_policy.py
=================
Hybrid quantum-classical stochastic policy for beam-pattern RL (Q-SAC actor).

Architecture
------------
  State  s  (obs_shape)
      │
      ▼  [Classical Encoder — skipped when obs_shape == n_qubits]
  Linear(obs_shape → n_qubits) + Tanh
      │
      ▼  Variational Quantum Circuit
  ╔════════════════════════════════════════════╗
  ║  AngleEmbedding: RY(x_i) on qubit i       ║
  ║  ┌─ layer 0 ──────────────────────────┐   ║
  ║  │  RY(θ_i) + RZ(φ_i)  ∀ qubit i    │   ║
  ║  │  CNOT ring: 0→1→…→(n-1)→0         │   ║
  ║  └────────────────────────────────────┘   ║
  ║  … × n_layers                             ║
  ║  Measure ⟨Z_i⟩ ∈ [-1, 1]  ∀ qubit i     ║
  ╚════════════════════════════════════════════╝
      │
      ▼  Classical Decoder
  Linear(n_qubits → 2 × action_shape)
      │
      ├── μ      → tanh → scaled to [low_action, high_action]
      └── log_σ  → clamped to [LOG_STD_MIN, LOG_STD_MAX]
      │
      ▼  Reparameterization trick (SAC standard)
  action ∈ [low_action, high_action]^action_shape

Gradient methods
----------------
  "backprop"        Fast torch autograd through the statevector simulation.
                    Only valid on "default.qubit" (simulation).
  "parameter-shift" Hardware-compatible. Each gradient evaluation runs the
                    circuit 2 × (n_layers × n_qubits × 2) extra times.
                    Slower, but required for real quantum hardware.

Scaling guide
-------------
  num_ant = n_qubits → no encoder, clean 1:1 qubit-antenna mapping.
  num_ant > n_qubits → classical encoder compresses state first.

  num_ant |  n_qubits | statevector RAM | speed
  --------|-----------|-----------------|-------
       2  |         2 |         32 B    | trivial  ← start here
       4  |         4 |        512 B    | fast
       8  |         8 |          4 KB   | fast
      16  |        16 |          1 MB   | ok
      32  |        32 |          4 GB   | very slow (use encoder → 8 qubits)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import pennylane as qml

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class QuantumGaussianPolicy(nn.Module):
    """
    Hybrid quantum-classical stochastic policy.

    The VQC replaces the classical MLP actor used in SAC.
    Classical critic networks remain unchanged (see create_agent.py).

    Parameters
    ----------
    observation_shape : int
        Dimension of the state vector (= num_ant).
    action_shape : int
        Dimension of the action vector (= num_ant).
    n_qubits : int
        Number of qubits.  Set equal to observation_shape for a direct 1:1
        angle-per-qubit encoding (recommended when num_ant ≤ 16).
    n_layers : int
        Variational circuit depth (number of [RY+RZ+CNOT] blocks).
    high_action : float
        Upper action bound (default π).
    low_action : float
        Lower action bound (default -π).
    diff_method : str
        PennyLane differentiation method: "backprop" or "parameter-shift".
    """

    def __init__(
        self,
        observation_shape: int,
        action_shape: int,
        n_qubits: int = 2,
        n_layers: int = 2,
        high_action: float = np.pi,
        low_action: float = -np.pi,
        diff_method: str = "backprop",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.action_shape = action_shape

        # ── Classical encoder (only when obs_shape ≠ n_qubits) ───────────
        self.use_encoder = (observation_shape != n_qubits)
        if self.use_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(observation_shape, n_qubits),
                nn.Tanh(),
            )

        # ── Variational Quantum Circuit ───────────────────────────────────
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def _vqc(inputs, weights):
            # 1. Angle embedding — encode each qubit with one state dimension
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            # 2. Variational layers: per-qubit rotations + entanglement
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                # CNOT ring: 0→1→2→…→(n-1)→0
                for qubit in range(n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])

            # 3. Pauli-Z expectation on every qubit → scalar ∈ [-1, 1]
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # TorchLayer wraps the QNode as an nn.Module;
        # "weights" becomes a trainable nn.Parameter automatically.
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.vqc_layer = qml.qnn.TorchLayer(_vqc, weight_shapes)

        # ── Classical decoder ─────────────────────────────────────────────
        # VQC output (n_qubits) → μ and log_σ (each action_shape)
        self.decoder = nn.Linear(n_qubits, 2 * action_shape)

        # Action scaling — stored as buffers so they move with .to(device)
        self.register_buffer(
            "action_scale",
            torch.tensor((high_action - low_action) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((high_action + low_action) / 2.0, dtype=torch.float32),
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map state to VQC input ∈ [-π/2, π/2].

        AngleEmbedding uses RY(θ); keeping |θ| ≤ π/2 avoids the
        redundancy that appears at |θ| > π.
        """
        if self.use_encoder:
            # encoder output ∈ [-1,1] (Tanh), scaled to [-π/2, π/2]
            return self.encoder(x) * (torch.pi / 2.0)
        # Direct mapping: state ∈ [-π, π] → [-π/2, π/2]
        return x / 2.0

    # ── Public interface (matches GaussianPolicy from modularl) ──────────

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor (batch, observation_shape)

        Returns
        -------
        action   : Tensor (batch, action_shape)  — sampled, tanh-scaled
        log_prob : Tensor (batch, 1)             — log π(action|state)
        mean     : Tensor (batch, action_shape)  — deterministic mean action
        """
        encoded = self._encode(x)               # (batch, n_qubits)
        vqc_out = self.vqc_layer(encoded)       # (batch, n_qubits), ∈ [-1, 1]

        raw = self.decoder(vqc_out)             # (batch, 2 * action_shape)
        mean_raw, log_std = raw.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        # Reparameterization trick
        dist  = Normal(mean_raw, std)
        x_t   = dist.rsample()
        y_t   = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Log-prob with tanh change-of-variables correction
        log_prob  = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)

        mean = torch.tanh(mean_raw) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action(self, x: torch.Tensor):
        """Alias for forward — matches GaussianPolicy interface."""
        return self.forward(x)

    def extra_repr(self) -> str:
        enc = f"encoder Linear({next(self.encoder.children()).in_features}→{self.n_qubits}) → " \
              if self.use_encoder else "no encoder, "
        return (
            f"{enc}VQC({self.n_qubits} qubits, {self.n_layers} layers), "
            f"decoder Linear({self.n_qubits}→{2*self.action_shape})"
        )
