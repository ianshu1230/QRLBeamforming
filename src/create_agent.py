from modularl.agents import SAC, TD3, DDPG
from modularl.policies import GaussianPolicy, DeterministicPolicy
from modularl.q_functions import SAQNetwork
from modularl.replay_buffers import ReplayBuffer
import torch.optim as optim


class CreateAgent:
    """
    Factory for RL agents used in beam codebook learning.

    Classical agents (sac, td3, ddpg)
        Actor and critics are classical MLP networks from modularl.

    Quantum agent (quantum_sac)
        Actor  : QuantumGaussianPolicy — VQC replaces the classical MLP.
        Critics: classical SAQNetwork (unchanged) — stable Q-value estimation.
        The rest of the SAC training loop is identical to the classical case.

    Adding a new algorithm
    ----------------------
    1. Add a method here following the same pattern.
    2. Add a branch in src/train.py (noise setup + train() noise handling).
    3. Add entries to run_experiments.py EXPERIMENTS / quantum_grid().
    """
    def __init__(
        self,
    ):
        pass

    def sac(self, train_opt, writer):
        actor = GaussianPolicy(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
            high_action=train_opt["high_action"],
            low_action=train_opt["low_action"],
        )
        qf1 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        qf2 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        replay_buffer = ReplayBuffer(
            buffer_size=train_opt["buffer_size"],
        )
        qf_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()),
            lr=train_opt["q_lr"],
            weight_decay=train_opt["q_weight_decay"],
        )
        actor_optimizer = optim.Adam(
            list(actor.parameters()),
            lr=train_opt["policy_lr"],
            weight_decay=train_opt["policy_weight_decay"],
        )
        agent = SAC(
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            actor_optimizer=actor_optimizer,
            qf_optimizer=qf_optimizer,
            replay_buffer=replay_buffer,
            gamma=train_opt["gamma"],
            entropy_lr=train_opt["q_lr"],
            tau=train_opt["tau"],
            batch_size=train_opt["batch_size"],
            learning_starts=train_opt["learning_starts"],
            device=train_opt["device"],
            target_network_frequency=train_opt["target_network_frequency"],
            policy_frequency=train_opt["policy_frequency"],
            target_entropy=-train_opt["action_shape"],
            writer=writer,
        )
        agent.init()
        return agent

    def quantum_sac(self, train_opt, writer):
        """
        Q-SAC: SAC with a Variational Quantum Circuit (VQC) actor.

        Only the actor is quantum; both critics remain classical.
        This hybrid design keeps training stable (classical Q-value anchor)
        while replacing the decision-making policy with a VQC.

        Relevant train_opt keys
        -----------------------
        n_qubits    : int   — qubits in the VQC (default = obs_shape)
        n_layers    : int   — variational circuit depth (default 2)
        diff_method : str   — "backprop" (fast sim) | "parameter-shift" (hw)
        """
        from Codebook_Learning_RL.quantum_policy import QuantumGaussianPolicy

        n_qubits = train_opt.get("n_qubits") or train_opt["obs_shape"]
        n_layers = train_opt.get("n_layers", 2)
        diff_method = train_opt.get("diff_method", "backprop")

        # Quantum actor — VQC replaces the classical Gaussian MLP
        actor = QuantumGaussianPolicy(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
            n_qubits=n_qubits,
            n_layers=n_layers,
            high_action=train_opt["high_action"],
            low_action=train_opt["low_action"],
            diff_method=diff_method,
        )

        # Classical critics — identical to standard SAC
        qf1 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        qf2 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        replay_buffer = ReplayBuffer(buffer_size=train_opt["buffer_size"])

        qf_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()),
            lr=train_opt["q_lr"],
            weight_decay=train_opt["q_weight_decay"],
        )
        actor_optimizer = optim.Adam(
            list(actor.parameters()),
            lr=train_opt["policy_lr"],
            weight_decay=train_opt["policy_weight_decay"],
        )

        agent = SAC(
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            actor_optimizer=actor_optimizer,
            qf_optimizer=qf_optimizer,
            replay_buffer=replay_buffer,
            gamma=train_opt["gamma"],
            entropy_lr=train_opt["q_lr"],
            tau=train_opt["tau"],
            batch_size=train_opt["batch_size"],
            learning_starts=train_opt["learning_starts"],
            device=train_opt["device"],
            target_network_frequency=train_opt["target_network_frequency"],
            policy_frequency=train_opt["policy_frequency"],
            target_entropy=-train_opt["action_shape"],
            writer=writer,
        )
        agent.init()
        print(f"  Q-SAC actor: {actor}")
        return agent

    def td3(self, train_opt, writer):
        actor = DeterministicPolicy(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
            high_action=train_opt["high_action"],
            low_action=train_opt["low_action"],
        )
        qf1 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        qf2 = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        replay_buffer = ReplayBuffer(
            buffer_size=train_opt["buffer_size"],
        )
        qf_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()),
            lr=train_opt["q_lr"],
            weight_decay=train_opt["q_weight_decay"],
        )
        actor_optimizer = optim.Adam(
            list(actor.parameters()),
            lr=train_opt["policy_lr"],
            weight_decay=train_opt["policy_weight_decay"],
        )
        agent = TD3(
            actor=actor,
            qf1=qf1,
            qf2=qf2,
            actor_optimizer=actor_optimizer,
            qf_optimizer=qf_optimizer,
            replay_buffer=replay_buffer,
            gamma=train_opt["gamma"],
            tau=train_opt["tau"],
            batch_size=train_opt["batch_size"],
            learning_starts=train_opt["learning_starts"],
            device=train_opt["device"],
            exploration_noise=0,
            policy_frequency=train_opt["policy_frequency"],
            writer=writer,
        )
        agent.init()
        return agent

    def ddpg(self, train_opt, writer):
        actor = DeterministicPolicy(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
            high_action=train_opt["high_action"],
            low_action=train_opt["low_action"],
        )
        qf = SAQNetwork(
            observation_shape=train_opt["obs_shape"],
            action_shape=train_opt["action_shape"],
        )
        replay_buffer = ReplayBuffer(
            buffer_size=train_opt["buffer_size"],
        )
        qf_optimizer = optim.Adam(
            list(qf.parameters()),
            lr=train_opt["q_lr"],
            weight_decay=train_opt["q_weight_decay"],
        )
        actor_optimizer = optim.Adam(
            list(actor.parameters()),
            lr=train_opt["policy_lr"],
            weight_decay=train_opt["policy_weight_decay"],
        )
        agent = DDPG(
            actor=actor,
            qf=qf,
            actor_optimizer=actor_optimizer,
            qf_optimizer=qf_optimizer,
            replay_buffer=replay_buffer,
            gamma=train_opt["gamma"],
            tau=train_opt["tau"],
            batch_size=train_opt["batch_size"],
            learning_starts=train_opt["learning_starts"],
            device=train_opt["device"],
            exploration_noise=0,
            policy_frequency=train_opt["policy_frequency"],
            writer=writer,
        )
        agent.init()
        return agent
