# Fast RL for Robust Beam Codebooks

Implementation of:
> **"Fast Reinforcement Learning for Robust Beam Codebooks in Future Communication Systems"**
> Nechi et al., IEEE Transactions on Communications, Vol. 74, 2026.

Multi-agent RL framework for learning beamforming codebooks in mmWave/THz massive MIMO systems without channel-state information. Supports SAC, TD3, DDPG, and a hybrid Quantum-SAC (VQC actor), evaluated on LoS and NLoS scenarios using the [DeepMIMO dataset](https://www.deepmimo.net/).

---

## Installation

```bash
pip install -r requirements.txt
pip install tensordict h5py tqdm tensorboard matplotlib scipy pennylane pennylane-lightning
```

---

## How to run

### Single run

```bash
python src/train.py
```

Uses settings from `experiments/configs/algo_config.yaml` and `experiments/configs/env_config.yaml`.

Override any parameter via CLI without editing the YAML:

```bash
python src/train.py --algo sac --scenario nlos --seed 0 --num_NNs 4 --num_loop 1000
```

**All CLI flags:**

| Flag | Default (YAML) | Options |
|---|---|---|
| `--algo` | `sac` | `sac` / `td3` / `ddpg` / `quantum_sac` |
| `--seed` | `0` | any int |
| `--scenario` | *(from path in env_config)* | `los` / `nlos` |
| `--num_NNs` | `9` | codebook size (beams) |
| `--num_ant` | `32` | BS antenna count |
| `--num_loop` | `1000` | outer training loops |
| `--num_iter` | `100` | RL update steps per outer loop |
| `--reward_type` | `ternary` | `binary` / `ternary` |
| `--exp_name` | *(from algo_config)* | run folder prefix |
| `--device` | `cuda:0` | any PyTorch device string |
| `--n_qubits` | `num_ant` | VQC qubits (`quantum_sac` only) |
| `--n_layers` | `2` | VQC circuit depth (`quantum_sac` only) |
| `--diff_method` | `backprop` | `backprop` / `parameter-shift` (`quantum_sac` only) |

### Batch run

Edit the experiment lists at the top of `run_experiments.py`, then:

```bash
# Run a named group
python run_experiments.py --group quantum

# Run all groups
python run_experiments.py --group all

# Dry run — print commands without executing
python run_experiments.py --group all --dry-run

# Run specific indices only
python run_experiments.py --group classical --idx 0 2 4
```

**Predefined groups:**

| Group | Content | # Runs |
|---|---|---|
| `classical` | SAC / TD3 / DDPG × NLoS / LoS × seed 0,1,42 × 32 ant, 4 beam | 18 |
| `classical_small` | SAC × NLoS × seed 0,1,42 × 2/4/8 ant, 2 beam | 9 |
| `quantum` | Q-SAC × NLoS × seed 0,1,42 × 2 ant, layers 1/2/4 | 9 |
| `quantum_scaling` | Q-SAC × NLoS × seed 0 × 2/4/8 ant, layer 2 | 3 |
| `all` | all of the above | 39 |
| `smoke` | 1 classical + 1 quantum (5 loops each, for quick sanity check) | 2 |

### Monitor training (TensorBoard)

```bash
# All runs
tensorboard --logdir=experiments/runs

# Single run
tensorboard --logdir=experiments/runs/<run_name>
```

---

## Output structure

Each run creates a folder: `experiments/runs/<exp_name>__<YYYY-MM-DD_HH-MM-SS>/`

```
<run>/
├── configs/
│   ├── algo_config.yaml          # snapshot of algo settings used
│   └── env_config.yaml           # snapshot of env settings used
│
├── beams/
│   ├── beams_N_max.txt           # append-only log: each new best for beam N
│   │                             #   odd lines  → BF gain value (%.2f)
│   │                             #   even lines → beam vector CSV (%.5f,…)
│   ├── beam_codebook.npy         # final codebook: complex (num_beams, num_ant)
│   └── beam_codebook.mat         # same, MATLAB format  {"beams": array}
│
├── results/
│   ├── beam_codebook.npy         # same codebook, copy under results/
│   ├── beam_codebook.mat         # same, MATLAB format
│   ├── gain_history_beam_N.npy   # BF gain per outer loop, beam N  (raw, 1-D)
│   ├── egc_history_beam_N.npy    # EGC upper bound per outer loop, beam N (1-D)
│   ├── metrics_beam_N.npy        # TensorBoard scalars dict (allow_pickle)
│   └── metrics_summary.json      # same scalars as JSON
│
├── plots/
│   ├── gain_convergence.png      # per-beam BF gain over training (normalised)
│   ├── average_gain.png          # mean gain across all beams
│   ├── egc_history.png           # EGC upper bound curves
│   ├── final_gain_vs_egc.png     # bar chart: learned gain vs EGC per beam
│   └── training_metrics_beam_N.png  # loss & Q-value curves per agent
│
├── beamforming_gain_records/
│   └── beam_N_gain_records.npy   # legacy: gain history + EGC, normalised by num_ant
│
├── NN_results_N/                 # TensorBoard event files for agent N
├── kmeans_model.sav              # trained K-means classifier (pickle)
├── sensing_beam.npy              # random sensing beams used for clustering
└── ou_noise_records.npy          # OUNoise state history (DDPG only)
```

### Key output files

**`results/beam_codebook.npy`** — the main result. Shape `(num_beams, num_ant)`, complex float.
Load with:
```python
import numpy as np
cb = np.load("results/beam_codebook.npy")   # (num_beams, num_ant), complex
```

**`results/gain_history_beam_N.npy`** — best BF gain at each outer loop for beam N. Raw (not normalised). Divide by `num_ant` to get the normalised value plotted in the figures.

**`results/metrics_summary.json`** — training scalars (actor loss, critic loss, Q-values) for all beams. Structure:
```json
{
  "beam_0": {
    "losses/qf_loss":    {"steps": [...], "values": [...]},
    "losses/actor_loss": {"steps": [...], "values": [...]},
    "losses/qf1_values": {"steps": [...], "values": [...]}
  },
  "beam_1": { ... }
}
```

**`beams/beams_N_max.txt`** — human-readable log of every improvement for beam N:
```
12.34          ← BF gain (dB-scale raw value)
0.12345,...    ← beam vector (2*num_ant values: real, imag interleaved)
15.67
0.23456,...
```
The last pair of lines is always the best beam found.

---

## How to add a new algorithm

Three files need to be touched: `create_agent.py`, `train.py`, and `run_experiments.py`.

### Step 1 — `src/create_agent.py`

Add a method to `CreateAgent` following the same pattern as `sac` / `td3` / `ddpg`:

```python
def my_algo(self, train_opt, writer):
    actor  = ...          # must implement .act_train(state) → Tensor
    qf1    = SAQNetwork(observation_shape=..., action_shape=...)
    replay = ReplayBuffer(buffer_size=train_opt["buffer_size"])
    agent  = MyAlgoAgent(actor=actor, qf1=qf1, replay_buffer=replay,
                         ..., writer=writer)
    agent.init()
    return agent
```

The agent must implement:

| Method / attribute | Purpose |
|---|---|
| `act_train(state) → Tensor` | action for exploration during training |
| `observe(s, a, r, s1, done)` | store transition + update networks |
| `rb.extend(TensorDict)` | replay buffer extension |
| `global_step` (int) | step counter used for TensorBoard logging |

### Step 2 — `src/train.py`

**2a.** In the `for beam_id in range(options["num_NNs"])` agent-creation loop, add a branch to instantiate the agent and its noise:

```python
elif train_opt["algo"] == "my_algo":
    agent = create_agent.my_algo(train_opt, agent_writer)
    noise_list.append(GaussianNoise(...))  # or None if algo handles exploration internally
```

**2b.** In the `train()` function, add a branch for how the noisy action is computed:

```python
elif algo == "my_algo":
    action_pred_noisy = noise.get_action(action, t=train_options["overall_iter"])
```

If the algorithm handles exploration internally (like SAC), set `noise_list.append(None)` and route to `action_pred_noisy = action`.

### Step 3 — `run_experiments.py`

Add entries to `CLASSICAL_EXPERIMENTS` (or a new list) and register it in `EXPERIMENT_GROUPS`:

```python
MY_ALGO_EXPERIMENTS = classical_grid(
    algo        = ["my_algo"],
    scenario    = ["nlos"],
    seed        = [0, 1, 42],
    num_ant     = [32],
    num_NNs     = [4],
    num_loop    = [1000],
    reward_type = ["ternary"],
)

EXPERIMENT_GROUPS = {
    ...
    "my_algo": MY_ALGO_EXPERIMENTS,
    "all":     CLASSICAL_EXPERIMENTS + ... + MY_ALGO_EXPERIMENTS,
}
```

Then run it:

```bash
python run_experiments.py --group my_algo
```

---

## Project structure

```
QRLBeamforming/
├── src/
│   ├── train.py                    # main entry point
│   ├── create_agent.py             # agent factory: SAC / TD3 / DDPG / Q-SAC
│   ├── save_results.py             # MetricsWriter, save_run_data(), plot_run()
│   └── Codebook_Learning_RL/
│       ├── env_ddpg.py             # RL environment (reward, BF gain, step)
│       ├── DDPG_classes.py         # OUNoise, GaussianNoise, Actor/Critic nets
│       ├── quantum_policy.py       # QuantumGaussianPolicy (VQC actor)
│       ├── clustering.py           # K-means clustering via sensing beams
│       ├── function_lib.py         # BF gain calculation, pairwise-diff features
│       ├── DataPrep.py             # load DeepMIMO .mat files
│       ├── DFT_gen.py              # DFT codebook baseline
│       ├── grid201-300.mat         # NLoS channel dataset (28 GHz)
│       └── grid1101-1400.mat       # LoS  channel dataset (60 GHz)
├── experiments/
│   ├── configs/
│   │   ├── algo_config.yaml        # RL hyperparameters
│   │   └── env_config.yaml         # antenna / dataset / codebook-size settings
│   └── runs/                       # auto-created; one sub-folder per run
├── run_experiments.py              # batch runner
└── requirements.txt
```

---

## Config reference

### `experiments/configs/algo_config.yaml`

```yaml
algo: "sac"             # sac | td3 | ddpg | quantum_sac
seed: 0
reward_type: "ternary"  # binary {-1,+1} | ternary {-1,0,+1} (paper Eq.22)

buffer_size: 8192
batch_size: 1024
gamma: 0                # discount factor (0 = myopic)

q_lr: 0.003
policy_lr: 0.003
q_weight_decay: 0.01
policy_weight_decay: 0.001
tau: 0.001
learning_starts: 0
target_network_frequency: 1
policy_frequency: 1

autotune: true          # SAC: auto-tune entropy coefficient
alpha: 0.2              # SAC: initial entropy coefficient

td3_noise_max: 0.5      # TD3: Gaussian noise σ at step 0
td3_noise_min: 0.05     # TD3: Gaussian noise σ floor
td3_noise_decay: 100000 # TD3: linear decay period (steps)

n_qubits: null          # Q-SAC: null = use num_ant
n_layers: 2             # Q-SAC: VQC circuit depth
diff_method: "backprop" # Q-SAC: "backprop" | "parameter-shift"

num_iter: 100           # RL update steps per outer loop
```

### `experiments/configs/env_config.yaml`

```yaml
num_ant: 32             # BS antenna count
num_bits: 4             # phase-shifter resolution (2^4 = 16 levels)
num_NNs: 9              # codebook size (one RL agent per beam)
ch_sample_ratio: 0.5    # fraction of users sampled each outer loop
num_loop: 1000          # outer training loops
path: 'src/Codebook_Learning_RL/grid201-300.mat'   # NLoS
# path: 'src/Codebook_Learning_RL/grid1101-1400.mat'  # LoS
device: "cuda:0"
```

---

## Citation

```bibtex
@article{nechi2026fast,
  title   = {Fast Reinforcement Learning for Robust Beam Codebooks
             in Future Communication Systems},
  author  = {Nechi, Anouar and Narjis, Zakaria and Buchty, Rainer
             and Berekovic, Mladen and Mulhem, Saleh},
  journal = {IEEE Transactions on Communications},
  volume  = {74},
  year    = {2026},
  doi     = {10.1109/TCOMM.2025.3642701}
}
```
