import os
import json
import numpy as np
import scipy.io as scio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class MetricsWriter:
    """
    Wraps SummaryWriter to capture scalar metrics as raw numpy arrays
    while transparently passing all calls to TensorBoard.
    """

    def __init__(self, tb_writer: SummaryWriter):
        self._tb = tb_writer
        self._scalars: dict = {}

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        self._tb.add_scalar(tag, scalar_value, global_step, **kwargs)
        bucket = self._scalars.setdefault(tag, {"steps": [], "values": []})
        bucket["steps"].append(global_step)
        bucket["values"].append(float(scalar_value))

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, **kwargs):
        self._tb.add_scalars(main_tag, tag_scalar_dict, global_step, **kwargs)
        for k, v in tag_scalar_dict.items():
            tag = f"{main_tag}/{k}"
            bucket = self._scalars.setdefault(tag, {"steps": [], "values": []})
            bucket["steps"].append(global_step)
            bucket["values"].append(float(v))

    def get_scalars(self) -> dict:
        return {
            k: {
                "steps": np.array(v["steps"]),
                "values": np.array(v["values"]),
            }
            for k, v in self._scalars.items()
        }

    def __getattr__(self, name):
        """Proxy everything else (add_text, close, flush, …) to TensorBoard."""
        return getattr(self._tb, name)


# ---------------------------------------------------------------------------
# Data saving
# ---------------------------------------------------------------------------

def save_run_data(run_dir, env_list, agent_writers, options, train_opt):
    """
    Save all training artefacts as raw numpy / mat files.

    Returns (all_gains, all_egc, all_metrics) for downstream plotting.
    """
    num_beam = options["num_NNs"]
    num_ant = options["num_ant"]
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Beam codebook (fix: read from per-beam txt files) ──────────────
    raw = np.zeros((num_beam, 2 * num_ant))
    for beam_id in range(num_beam):
        beam_path = os.path.join(run_dir, "beams", f"beams_{beam_id}_max.txt")
        if os.path.exists(beam_path):
            with open(beam_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            # last line is the bf-vector CSV
            raw[beam_id, :] = np.fromstring(lines[-1], sep=",")

    beam_codebook = (1 / np.sqrt(num_ant)) * (raw[:, ::2] + 1j * raw[:, 1::2])
    np.save(os.path.join(results_dir, "beam_codebook.npy"), beam_codebook)
    scio.savemat(
        os.path.join(results_dir, "beam_codebook.mat"), {"beams": beam_codebook}
    )

    # ── 2. Gain & EGC history per beam ────────────────────────────────────
    all_gains, all_egc = [], []
    for beam_id in range(num_beam):
        gain = np.array(env_list[beam_id].gain_history[1:])
        egc = np.array(env_list[beam_id].EGC_history)
        np.save(os.path.join(results_dir, f"gain_history_beam_{beam_id}.npy"), gain)
        np.save(os.path.join(results_dir, f"egc_history_beam_{beam_id}.npy"), egc)
        all_gains.append(gain)
        all_egc.append(egc)

    # ── 3. Agent training metrics (loss, Q-values) ────────────────────────
    all_metrics = {}
    for beam_id, aw in enumerate(agent_writers):
        if not isinstance(aw, MetricsWriter):
            continue
        scalars = aw.get_scalars()
        if not scalars:
            continue
        # save raw arrays
        np.save(
            os.path.join(results_dir, f"metrics_beam_{beam_id}.npy"),
            scalars,
            allow_pickle=True,
        )
        # also JSON for easy inspection
        json_data = {
            k: {"steps": v["steps"].tolist(), "values": v["values"].tolist()}
            for k, v in scalars.items()
        }
        all_metrics[f"beam_{beam_id}"] = json_data

    with open(os.path.join(results_dir, "metrics_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_gains, all_egc, all_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_run(run_dir, all_gains, all_egc, all_metrics, options, train_opt):
    """Generate all diagnostic figures and save them under run_dir/plots/."""
    num_beam = options["num_NNs"]
    num_ant = options["num_ant"]
    algo = train_opt.get("algo", "RL").upper()
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Fig 1: per-beam gain convergence ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, gain in enumerate(all_gains):
        if len(gain):
            ax.plot(gain / num_ant, label=f"Beam {i}")
    ax.set_xlabel("Outer loop iteration")
    ax.set_ylabel("Normalised BF gain")
    ax.set_title(f"Beamforming gain convergence — {algo}")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, os.path.join(plots_dir, "gain_convergence.png"))

    # ── Fig 2: average gain across beams ──────────────────────────────────
    valid = [g for g in all_gains if len(g)]
    if valid:
        min_len = min(len(g) for g in valid)
        avg = np.mean([g[:min_len] for g in valid], axis=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(avg / num_ant, linewidth=2)
        ax.set_xlabel("Outer loop iteration")
        ax.set_ylabel("Normalised BF gain (avg)")
        ax.set_title(f"Average BF gain — {algo}")
        ax.grid(True, alpha=0.3)
        _save_fig(fig, os.path.join(plots_dir, "average_gain.png"))

    # ── Fig 3: EGC upper bound ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, egc in enumerate(all_egc):
        if len(egc):
            ax.plot(np.array(egc) / num_ant, label=f"Beam {i}")
    ax.set_xlabel("Outer loop iteration")
    ax.set_ylabel("Normalised EGC upper bound")
    ax.set_title("Equal Gain Combining upper bound")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, os.path.join(plots_dir, "egc_history.png"))

    # ── Fig 4: final gain vs EGC per beam ────────────────────────────────
    final_gains = [g[-1] / num_ant if len(g) else 0 for g in all_gains]
    final_egc = [e[-1] / num_ant if len(e) else 0 for e in all_egc]
    x = np.arange(num_beam)
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.35
    ax.bar(x - w / 2, final_gains, w, label="Learned BF gain")
    ax.bar(x + w / 2, final_egc, w, label="EGC upper bound")
    ax.set_xlabel("Beam index")
    ax.set_ylabel("Normalised gain")
    ax.set_title(f"Final gain vs EGC — {algo}")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save_fig(fig, os.path.join(plots_dir, "final_gain_vs_egc.png"))

    # ── Fig 5: training metrics (loss/Q) per beam ─────────────────────────
    for beam_key, beam_metrics in all_metrics.items():
        loss_keys = [k for k in beam_metrics if "loss" in k.lower() or "value" in k.lower()]
        if not loss_keys:
            continue
        n = len(loss_keys)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
        for ax, key in zip(axes[:, 0], loss_keys):
            steps = beam_metrics[key]["steps"]
            values = beam_metrics[key]["values"]
            ax.plot(steps, values, linewidth=0.8)
            ax.set_xlabel("Global step")
            ax.set_ylabel(key.split("/")[-1])
            ax.set_title(f"{beam_key} — {key}")
            ax.grid(True, alpha=0.3)
        _save_fig(fig, os.path.join(plots_dir, f"training_metrics_{beam_key}.png"))
