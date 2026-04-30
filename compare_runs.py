"""
compare_runs.py — Cross-run comparison plotter
===============================================

Groups completed runs by a chosen key, averages across seeds,
and plots normalised BF gain with mean ± std shading.

Usage
-----
  # Compare SAC across antenna counts (classical_small group)
  python compare_runs.py --filter "sac_nlos_.*ant" --group_by num_ant

  # Compare Q-SAC across VQC depths
  python compare_runs.py --filter "quantum_sac_nlos_2ant" --group_by n_layers

  # Compare classical vs quantum at same antenna count
  python compare_runs.py --filter ".*_nlos_2ant" --group_by algo

  # Compare everything, grouped by algo
  python compare_runs.py --group_by algo

Options
-------
  --runs_dir   Path to experiments/runs/  (default: experiments/runs)
  --filter     Regex applied to run folder name  (default: match all)
  --group_by   Config key to group by: algo | num_ant | n_layers | num_NNs | scenario
               Also accepts compound keys separated by comma: "algo,num_ant"
  --out        Output folder for figures  (default: experiments/comparisons)
  --smooth     Gaussian smoothing sigma in outer-loop units  (default: 0)
  --max_loops  Truncate all curves to this many outer loops  (default: no limit)
"""

import os
import re
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d


# ── Helpers ────────────────────────────────────────────────────────────────

def load_run(run_dir: str) -> dict | None:
    """
    Read one completed run folder.  Returns None if configs or gain files
    are missing.
    """
    algo_cfg_path = os.path.join(run_dir, "configs", "algo_config.yaml")
    env_cfg_path  = os.path.join(run_dir, "configs", "env_config.yaml")
    if not (os.path.exists(algo_cfg_path) and os.path.exists(env_cfg_path)):
        return None

    with open(algo_cfg_path) as f:
        algo_cfg = yaml.safe_load(f)
    with open(env_cfg_path) as f:
        env_cfg = yaml.safe_load(f)

    num_beams = env_cfg.get("num_NNs", 4)
    num_ant   = env_cfg.get("num_ant", 32)
    num_iter  = algo_cfg.get("num_iter", 100)
    path      = env_cfg.get("path", "")
    scenario  = "nlos" if "201-300" in path else "los"

    # ── Load per-beam gain histories ──────────────────────────────────────
    gains = []
    for b in range(num_beams):
        fpath = os.path.join(run_dir, "results", f"gain_history_beam_{b}.npy")
        if not os.path.exists(fpath):
            continue
        try:
            g = np.load(fpath, allow_pickle=False)
            gains.append(g)
        except Exception as e:
            print(f"  [WARN] skip {fpath}: {e}")

    if not gains:
        return None

    # Average across beams, normalise by num_ant
    min_len = min(len(g) for g in gains)
    avg     = np.mean([g[:min_len] for g in gains], axis=0) / num_ant

    # Downsample from RL-step resolution to outer-loop resolution
    # gain_history appends every RL step; take the last step of each outer loop
    avg_per_loop = avg[num_iter - 1 :: num_iter]

    return {
        "run_dir":  run_dir,
        "run_name": os.path.basename(run_dir),
        "algo":     algo_cfg.get("algo", "?"),
        "seed":     algo_cfg.get("seed", 0),
        "num_ant":  num_ant,
        "num_NNs":  num_beams,
        "num_bits": env_cfg.get("num_bits", 4),
        "scenario": scenario,
        "n_qubits": algo_cfg.get("n_qubits"),
        "n_layers": algo_cfg.get("n_layers"),
        "diff_method": algo_cfg.get("diff_method"),
        "reward_type": algo_cfg.get("reward_type", "ternary"),
        "num_loop": len(avg_per_loop),
        "gain":     avg_per_loop,
    }


def make_label(run: dict, group_keys: list[str]) -> str:
    parts = []
    for k in group_keys:
        v = run.get(k)
        if v is None:
            continue
        if k == "num_ant":
            parts.append(f"{v} ant")
        elif k == "n_layers":
            parts.append(f"{v}L")
        elif k == "n_qubits":
            parts.append(f"{v}q")
        else:
            parts.append(str(v))
    return ", ".join(parts) if parts else "?"


def _save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cross-run comparison plotter")
    p.add_argument("--runs_dir", default="experiments/runs")
    p.add_argument("--filter",   default=None,
                   help="Regex to match run folder names (applied to folder name only, not full path)")
    p.add_argument("--group_by", default="algo",
                   help="Comma-separated config keys to group by, e.g. 'algo' or 'algo,num_ant'")
    p.add_argument("--out",      default="experiments/comparisons")
    p.add_argument("--smooth",   type=float, default=0,
                   help="Gaussian smoothing sigma (outer-loop units, 0 = no smoothing)")
    p.add_argument("--max_loops", type=int, default=None,
                   help="Truncate all curves to this many outer loops")
    return p.parse_args()


def main():
    args = parse_args()
    group_keys = [k.strip() for k in args.group_by.split(",")]

    # ── Collect runs ───────────────────────────────────────────────────────
    runs_dir = args.runs_dir
    run_dirs = sorted(
        d for d in (os.path.join(runs_dir, e) for e in os.listdir(runs_dir))
        if os.path.isdir(d) and not os.path.basename(d).startswith("batch_")
    )

    pattern = re.compile(args.filter) if args.filter else None
    runs = []
    for d in run_dirs:
        name = os.path.basename(d)
        if pattern and not pattern.search(name):
            continue
        r = load_run(d)
        if r is None:
            print(f"  [skip] {name}")
            continue
        runs.append(r)

    if not runs:
        print("No matching completed runs found.")
        return

    print(f"Found {len(runs)} run(s) matching filter: {args.filter!r}")

    # ── Group by key, then average across seeds ────────────────────────────
    groups: dict[str, list] = defaultdict(list)
    for r in runs:
        label = make_label(r, group_keys)
        groups[label].append(r)

    print(f"\nGroups ({args.group_by}):")
    for label, members in sorted(groups.items()):
        seeds = sorted(set(m["seed"] for m in members))
        print(f"  {label!r:30s}  {len(members)} run(s), seeds={seeds}")

    # ── Prepare output folder ──────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)

    # ── Plot ───────────────────────────────────────────────────────────────
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (label, members) in enumerate(sorted(groups.items())):
        color = colors[idx % len(colors)]

        # Align curves to same length
        curves = [m["gain"] for m in members]
        if args.max_loops:
            curves = [c[:args.max_loops] for c in curves]
        min_len = min(len(c) for c in curves)
        curves  = np.array([c[:min_len] for c in curves])  # (n_seeds, loops)

        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)

        if args.smooth > 0:
            mean = gaussian_filter1d(mean, sigma=args.smooth)
            std  = gaussian_filter1d(std,  sigma=args.smooth)

        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Outer loop")
    ax.set_ylabel("Normalised BF gain (mean ± std across seeds)")
    ax.set_title(f"Comparison by {args.group_by}  |  filter: {args.filter or 'all'}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Build output filename from filter + group_by
    safe = lambda s: re.sub(r"[^\w\-]", "_", s or "all")
    fname = f"compare__{safe(args.filter)}__{safe(args.group_by)}.png"
    _save_fig(fig, os.path.join(args.out, fname))

    # ── Final gain bar chart ───────────────────────────────────────────────
    labels_sorted = sorted(groups.keys())
    final_means, final_stds = [], []
    for label in labels_sorted:
        curves = [m["gain"] for m in groups[label]]
        if args.max_loops:
            curves = [c[:args.max_loops] for c in curves]
        finals = np.array([c[-1] for c in curves])
        final_means.append(finals.mean())
        final_stds.append(finals.std())

    fig2, ax2 = plt.subplots(figsize=(max(6, len(labels_sorted) * 1.2), 4))
    x = np.arange(len(labels_sorted))
    ax2.bar(x, final_means, yerr=final_stds, capsize=5,
            color=colors[:len(labels_sorted)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_sorted, rotation=15, ha="right")
    ax2.set_ylabel("Final normalised BF gain")
    ax2.set_title(f"Final gain comparison by {args.group_by}")
    ax2.grid(True, alpha=0.3, axis="y")

    fname2 = f"finalbar__{safe(args.filter)}__{safe(args.group_by)}.png"
    _save_fig(fig2, os.path.join(args.out, fname2))

    print(f"\nDone. Figures in: {args.out}/")


if __name__ == "__main__":
    main()
