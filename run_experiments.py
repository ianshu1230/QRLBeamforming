#!/usr/bin/env python
"""
run_experiments.py — Batch experiment runner
=============================================

Usage
-----
  python run_experiments.py                   # run all in EXPERIMENTS
  python run_experiments.py --dry-run         # print commands, no execution
  python run_experiments.py --idx 0 2 4       # run only these indices
  python run_experiments.py --group quantum   # run a named group

Editing this file
-----------------
• Classical experiments → CLASSICAL_EXPERIMENTS list (flat).
• Quantum experiments  → edit the quantum_grid() call at the bottom.
  quantum_grid() takes keyword arguments where each value is a LIST.
  It returns the full Cartesian product of all parameter lists.

Each dict in EXPERIMENTS maps directly to a CLI argument of src/train.py.
Keys not present in the dict fall back to experiments/configs/*.yaml.

Adding a new algorithm
----------------------
1. Implement CreateAgent.<algo>() in src/create_agent.py.
2. Add the noise branch in src/train.py (see existing examples).
3. Add entries here in quantum_grid() or CLASSICAL_EXPERIMENTS.
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from itertools import product as cartesian_product


# ── Helpers ────────────────────────────────────────────────────────────────

def quantum_grid(**param_lists) -> list[dict]:
    """
    Build every combination of the given parameter lists (grid search).

    Parameters
    ----------
    **param_lists : each keyword argument must be a list of values.

    Returns
    -------
    list of dicts, one per combination, with an auto-generated exp_name.

    Example
    -------
    quantum_grid(
        algo        = ["quantum_sac"],
        scenario    = ["nlos"],
        seed        = [0, 1, 42],
        num_ant     = [2, 4],
        n_qubits    = [2, 4],   # must equal num_ant for direct 1:1 mapping
        n_layers    = [1, 2, 4],
        num_NNs     = [2],
        num_loop    = [500],
        reward_type = ["ternary"],
    )
    """
    keys   = list(param_lists.keys())
    combos = list(cartesian_product(*param_lists.values()))
    exps   = []
    for combo in combos:
        exp = dict(zip(keys, combo))
        # Auto-generate a descriptive experiment name
        parts = [exp.get("algo", "rl"), exp.get("scenario", "")]
        if "num_ant"  in exp: parts.append(f"{exp['num_ant']}ant")
        if "num_NNs"  in exp: parts.append(f"{exp['num_NNs']}beam")
        if "n_qubits" in exp: parts.append(f"{exp['n_qubits']}q")
        if "n_layers" in exp: parts.append(f"{exp['n_layers']}L")
        if "reward_type" in exp and exp["reward_type"] != "ternary":
            parts.append(exp["reward_type"])
        parts.append(f"s{exp.get('seed', 0)}")
        exp["exp_name"] = "_".join(str(p) for p in parts if p)
        exps.append(exp)
    return exps


def classical_grid(**param_lists) -> list[dict]:
    """Same as quantum_grid but for classical algorithms (no quantum keys)."""
    keys   = list(param_lists.keys())
    combos = list(cartesian_product(*param_lists.values()))
    exps   = []
    for combo in combos:
        exp = dict(zip(keys, combo))
        parts = [exp.get("algo", "rl"), exp.get("scenario", "")]
        if "num_ant"     in exp: parts.append(f"{exp['num_ant']}ant")
        if "num_NNs"     in exp: parts.append(f"{exp['num_NNs']}beam")
        if "reward_type" in exp and exp["reward_type"] != "ternary":
            parts.append(exp["reward_type"])
        parts.append(f"s{exp.get('seed', 0)}")
        exp["exp_name"] = "_".join(str(p) for p in parts if p)
        exps.append(exp)
    return exps


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS — edit these to control what gets run
# ══════════════════════════════════════════════════════════════════════════════

# ── Group A: Classical baselines (paper replication) ──────────────────────
CLASSICAL_EXPERIMENTS = classical_grid(
    algo        = ["sac", "td3", "ddpg"],
    scenario    = ["nlos", "los"],
    seed        = [0, 1, 42],
    num_ant     = [32],
    num_NNs     = [4],
    num_loop    = [1000],
    reward_type = ["ternary"],
)

# ── Group B: Classical baselines matching Quantum antenna count ────────────
# Needed for fair comparison: SAC(2 ant) vs Q-SAC(2 ant).
CLASSICAL_SMALL_ANT = classical_grid(
    algo        = ["sac"],
    scenario    = ["nlos"],
    seed        = [0, 1, 42],
    num_ant     = [2, 4, 8],
    num_NNs     = [2],
    num_loop    = [500],
    reward_type = ["ternary"],
)

# ── Group C: Quantum experiments — start small, scale up ──────────────────
#
# IMPORTANT: n_qubits should equal num_ant for direct 1:1 qubit-antenna mapping.
# If n_qubits < num_ant, a classical linear encoder compresses the state first.
#
# Recommended order:
#   1. Debug with num_ant=2, n_qubits=2 (trivial: 4-state statevector)
#   2. Ablation: vary n_layers with num_ant fixed
#   3. Scale: num_ant=4 → 8 → 16

QUANTUM_EXPERIMENTS = quantum_grid(
    algo        = ["quantum_sac"],
    scenario    = ["nlos"],
    seed        = [0, 1, 42],
    num_ant     = [2],          # 2 → 4 → 8 as you scale up
    n_qubits    = [2],          # set equal to num_ant for direct mapping
    n_layers    = [1, 2, 4],    # circuit depth ablation
    num_NNs     = [2],
    num_loop    = [500],
    reward_type = ["ternary"],
    diff_method = ["backprop"],
)

# ── Group D: Quantum scaling study ────────────────────────────────────────
# Run after Group C to study scaling behaviour.
QUANTUM_SCALING = quantum_grid(
    algo        = ["quantum_sac"],
    scenario    = ["nlos"],
    seed        = [0],
    num_ant     = [2, 4, 8],
    n_qubits    = [2, 4, 8],    # paired with num_ant
    n_layers    = [2],
    num_NNs     = [2],
    num_loop    = [500],
    reward_type = ["ternary"],
    diff_method = ["backprop"],
)

# ── MASTER LIST — choose which groups to run ──────────────────────────────
EXPERIMENT_GROUPS = {
    "classical":       CLASSICAL_EXPERIMENTS,
    "classical_small": CLASSICAL_SMALL_ANT,
    "quantum":         QUANTUM_EXPERIMENTS,
    "quantum_scaling": QUANTUM_SCALING,
    "all":             (CLASSICAL_EXPERIMENTS + CLASSICAL_SMALL_ANT
                        + QUANTUM_EXPERIMENTS + QUANTUM_SCALING),
    # Quick sanity-check: 1 experiment per group
    "smoke":           [
        classical_grid(algo=["sac"],         scenario=["nlos"], seed=[0],
                       num_ant=[2], num_NNs=[2], num_loop=[5])[0],
        quantum_grid(  algo=["quantum_sac"], scenario=["nlos"], seed=[0],
                       num_ant=[2], n_qubits=[2], n_layers=[1],
                       num_NNs=[2], num_loop=[5], diff_method=["backprop"])[0],
    ],
}

# Default group to run when no --group is specified
DEFAULT_GROUP = "quantum"


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def build_cmd(exp: dict) -> list[str]:
    cmd = [sys.executable, "src/train.py"]
    for key, val in exp.items():
        cmd += [f"--{key}", str(val)]
    return cmd


def run_experiment(exp: dict, dry_run: bool) -> bool:
    cmd  = build_cmd(exp)
    name = exp.get("exp_name", str(exp))
    print(f"\n{'─'*64}")
    print(f"  {name}")
    print(f"  {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def parse_args():
    p = argparse.ArgumentParser(description="Batch experiment runner")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--idx", type=int, nargs="*", default=None,
                   help="Run only experiments at these 0-based indices")
    p.add_argument("--group", type=str, default=DEFAULT_GROUP,
                   choices=list(EXPERIMENT_GROUPS.keys()),
                   help=f"Experiment group to run (default: {DEFAULT_GROUP})")
    return p.parse_args()


def main():
    args = parse_args()

    all_exps = EXPERIMENT_GROUPS[args.group]
    to_run   = (
        [(i, all_exps[i]) for i in args.idx]
        if args.idx is not None
        else list(enumerate(all_exps))
    )

    tag = " [DRY RUN]" if args.dry_run else ""
    print(f"\nGroup: {args.group} — {len(to_run)} experiment(s){tag}")
    print("─" * 64)
    for i, exp in to_run:
        print(f"  #{i:02d}  {exp.get('exp_name', '?')}")

    results = []
    for pos, (i, exp) in enumerate(to_run):
        print(f"\n[{pos+1}/{len(to_run)}] #{i:02d} {exp.get('exp_name', '?')}")
        ok = run_experiment(exp, args.dry_run)
        results.append((i, exp.get("exp_name", f"exp_{i}"), ok))
        print("  →", "OK" if ok else "FAILED")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f"SUMMARY  (group: {args.group})")
    print(f"{'═'*64}")
    ok_count = sum(1 for _, _, ok in results if ok)
    for i, name, ok in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] #{i:02d}  {name}")
    print(f"\n  {ok_count}/{len(results)} succeeded")

    # ── Log file ───────────────────────────────────────────────────────────
    log_dir  = os.path.join("experiments", "runs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"batch_{args.group}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    )
    with open(log_path, "w") as f:
        f.write(f"group: {args.group}\n")
        for i, name, ok in results:
            f.write(f"{'OK' if ok else 'FAIL'}\t#{i:02d}\t{name}\n")
    print(f"  log → {log_path}")

    if any(not ok for _, _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
