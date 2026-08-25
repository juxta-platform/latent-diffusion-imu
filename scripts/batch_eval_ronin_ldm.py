#!/usr/bin/env python3
"""Batch RoNIN evaluation across real/synthetic IMU pairs.

Compares original vs LDM-generated IMU on a single RoNIN checkpoint.
Which LDM runs execute depends on which checkpoints you pass:

  --ldm_ckpt
      1) eval_ronin_ldm.py on real HDF5  (real_imu vs noise_generated)
      2) eval_ronin_ldm.py on synthetic parquet + real cond
         (synthetic_imu vs strength_generated)

  --ldm_sim_cond_ckpt
      3) eval_ronin_ldm_sim_cond.py on pair_dir
         (real IMU vs sim_cond_generated)

Pass either flag, or both. Then aggregates ATE/RTE into per-trajectory
overlays and dataset-level summary plots.
"""

import argparse
import csv
import json
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def discover_pairs_from_manifest(test_root):
    """Read manifest.csv and return list of pair_id strings (status=generated)."""
    manifest = osp.join(test_root, "manifest.csv")
    if not osp.isfile(manifest):
        return None
    pairs = []
    with open(manifest, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "").strip() == "generated":
                pairs.append(row["pair_id"].strip())
    return sorted(pairs)


def discover_pairs_recursive(test_root):
    """Fallback: find directories containing both real.hdf5 and synthetic.parquet."""
    pairs = []
    for root, _dirs, files in os.walk(test_root):
        if "real.hdf5" in files and "synthetic.parquet" in files:
            rel = osp.relpath(root, test_root)
            pairs.append(rel)
    return sorted(pairs)


def discover_pairs(test_root):
    """Return sorted list of pair_id strings."""
    pairs = discover_pairs_from_manifest(test_root)
    if pairs is None:
        pairs = discover_pairs_recursive(test_root)
    return pairs


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

RUN_NAMES = ["real_ldm", "synthetic_ldm", "sim_cond_ldm"]
METHOD_LABELS = [
    "real_imu",
    "noise_generated",
    "synthetic_imu",
    "strength_generated",
    "sim_cond_generated",
]


def build_commands(pair_dir, args, outdir_base):
    """Build eval subprocesses for a pair, based on which LDM ckpts were given.

    Returns list of (run_name, cmd_list, outdir) tuples.
    """
    real_hdf5 = osp.join(pair_dir, "real.hdf5")
    synthetic_pq = osp.join(pair_dir, "synthetic.parquet")

    commands = []

    if args.ldm_ckpt:
        # Run 1: eval_ronin_ldm.py on real HDF5 (original vs traj-cond gen)
        run1_out = osp.join(outdir_base, "real_ldm")
        cmd1 = [
            sys.executable, "scripts/eval_ronin_ldm.py",
            "--input", real_hdf5,
            "--dataset", "hybrid",
            "--ldm_ckpt", args.ldm_ckpt,
            "--first_stage_ckpt", args.vae_ckpt,
            "--ronin_ckpt", args.ronin_ckpt,
            "--stats", args.stats,
            "--outdir", run1_out,
            "--save_trajectories",
            "--hdf5_out", "",
            "--parquet_out", "",
            "--seed", str(args.seed),
            "--ddim_steps", str(args.ddim_steps),
            "--ddim_eta", str(args.ddim_eta),
        ]
        if args.ldm_config:
            cmd1 += ["--ldm_config", args.ldm_config]
        if args.cpu:
            cmd1.append("--cpu")
        cmd1 += ["--rte_delta_sec", str(args.rte_delta_sec)]
        commands.append(("real_ldm", cmd1, run1_out))

        # Run 2: eval_ronin_ldm.py on synthetic parquet with real cond
        if osp.isfile(synthetic_pq):
            run2_out = osp.join(outdir_base, "synthetic_ldm")
            cmd2 = [
                sys.executable, "scripts/eval_ronin_ldm.py",
                "--imu_input", synthetic_pq,
                "--cond_input", real_hdf5,
                "--ldm_ckpt", args.ldm_ckpt,
                "--first_stage_ckpt", args.vae_ckpt,
                "--ronin_ckpt", args.ronin_ckpt,
                "--stats", args.stats,
                "--outdir", run2_out,
                "--strength", str(args.strength),
                "--trim_to_match",
                "--save_trajectories",
                "--hdf5_out", "",
                "--parquet_out", "",
                "--seed", str(args.seed),
                "--ddim_steps", str(args.ddim_steps),
                "--ddim_eta", str(args.ddim_eta),
            ]
            if args.ldm_config:
                cmd2 += ["--ldm_config", args.ldm_config]
            if args.cpu:
                cmd2.append("--cpu")
            cmd2 += ["--rte_delta_sec", str(args.rte_delta_sec)]
            commands.append(("synthetic_ldm", cmd2, run2_out))
        else:
            print(f"  [skip] synthetic_ldm — no parquet at {synthetic_pq}")

    if args.ldm_sim_cond_ckpt:
        # Run 3: eval_ronin_ldm_sim_cond.py (original vs sim-cond gen)
        run3_out = osp.join(outdir_base, "sim_cond_ldm")
        cmd3 = [
            sys.executable, "scripts/eval_ronin_ldm_sim_cond.py",
            "--pair_dir", pair_dir,
            "--ldm_ckpt", args.ldm_sim_cond_ckpt,
            "--first_stage_ckpt", args.vae_ckpt,
            "--ronin_ckpt", args.ronin_ckpt,
            "--stats", args.stats_sim_cond,
            "--outdir", run3_out,
            "--trim_to_match",
            "--save_trajectories",
            "--hdf5_out", "",
            "--parquet_out", "",
            "--seed", str(args.seed),
            "--ddim_steps", str(args.ddim_steps),
            "--ddim_eta", str(args.ddim_eta),
        ]
        if args.ldm_sim_cond_config:
            cmd3 += ["--ldm_config", args.ldm_sim_cond_config]
        if args.cpu:
            cmd3.append("--cpu")
        cmd3 += ["--rte_delta_sec", str(args.rte_delta_sec)]
        commands.append(("sim_cond_ldm", cmd3, run3_out))

    return commands


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def run_is_complete(outdir):
    """Check if a run has both metrics.json and trajectories.npz."""
    return (
        osp.isfile(osp.join(outdir, "metrics.json"))
        and osp.isfile(osp.join(outdir, "trajectories.npz"))
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_command(run_name, cmd, outdir, pair_id, dry_run=False, force=False):
    """Execute a single eval subprocess. Returns (success, metrics_dict|None)."""
    if not force and run_is_complete(outdir):
        print(f"  [SKIP] {pair_id}/{run_name} — already complete")
        with open(osp.join(outdir, "metrics.json")) as f:
            return True, json.load(f)

    if dry_run:
        print(f"  [DRY-RUN] {pair_id}/{run_name}")
        print(f"    cmd: {' '.join(cmd)}")
        return True, None

    os.makedirs(outdir, exist_ok=True)
    log_path = osp.join(outdir, "run.log")
    print(f"  [RUN] {pair_id}/{run_name} ...")

    with open(log_path, "w") as log_f:
        log_f.write(f"CMD: {' '.join(cmd)}\n\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=osp.dirname(osp.dirname(osp.abspath(__file__))),
        )

    if proc.returncode != 0:
        print(f"  [FAIL] {pair_id}/{run_name} — exit code {proc.returncode} (see {log_path})")
        return False, None

    metrics_path = osp.join(outdir, "metrics.json")
    if osp.isfile(metrics_path):
        with open(metrics_path) as f:
            return True, json.load(f)
    return True, None


# ---------------------------------------------------------------------------
# Per-trajectory comparison plots
# ---------------------------------------------------------------------------

def _metric_block(block):
    return {
        "ate": block["ate"],
        "rte": block["rte"],
        "rte_short": block.get("rte_short", block["rte"]),
    }


def _format_rte_sec(sec):
    if float(sec).is_integer():
        return f"{int(sec)}s"
    return f"{sec:g}s"


def plot_trajectory_overlay(pair_outdir, pair_id):
    """Create five-method trajectory overlay and ATE/RTE bar chart for one pair."""
    traj_data = {}
    metrics_data = {}

    # Load from the three run subdirs
    for run_name in RUN_NAMES:
        run_dir = osp.join(pair_outdir, run_name)
        npz_path = osp.join(run_dir, "trajectories.npz")
        met_path = osp.join(run_dir, "metrics.json")
        if not osp.isfile(npz_path) or not osp.isfile(met_path):
            continue
        traj_data[run_name] = dict(np.load(npz_path))
        with open(met_path) as f:
            metrics_data[run_name] = json.load(f)

    if not traj_data:
        return None

    rte_delta_sec = next(
        (m.get("rte_delta_sec", 10.0) for m in metrics_data.values()),
        10.0,
    )
    short = _format_rte_sec(rte_delta_sec)

    # Map run outputs to the 5 method labels
    methods = {}
    if "real_ldm" in traj_data:
        d = traj_data["real_ldm"]
        m = metrics_data["real_ldm"]
        methods["real_imu"] = {
            "pos_pred": d["pos_pred_orig"], "pos_gt": d["pos_gt"],
            **_metric_block(m["original"]),
        }
        methods["noise_generated"] = {
            "pos_pred": d["pos_pred_gen"], "pos_gt": d["pos_gt"],
            **_metric_block(m["ldm_gen"]),
        }
    if "synthetic_ldm" in traj_data:
        d = traj_data["synthetic_ldm"]
        m = metrics_data["synthetic_ldm"]
        methods["synthetic_imu"] = {
            "pos_pred": d["pos_pred_orig"], "pos_gt": d["pos_gt"],
            **_metric_block(m["original"]),
        }
        methods["strength_generated"] = {
            "pos_pred": d["pos_pred_gen"], "pos_gt": d["pos_gt"],
            **_metric_block(m["ldm_gen"]),
        }
    if "sim_cond_ldm" in traj_data:
        d = traj_data["sim_cond_ldm"]
        m = metrics_data["sim_cond_ldm"]
        # Original IMU lives on this run when traj-LDM (real_ldm) was skipped.
        if "real_imu" not in methods:
            methods["real_imu"] = {
                "pos_pred": d["pos_pred_orig"], "pos_gt": d["pos_gt"],
                **_metric_block(m["original"]),
            }
        methods["sim_cond_generated"] = {
            "pos_pred": d["pos_pred_gen"], "pos_gt": d["pos_gt"],
            **_metric_block(m["ldm_gen"]),
        }

    if not methods:
        return None

    # --- Trajectory overlay ---
    colors = {
        "real_imu": "blue",
        "noise_generated": "red",
        "synthetic_imu": "green",
        "strength_generated": "orange",
        "sim_cond_generated": "purple",
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gt = list(methods.values())[0]["pos_gt"]
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2.0, label="GT", zorder=10)
    for label in METHOD_LABELS:
        if label not in methods:
            continue
        m = methods[label]
        ax.plot(
            m["pos_pred"][:, 0], m["pos_pred"][:, 1],
            color=colors[label], lw=1.0, alpha=0.8,
            label=(f"{label} (ATE={m['ate']:.3f}, "
                   f"RTE_60s={m['rte']:.3f}, RTE_{short}={m['rte_short']:.3f})"),
        )
    ax.set_title(f"Trajectory Comparison — {pair_id}", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    overlay_path = osp.join(pair_outdir, "five_method_trajectory_overlay.png")
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)

    # --- ATE/RTE grouped bar chart ---
    present_labels = [l for l in METHOD_LABELS if l in methods]
    ates = [methods[l]["ate"] for l in present_labels]
    rtes = [methods[l]["rte"] for l in present_labels]
    rtes_short = [methods[l]["rte_short"] for l in present_labels]

    x = np.arange(len(present_labels))
    width = 0.25
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    bars1 = ax2.bar(x - width, ates, width, label="ATE", color="steelblue")
    bars2 = ax2.bar(x, rtes, width, label="RTE 60s", color="salmon")
    bars3 = ax2.bar(x + width, rtes_short, width, label=f"RTE {short}", color="seagreen")
    ax2.set_xticks(x)
    ax2.set_xticklabels(present_labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Error (m)")
    ax2.set_title(f"ATE / RTE Comparison — {pair_id}", fontsize=11)
    ax2.legend()
    ax2.bar_label(bars1, fmt="%.3f", fontsize=6, padding=2)
    ax2.bar_label(bars2, fmt="%.3f", fontsize=6, padding=2)
    ax2.bar_label(bars3, fmt="%.3f", fontsize=6, padding=2)
    fig2.tight_layout()
    bar_path = osp.join(pair_outdir, "five_method_ate_rte.png")
    fig2.savefig(bar_path, dpi=150)
    plt.close(fig2)

    return {
        label: {
            "ate": methods[label]["ate"],
            "rte": methods[label]["rte"],
            "rte_short": methods[label]["rte_short"],
        }
        for label in present_labels
    }


# ---------------------------------------------------------------------------
# Dataset-level aggregate plots
# ---------------------------------------------------------------------------

def plot_aggregate(all_results, outdir):
    """Create mean ATE/RTE bar chart across all trajectories and save summary CSV/JSON."""
    if not all_results:
        print("No results to aggregate.")
        return

    # Collect per-method lists
    per_method = {l: {"ate": [], "rte": [], "rte_short": []} for l in METHOD_LABELS}
    for pair_id, methods in all_results.items():
        for label, vals in methods.items():
            per_method[label]["ate"].append(vals["ate"])
            per_method[label]["rte"].append(vals["rte"])
            per_method[label]["rte_short"].append(vals.get("rte_short", vals["rte"]))

    present = [l for l in METHOD_LABELS if per_method[l]["ate"]]
    if not present:
        print("No methods have results to aggregate.")
        return

    mean_ate = [np.mean(per_method[l]["ate"]) for l in present]
    std_ate = [np.std(per_method[l]["ate"]) for l in present]
    mean_rte = [np.mean(per_method[l]["rte"]) for l in present]
    std_rte = [np.std(per_method[l]["rte"]) for l in present]
    mean_rte_short = [np.mean(per_method[l]["rte_short"]) for l in present]
    std_rte_short = [np.std(per_method[l]["rte_short"]) for l in present]

    x = np.arange(len(present))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width, mean_ate, width, yerr=std_ate,
                   label="ATE", color="steelblue", capsize=3)
    bars2 = ax.bar(x, mean_rte, width, yerr=std_rte,
                   label="RTE 60s", color="salmon", capsize=3)
    bars3 = ax.bar(x + width, mean_rte_short, width, yerr=std_rte_short,
                   label="RTE short", color="seagreen", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Error (m)")
    ax.set_title(f"Mean ATE / RTE Across {len(all_results)} Trajectories", fontsize=11)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", fontsize=6, padding=2)
    ax.bar_label(bars2, fmt="%.3f", fontsize=6, padding=2)
    ax.bar_label(bars3, fmt="%.3f", fontsize=6, padding=2)
    fig.tight_layout()
    fig.savefig(osp.join(outdir, "aggregate_ate_rte.png"), dpi=150)
    plt.close(fig)

    # Summary JSON
    summary = {
        "n_trajectories": len(all_results),
        "methods": {},
        "per_trajectory": all_results,
    }
    for l in present:
        summary["methods"][l] = {
            "mean_ate": float(np.mean(per_method[l]["ate"])),
            "std_ate": float(np.std(per_method[l]["ate"])),
            "mean_rte": float(np.mean(per_method[l]["rte"])),
            "std_rte": float(np.std(per_method[l]["rte"])),
            "mean_rte_short": float(np.mean(per_method[l]["rte_short"])),
            "std_rte_short": float(np.std(per_method[l]["rte_short"])),
            "n": len(per_method[l]["ate"]),
        }

    with open(osp.join(outdir, "aggregate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Summary CSV
    csv_path = osp.join(outdir, "aggregate_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "mean_ate", "std_ate", "mean_rte", "std_rte",
            "mean_rte_short", "std_rte_short", "n",
        ])
        for l in present:
            s = summary["methods"][l]
            writer.writerow([
                l, f"{s['mean_ate']:.4f}", f"{s['std_ate']:.4f}",
                f"{s['mean_rte']:.4f}", f"{s['std_rte']:.4f}",
                f"{s['mean_rte_short']:.4f}", f"{s['std_rte_short']:.4f}", s["n"],
            ])

    print(f"\nAggregate results ({len(all_results)} trajectories):")
    for l in present:
        s = summary["methods"][l]
        print(f"  {l:25s}  ATE={s['mean_ate']:.4f}±{s['std_ate']:.4f}  "
              f"RTE_60s={s['mean_rte']:.4f}±{s['std_rte']:.4f}  "
              f"RTE_short={s['mean_rte_short']:.4f}±{s['std_rte_short']:.4f}  "
              f"(n={s['n']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch RoNIN evaluation across real/synthetic IMU pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test_root", type=str, required=True,
                        help="Root directory of test pairs (e.g. data/real_sim_imu_pairs_test)")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="VAE first-stage checkpoint")
    parser.add_argument("--ronin_ckpt", type=str, required=True,
                        help="RoNIN checkpoint")
    parser.add_argument("--ldm_ckpt", type=str, default=None,
                        help="Trajectory-conditioned LDM (eval_ronin_ldm.py). "
                             "Omit to skip those runs.")
    parser.add_argument("--ldm_sim_cond_ckpt", type=str, default=None,
                        help="Sim-conditioned LDM (eval_ronin_ldm_sim_cond.py). "
                             "Omit to skip that run.")
    parser.add_argument("--stats", type=str, default=None,
                        help="stats.pt for --ldm_ckpt (required if that ckpt is set)")
    parser.add_argument("--stats_sim_cond", type=str, default=None,
                        help="stats.pt for --ldm_sim_cond_ckpt (required if that ckpt is set)")
    parser.add_argument("--outdir", type=str, default="outputs/batch_eval",
                        help="Root output directory")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Strength for the synthetic-parquet LDM run")
    parser.add_argument("--ldm_config", type=str, default=None,
                        help="Override LDM config for eval_ronin_ldm.py")
    parser.add_argument("--ldm_sim_cond_config", type=str, default=None,
                        help="Override LDM config for eval_ronin_ldm_sim_cond.py")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument(
        "--rte_delta_sec", type=float, default=10.0,
        help="Second RTE window in seconds forwarded to eval subprocesses "
             "(in addition to the default 60s RTE)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip eval runs, only regenerate comparison plots from existing results")
    args = parser.parse_args()

    if not args.ldm_ckpt and not args.ldm_sim_cond_ckpt:
        parser.error("provide --ldm_ckpt and/or --ldm_sim_cond_ckpt")
    if args.ldm_ckpt and not args.stats:
        parser.error("--stats is required when --ldm_ckpt is set")
    if args.ldm_sim_cond_ckpt and not args.stats_sim_cond:
        parser.error("--stats_sim_cond is required when --ldm_sim_cond_ckpt is set")

    enabled = []
    if args.ldm_ckpt:
        enabled.append("traj LDM (--ldm_ckpt)")
    if args.ldm_sim_cond_ckpt:
        enabled.append("sim-cond LDM (--ldm_sim_cond_ckpt)")
    print("Enabled runs: " + ", ".join(enabled))

    test_root = osp.abspath(args.test_root)
    outdir = osp.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Discover pairs
    pairs = discover_pairs(test_root)
    print(f"Found {len(pairs)} pairs in {test_root}")
    if not pairs:
        print("No pairs found. Exiting.")
        return

    # Run evaluations
    failed = []
    all_results = {}

    for i, pair_id in enumerate(pairs, 1):
        pair_dir = osp.join(test_root, pair_id)
        pair_name = pair_id.replace("/", "_")
        pair_outdir = osp.join(outdir, pair_id)

        print(f"\n{'='*70}")
        print(f"[{i}/{len(pairs)}] {pair_id}")
        print(f"{'='*70}")

        if not args.plots_only:
            commands = build_commands(pair_dir, args, pair_outdir)
            for run_name, cmd, run_outdir in commands:
                ok, _metrics = run_command(
                    run_name, cmd, run_outdir, pair_id,
                    dry_run=args.dry_run, force=args.force,
                )
                if not ok:
                    failed.append(f"{pair_id}/{run_name}")

        # Generate per-trajectory comparison plots
        if not args.dry_run:
            result = plot_trajectory_overlay(pair_outdir, pair_id)
            if result:
                all_results[pair_id] = result

    # Aggregate
    if not args.dry_run and all_results:
        print(f"\n{'='*70}")
        print("AGGREGATE RESULTS")
        print(f"{'='*70}")
        plot_aggregate(all_results, outdir)

    # Report failures
    if failed:
        print(f"\n{'='*70}")
        print(f"FAILED RUNS ({len(failed)}):")
        for f_item in failed:
            print(f"  - {f_item}")
        failed_path = osp.join(outdir, "failed_runs.json")
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)

    n_cmds = 0
    if not args.plots_only:
        # Count from the first pair so dry-run / summary match what we will run.
        sample_cmds = build_commands(osp.join(test_root, pairs[0]), args, "/tmp")
        n_cmds = len(sample_cmds)

    if args.dry_run:
        print(f"\n[DRY-RUN] Would execute {len(pairs) * n_cmds} commands for {len(pairs)} pairs "
              f"({n_cmds} per pair)")
    else:
        print(f"\nDone. Results in {outdir}")


if __name__ == "__main__":
    main()
