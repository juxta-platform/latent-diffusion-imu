#!/usr/bin/env python3
"""Benchmark RoNIN trajectory error across five IMU sources on paired data.

For every real/synthetic pair it runs up to three LDM passes, which together
produce the five methods being compared:

    --ldm_ckpt            real.hdf5                -> real_imu, noise_generated
                          synthetic.parquet + real trajectory
                                                   -> synthetic_imu, strength_generated
    --ldm_sim_cond_ckpt   pair dir                 -> real_imu, sim_cond_generated

Pass either checkpoint or both. Results land in per-pair subdirectories with a
five-method trajectory overlay and ATE/RTE bars, plus a dataset-level
aggregate. Unlike the script this replaces, the passes run in process and share
one loaded VAE, LDM and RoNIN instead of reloading them for every pair.

Example:
    python scripts/compare_ldm_modes.py \\
        --input_dir data/real_sim_paired_data_ldm \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --ldm_ckpt logs/ldm_1d/ldm_world_full_dataset_velbugfix/checkpoints/last.ckpt \\
        --ldm_stats data/hdf5_data_processed_vae_training/stats.pt \\
        --ldm_sim_cond_ckpt logs/ldm_1d_sim_cond/ldm_world_sim_cond/checkpoints/best-019.ckpt \\
        --ldm_sim_cond_stats data/real_sim_paired_data_processed_ldm_training/stats.pt \\
        --ronin_ckpt ../juxta-ronin/models/base_from_scratch_198.pt \\
        --outdir outputs/ldm_mode_comparison
"""

import argparse
import os
import os.path as osp


import numpy as np

from ldm.evaluation import args as eval_args
from ldm.evaluation import (
    batch, generate, metrics, models, paths, plots, report, ronin, sequences, stats,
)
from ldm.evaluation.constants import SAMPLE_RATE
from ldm.evaluation.plots import METHOD_LABELS

# Each pass writes one subdirectory per pair and contributes two of the five
# methods: the IMU it started from, and what the LDM generated from it.
RUNS = {
    "real_ldm": {"bundle": "traj", "methods": ("real_imu", "noise_generated")},
    "synthetic_ldm": {"bundle": "traj", "methods": ("synthetic_imu", "strength_generated")},
    "sim_cond_ldm": {"bundle": "sim_cond", "methods": ("real_imu", "sim_cond_generated")},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare RoNIN trajectory error across real, synthetic and "
                    "LDM-generated IMU over a paired dataset",
    )
    eval_args.add_input_args(parser, dataset_dir=False, pair=True)
    eval_args.add_vae_args(parser, with_stats=False)
    group = parser.add_argument_group("LDM (pass at least one checkpoint)")
    group.add_argument("--ldm_ckpt", type=str, default=None,
                       help="Trajectory-conditioned LDM checkpoint")
    group.add_argument("--ldm_config", type=str, default=None)
    group.add_argument("--ldm_stats", type=str, default=None,
                       help="stats.pt the trajectory-conditioned LDM was trained with")
    group.add_argument("--ldm_sim_cond_ckpt", type=str, default=None,
                       help="Sim-conditioned LDM checkpoint")
    group.add_argument("--ldm_sim_cond_config", type=str, default=None)
    group.add_argument("--ldm_sim_cond_stats", type=str, default=None,
                       help="stats.pt the sim-conditioned LDM was trained with")
    eval_args.add_ronin_args(parser, required=True)
    eval_args.add_sampling_args(parser)
    parser.set_defaults(strength=0.5)
    eval_args.add_output_args(parser, default_outdir="outputs/ldm_mode_comparison",
                              plots=False)
    eval_args.add_batch_args(parser)
    return parser


def validate(parser, args):
    if not 0 <= args.strength <= 1:
        parser.error("--strength must be in [0, 1]")
    if args.ldm_ckpt is None and args.ldm_sim_cond_ckpt is None:
        parser.error("Provide --ldm_ckpt, --ldm_sim_cond_ckpt, or both")
    if args.ldm_ckpt and args.ldm_stats is None:
        parser.error("--ldm_ckpt needs --ldm_stats")
    if args.ldm_sim_cond_ckpt and args.ldm_sim_cond_stats is None:
        parser.error("--ldm_sim_cond_ckpt needs --ldm_sim_cond_stats")
    return eval_args.validate_inputs(parser, args, allow_dataset_dir=False)


def plan_runs(recording, args):
    """Which passes apply to one pair, given the checkpoints and files present."""
    planned = []
    if args.ldm_ckpt:
        planned.append(("real_ldm", dict(
            imu_path=recording.path, cond_path=recording.path, sim_path=None,
            strength=1.0, trim=False,
        )))
        if recording.sim_path:
            planned.append(("synthetic_ldm", dict(
                imu_path=recording.sim_path, cond_path=recording.path, sim_path=None,
                strength=args.strength, trim=True,
            )))
        else:
            print(f"  [skip] synthetic_ldm — no synthetic parquet for {recording.name}")
    if args.ldm_sim_cond_ckpt:
        if recording.sim_path:
            planned.append(("sim_cond_ldm", dict(
                imu_path=recording.path, cond_path=recording.path,
                sim_path=recording.sim_path, strength=1.0, trim=True,
            )))
        else:
            print(f"  [skip] sim_cond_ldm — no synthetic parquet for {recording.name}")
    return planned


def collect_methods(pair_outdir):
    """Map the passes that completed for one pair onto the five method labels.

    Reads the ``metrics.json`` / ``trajectories.npz`` each pass leaves behind, so
    it also serves ``--plots_only``.
    """
    methods = {}
    rte_delta_sec = 10.0
    for run_name, spec in RUNS.items():
        run_dir = osp.join(pair_outdir, run_name)
        npz_path = osp.join(run_dir, "trajectories.npz")
        metrics_path = osp.join(run_dir, "metrics.json")
        if not (osp.isfile(npz_path) and osp.isfile(metrics_path)):
            continue
        arrays = dict(np.load(npz_path))
        payload = report.read_json(metrics_path)
        rte_delta_sec = payload.get("rte_delta_sec", rte_delta_sec)

        source_label, generated_label = spec["methods"]
        # real_imu is shared: the first pass that produced it wins.
        if source_label not in methods:
            methods[source_label] = {
                "pos_pred": arrays["pos_pred_orig"], "pos_gt": arrays["pos_gt"],
                **_metric_block(payload["original"]),
            }
        methods[generated_label] = {
            "pos_pred": arrays["pos_pred_gen"], "pos_gt": arrays["pos_gt"],
            **_metric_block(payload["ldm_gen"]),
        }
    return methods, rte_delta_sec


def _metric_block(block):
    return {
        "ate": block["ate"],
        "rte": block["rte"],
        "rte_short": block.get("rte_short", block["rte"]),
    }


# ---------------------------------------------------------------------------
# Main flow, in call order
# ---------------------------------------------------------------------------

def load_models(args, device):
    """Load whichever LDMs were requested, plus the shared RoNIN network."""
    bundles = {}
    for key, ckpt, config, stats_path, mode in (
        ("traj", args.ldm_ckpt, args.ldm_config, args.ldm_stats, "traj"),
        ("sim_cond", args.ldm_sim_cond_ckpt, args.ldm_sim_cond_config,
         args.ldm_sim_cond_stats, "sim_cond"),
    ):
        if not ckpt:
            continue
        config = config or models.default_ldm_config(mode)
        print(f"Loading {mode} LDM from {ckpt}\n  config: {config}")
        ldm, sampler, _ = models.load_ldm(
            config, ckpt, device, vae_ckpt=args.vae_ckpt, scale_factor=args.scale_factor,
        )
        bundles[key] = {
            "ldm": ldm,
            "sampler": sampler,
            "stats": stats.require_velocity(stats.load_stats(stats_path)),
            "ckpt": osp.abspath(ckpt),
            "config": config,
        }
        print(f"  scale_factor={float(ldm.scale_factor)}")

    print("Loading RoNIN ...")
    ronin_net = models.load_ronin(
        args.ronin_ckpt, device, arch=args.ronin_arch,
        window_size=args.ronin_window, use_3d=args.ronin_3d,
        ronin_root=args.ronin_root,
    )
    return bundles, ronin_net


def run_pass(run_name, spec, bundle, ronin_net, run_outdir, device, args):
    """One LDM pass: generate, run RoNIN on both IMUs, write metrics and arrays."""
    generate.seed_everything(args.seed)
    imu_ds = ronin.load_strided_dataset(
        spec["imu_path"], None, step_size=args.ronin_step, window_size=args.ronin_window,
    )
    if osp.abspath(spec["cond_path"]) != osp.abspath(spec["imu_path"]):
        cond_ds = ronin.load_strided_dataset(
            spec["cond_path"], None, step_size=args.ronin_step,
            window_size=args.ronin_window,
        )
        features, ts, gt_pos, _ts_imu, n = ronin.align_imu_and_cond(
            imu_ds, cond_ds, trim=spec["trim"],
        )
    else:
        features = imu_ds.features[0]
        ts = imu_ds.ts_full[0]
        gt_pos = imu_ds.gt_pos_full[0]
        n = min(features.shape[0], len(ts), len(gt_pos))
        if n < features.shape[0]:
            ronin.trim_dataset_to(imu_ds, n)
            features, ts, gt_pos = imu_ds.features[0], ts[:n], gt_pos[:n]

    sim_features = None
    if spec["sim_path"] is not None:
        sim_features = sequences.load_sim_on_timeline(spec["sim_path"], ts)
        if sim_features.shape[0] < n:
            n = sim_features.shape[0]
            ronin.trim_dataset_to(imu_ds, n)
            features, ts, gt_pos = imu_ds.features[0], ts[:n], gt_pos[:n]
        sim_features = sim_features[:n]

    print(f"    {run_name}: {n} samples ({n / SAMPLE_RATE:.1f}s), "
          f"{len(imu_ds)} RoNIN windows")
    features_gen = generate.ldm_generate(
        features, ts, gt_pos, bundle["ldm"], bundle["sampler"], bundle["stats"], device,
        sim_features=sim_features, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
        use_ema=not args.no_ema, strength=spec["strength"],
        initial_features=features[:, [3, 4, 5, 0, 1, 2]] if spec["strength"] < 1 else None,
    )

    rte_delta = int(round(args.rte_delta_sec * SAMPLE_RATE))
    res_orig = ronin.run_ronin_pipeline(
        ronin_net, imu_ds, device, use_3d=args.ronin_3d, rte_delta=rte_delta,
    )
    res_gen = ronin.run_ronin_pipeline(
        ronin_net, ronin.with_features(imu_ds, features_gen), device,
        use_3d=args.ronin_3d, rte_delta=rte_delta,
    )

    payload = {
        "run": run_name,
        "imu_input": osp.abspath(spec["imu_path"]),
        "cond_input": osp.abspath(spec["cond_path"]),
        "sim_input": None if spec["sim_path"] is None else osp.abspath(spec["sim_path"]),
        "ldm_ckpt": bundle["ckpt"],
        "ronin_ckpt": osp.abspath(args.ronin_ckpt),
        "n_samples": int(n),
        "strength": spec["strength"],
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
        "seed": args.seed,
        "rte_delta_sec": args.rte_delta_sec,
        "original": metrics.trajectory_metrics(res_orig),
        "ldm_gen": metrics.trajectory_metrics(res_gen),
        "delta": metrics.trajectory_delta(
            metrics.trajectory_metrics(res_gen), metrics.trajectory_metrics(res_orig),
        ),
    }
    report.write_json(payload, run_outdir, "metrics.json")
    report.write_trajectories(
        run_outdir, res_orig["pos_gt"], res_orig["pos_pred"], res_gen["pos_pred"],
    )
    print(f"    {run_name}: ATE {res_orig['ate']:.4f} -> {res_gen['ate']:.4f}")
    return payload


def compare_pair(recording, pair_outdir, bundles, ronin_net, device, args):
    """Run every applicable pass for one pair and plot the five-method figures."""
    for run_name, spec in plan_runs(recording, args):
        run_outdir = osp.join(pair_outdir, run_name)
        if not args.force and batch.is_complete(run_outdir, ("trajectories.npz",)):
            print(f"    [skip] {run_name} — already complete")
            continue
        os.makedirs(run_outdir, exist_ok=True)
        run_pass(run_name, spec, bundles[RUNS[run_name]["bundle"]], ronin_net,
                 run_outdir, device, args)

    return plot_pair(recording, pair_outdir)


def plot_pair(recording, pair_outdir, _result=None):
    """Five-method overlay and ATE/RTE bars for one pair, from what it wrote."""
    methods, rte_delta_sec = collect_methods(pair_outdir)
    if not methods:
        return None
    plots.plot_method_overlay(methods, pair_outdir, recording.name,
                              rte_delta_sec=rte_delta_sec)
    plots.plot_method_bars(methods, pair_outdir, recording.name,
                           rte_delta_sec=rte_delta_sec)
    summary = {
        "pair": recording.name,
        "rte_delta_sec": rte_delta_sec,
        "methods": {
            label: {key: methods[label][key] for key in ("ate", "rte", "rte_short")}
            for label in METHOD_LABELS if label in methods
        },
    }
    report.write_json(summary, pair_outdir, "metrics.json")
    return summary


def aggregate(results, failures, args):
    """Dataset-level mean ATE/RTE per method, as a figure, JSON and CSV."""
    per_method = {label: {"ate": [], "rte": [], "rte_short": []}
                  for label in METHOD_LABELS}
    per_pair = {}
    for name, summary in results:
        per_pair[name] = summary["methods"]
        for label, values in summary["methods"].items():
            for key in ("ate", "rte", "rte_short"):
                per_method[label][key].append(values[key])

    present = [label for label in METHOD_LABELS if per_method[label]["ate"]]
    if not present:
        print("No methods have results to aggregate.")
        return None

    plots.plot_aggregate_bars(per_method, len(per_pair), args.outdir)
    summary = {
        "input_dir": osp.abspath(args.input_dir or args.input),
        "n_trajectories": len(per_pair),
        "n_failures": len(failures),
        "methods": {
            label: {
                "mean_ate": float(np.mean(per_method[label]["ate"])),
                "std_ate": float(np.std(per_method[label]["ate"])),
                "mean_rte": float(np.mean(per_method[label]["rte"])),
                "std_rte": float(np.std(per_method[label]["rte"])),
                "mean_rte_short": float(np.mean(per_method[label]["rte_short"])),
                "std_rte_short": float(np.std(per_method[label]["rte_short"])),
                "n": len(per_method[label]["ate"]),
            }
            for label in present
        },
        "per_trajectory": per_pair,
        "failures": failures,
    }
    report.write_json(summary, args.outdir, "aggregate_summary.json")
    report.write_csv(
        [
            [label] + [f"{summary['methods'][label][key]:.4f}" for key in (
                "mean_ate", "std_ate", "mean_rte", "std_rte",
                "mean_rte_short", "std_rte_short",
            )] + [summary["methods"][label]["n"]]
            for label in present
        ],
        ["method", "mean_ate", "std_ate", "mean_rte", "std_rte",
         "mean_rte_short", "std_rte_short", "n"],
        args.outdir, "aggregate_summary.csv",
    )

    print(f"\nAggregate results ({len(per_pair)} trajectories):")
    for label in present:
        s = summary["methods"][label]
        print(f"  {label:25s}  ATE={s['mean_ate']:.4f}±{s['std_ate']:.4f}  "
              f"RTE_60s={s['mean_rte']:.4f}±{s['std_rte']:.4f}  "
              f"RTE_short={s['mean_rte_short']:.4f}±{s['std_rte_short']:.4f}  "
              f"(n={s['n']})")
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)

    generate.seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = eval_args.resolve_device(args)
    print(f"Device: {device}")

    if args.input_dir is not None:
        _layout, recordings = paths.resolve_input_dir(args.input_dir, pairs_only=True)
    else:
        recordings = [paths.resolve_input(args.input, sim_path=args.sim_input)]
    print(f"Comparing {len(recordings)} pairs")

    bundles, ronin_net = ({}, None) if args.dry_run or args.plots_only else load_models(args, device)
    results, failures = batch.run_batch(
        recordings, args.outdir,
        lambda item, item_outdir: compare_pair(
            item, item_outdir, bundles, ronin_net, device, args,
        ),
        kind="pair", replot=plot_pair, force=args.force, dry_run=args.dry_run,
        plots_only=args.plots_only, strict=args.strict,
        complete=lambda item, dest: all(
            batch.is_complete(osp.join(dest, name), ("trajectories.npz",))
            for name, _spec in plan_runs(item, args)
        ),
    )
    if args.dry_run:
        return
    aggregate(results, failures, args)
    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
