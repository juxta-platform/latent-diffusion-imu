#!/usr/bin/env python3
"""Evaluate a trained 1D IMU VAE.

Always reports signal-space reconstruction quality (loss, RMSE, MAE, both
standardized and in physical units) with per-window overlays. Passing
``--ronin_ckpt`` additionally runs RoNIN on the original and on the
VAE-reconstructed IMU and reports ATE/RTE plus trajectory plots.

Three input modes, one of which must be given:

    --input PATH            one .hdf5/.parquet, or a pair directory
    --input_dir PATH        a directory of recordings, pair folders or classes
    --dataset_dir PATH      preprocessed .pt windows, with --split

Examples:
    python scripts/eval_vae.py \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --dataset_dir data/hdf5_data_processed_vae_training --split val \\
        --outdir outputs/vae_eval

    python scripts/eval_vae.py \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --vae_stats data/hdf5_data_processed_vae_training/stats.pt \\
        --input_dir data/hdf5_data --ronin_ckpt ../juxta-ronin/models/base.pt \\
        --outdir outputs/vae_eval_dir
"""

import argparse
import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

from ldm.data.imu_dataset import IMUDataset
from ldm.evaluation import args as eval_args
from ldm.evaluation import (
    batch, generate, metrics, models, paths, plots, report, ronin, sequences,
    stats, windows,
)
from ldm.evaluation.constants import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a 1D IMU VAE: reconstruction metrics, overlays and "
                    "optional RoNIN trajectory metrics",
    )
    eval_args.add_input_args(parser)
    eval_args.add_vae_args(parser)
    eval_args.add_ronin_args(parser)
    eval_args.add_windowing_args(parser)
    eval_args.add_frame_args(parser, required=False)
    eval_args.add_output_args(parser, default_outdir="outputs/vae_eval")
    eval_args.add_export_args(parser, imu=False)
    eval_args.add_batch_args(parser)
    parser.add_argument("--sample_posterior", action="store_true",
                        help="Sample the posterior instead of using its mean")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Stop after this many batches (quick smoke runs)")
    parser.add_argument("--logdir", type=str, default=None,
                        help="Lightning log dir with tfevents for training curves "
                             "(default: guessed from --vae_ckpt)")
    return parser


def resolve_training_logdir(logdir, vae_ckpt):
    """Explicit ``--logdir``, else the run directory above the checkpoint."""
    if logdir is not None:
        return logdir
    guess = osp.abspath(osp.join(osp.dirname(vae_ckpt), ".."))
    if osp.isdir(guess) and any("tfevents" in name for name in os.listdir(guess)):
        return guess
    return None


def summarize_recording(recording, source, window_metrics, trajectory=None):
    """Merge the window and trajectory halves of one recording's metrics."""
    merged = {"input": osp.abspath(recording.path), "source": source, **window_metrics}
    if recording.label:
        merged["label"] = recording.label
    if trajectory:
        merged.update(trajectory)
    return merged


# ---------------------------------------------------------------------------
# Main flow, in call order
# ---------------------------------------------------------------------------

def load_models(args, device):
    """Load the VAE, its statistics and - when requested - RoNIN."""
    print("Loading VAE ...")
    vae, config = models.load_vae(args.vae_config, args.vae_ckpt, device)
    args.imu_frame = eval_args.resolve_imu_frame(
        None, args, {"imu_frame": vae.imu_frame}, "the VAE checkpoint",
    )
    vae_stats = stats.resolve_stats(
        args.vae_stats, args.dataset_dir, model=vae, outdir=args.outdir,
        what="vae_stats",
    )
    print(f"  stats: {vae_stats['path']}")

    ronin_net = None
    if args.ronin_ckpt:
        print("Loading RoNIN ...")
        ronin_net = models.load_ronin(
            args.ronin_ckpt, device, arch=args.ronin_arch,
            window_size=args.ronin_window, use_3d=args.ronin_3d,
            ronin_root=args.ronin_root,
        )
    return vae, config, vae_stats, ronin_net


@torch.no_grad()
def evaluate_windows(vae, loader, device, kl_weight, use_posterior_mean,
                     max_batches=None):
    """Reconstruct every window in a loader and accumulate signal-space metrics."""
    totals = {"total_loss": 0.0, "rec_loss": 0.0, "kl_loss": 0.0, "mae": 0.0, "n": 0}
    all_inputs, all_recons = [], []

    for index, item in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        x = item["imu"].float().to(device)
        recon, posterior = vae(x, sample_posterior=not use_posterior_mean)

        rec = torch.nn.functional.mse_loss(recon, x)
        kl = vae._kl_loss(posterior)
        batch_size = x.shape[0]
        totals["rec_loss"] += rec.item() * batch_size
        totals["kl_loss"] += kl.item() * batch_size
        totals["total_loss"] += (rec + kl_weight * kl).item() * batch_size
        totals["mae"] += torch.nn.functional.l1_loss(recon, x).item() * batch_size
        totals["n"] += batch_size

        all_inputs.append(x.cpu())
        all_recons.append(recon.cpu())

    if totals["n"] == 0:
        raise ValueError("No windows were evaluated")

    inputs = torch.cat(all_inputs, dim=0)
    recons = torch.cat(all_recons, dim=0)
    n = totals["n"]
    window_metrics = {
        "n_windows": n,
        "total_loss": totals["total_loss"] / n,
        "rec_loss": totals["rec_loss"] / n,
        "rmse": float(np.sqrt(totals["rec_loss"] / n)),
        "kl_loss": totals["kl_loss"] / n,
        "mae": totals["mae"] / n,
        "use_posterior_mean": use_posterior_mean,
        "kl_weight": kl_weight,
        **metrics.channel_metrics(inputs, recons),
    }
    return window_metrics, inputs, recons


def write_window_outputs(window_metrics, inputs, recons, vae_stats, outdir, args,
                         title):
    """Physical-unit metrics, the sample dump and both overlay figure sets."""
    inputs_phys = windows.inverse_standardize(
        inputs, vae_stats["imu_mean"], vae_stats["imu_std"]
    )
    recons_phys = windows.inverse_standardize(
        recons, vae_stats["imu_mean"], vae_stats["imu_std"]
    )
    window_metrics.update(metrics.physical_metrics(inputs_phys, recons_phys))
    window_metrics["sample_rate"] = args.sample_rate
    report.save_signal_plot_data(outdir, inputs.numpy(), recons.numpy(),
                                 inputs_phys.numpy(), recons_phys.numpy(), args.n_plot)

    n_save = inputs.shape[0] if args.n_plot < 0 else min(args.n_plot, inputs.shape[0])
    os.makedirs(outdir, exist_ok=True)
    torch.save(
        {
            "imu": inputs[:n_save],
            "recon": recons[:n_save],
            "imu_physical": inputs_phys[:n_save],
            "recon_physical": recons_phys[:n_save],
            "source": title,
        },
        osp.join(outdir, "recon_samples.pt"),
    )

    for arrays, dest, suffix in (
        ((inputs, recons), outdir, ""),
        ((inputs_phys, recons_phys), osp.join(outdir, "physical"), " [physical]"),
    ):
        plots.plot_signal_overlays(
            arrays[0].numpy(), arrays[1].numpy(), dest, args.n_plot,
            sample_rate=args.sample_rate, title_prefix=f"{title}{suffix}",
            input_label="input", output_label="recon",
        )
    return window_metrics


def evaluate_trajectory(recording, dataset, recons, outdir, vae_stats,
                        ronin_net, device, args):
    """Score the same reconstruction as the signal metrics, in world frame."""
    decoded = windows.inverse_standardize(
        recons, vae_stats["imu_mean"], vae_stats["imu_std"],
    ).permute(0, 2, 1).reshape(-1, 6).numpy()
    # A max_batches smoke run scores only the evaluated span. On a full run,
    # retain the short trailing remainder as in the original evaluation.
    n = len(dataset.imu) if len(recons) == len(dataset) else len(decoded)
    raw = dataset.imu[:n]
    recon = raw.copy()
    recon[:len(decoded)] = decoded
    raw_world = sequences.imu_to_world(
        recording.path, raw, args.imu_frame, args.world_heading,
    )
    recon_world = sequences.imu_to_world(
        recording.path, recon, args.imu_frame, args.world_heading,
    )
    time = None if dataset.time is None else dataset.time[:n]
    pos = None if dataset.pos is None else dataset.pos[:n]
    dataset_orig = ronin.dataset_from_arrays(
        time, pos, raw_world, args.ronin_step, args.ronin_window,
    )
    dataset_recon = ronin.dataset_from_arrays(
        time, pos, recon_world, args.ronin_step, args.ronin_window,
    )
    block, _results = ronin.compare_against_baseline(
        ronin_net, dataset_orig, dataset_recon, outdir, device, args,
        other_key="vae_recon", other_label="VAE recon",
        title="RoNIN trajectory: Original vs VAE-reconstructed IMU",
    )
    plots.plot_full_sequence_windows(
        dataset_orig.features[0], dataset_recon.features[0], outdir,
        n_plot=args.n_plot, window=int(args.window_sec * SAMPLE_RATE),
        label_a="original", label_b="VAE recon",
    )
    return {**block, "n_samples": int(n)}


def evaluate_recording(recording, outdir, vae, vae_stats, ronin_net, device, args,
                       kl_weight):
    """Window metrics for one recording, plus trajectory metrics when enabled."""
    print(f"  {recording.path}")
    dataset, source = sequences.resolve_window_dataset(
        recording.path, vae_stats, args.imu_frame, window_sec=args.window_sec,
        sample_rate=args.sample_rate, latent_length=args.latent_length,
        already_world=args.already_world,
        parquet_from_world=args.parquet_from_world, world_heading=args.world_heading,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    window_metrics, inputs, recons = evaluate_windows(
        vae, loader, device, kl_weight, not args.sample_posterior, args.max_batches,
    )
    write_window_outputs(
        window_metrics, inputs, recons, vae_stats, outdir, args,
        title=f"VAE recon ({source})",
    )

    trajectory = None
    if ronin_net is not None:
        trajectory = evaluate_trajectory(
            recording, dataset, recons, outdir, vae_stats, ronin_net, device, args,
        )

    result = summarize_recording(recording, source, window_metrics, trajectory)
    result["vae_ckpt"] = osp.abspath(args.vae_ckpt)
    result["imu_frame"] = args.imu_frame
    report.write_json(result, outdir, "metrics.json", what="Metrics")
    return result


def evaluate_dataset_split(args, vae, vae_stats, device, kl_weight):
    """Window metrics over a preprocessed train/val split."""
    dataset = IMUDataset(args.dataset_dir, vae_stats["path"], split=args.split)
    print(f"Evaluating split:{args.split} ({len(dataset)} windows)")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    window_metrics, inputs, recons = evaluate_windows(
        vae, loader, device, kl_weight, not args.sample_posterior, args.max_batches,
    )
    write_window_outputs(
        window_metrics, inputs, recons, vae_stats, args.outdir, args,
        title=f"VAE recon (split:{args.split})",
    )
    result = {
        "dataset_dir": osp.abspath(args.dataset_dir),
        "split": args.split,
        "source": f"split:{args.split}",
        "vae_ckpt": osp.abspath(args.vae_ckpt),
        **window_metrics,
    }
    report.write_json(result, args.outdir, "metrics.json", what="Metrics", echo=True)
    return result


def evaluate_directory(args, vae, vae_stats, ronin_net, device, kl_weight):
    """Run the single-recording eval over every item under ``--input_dir``."""
    layout, recordings = paths.resolve_input_dir(args.input_dir)
    print(f"Found {len(recordings)} recordings under {args.input_dir} ({layout} layout)")

    results, failures = batch.run_batch(
        recordings, args.outdir,
        lambda item, item_outdir: evaluate_recording(
            item, item_outdir, vae, vae_stats, ronin_net, device, args, kl_weight,
        ),
        kind="recording", force=args.force, dry_run=args.dry_run,
        plots_only=args.plots_only, strict=args.strict,
    )
    if args.dry_run:
        return None

    rows = [result for _name, result in results]
    aggregate = {
        "rmse_mean": float(np.mean([r["rmse"] for r in rows])) if rows else None,
        "physical_rmse_mean": (
            float(np.mean([r["physical_rmse"] for r in rows])) if rows else None
        ),
    }
    if ronin_net is not None:
        for side in ("original", "vae_recon"):
            aggregate[side] = {
                f"{key}_mean": metrics.mean_metric(rows, side, key)
                for key in ("ate", "rte", "rte_short")
            }
    return batch.summarize(
        results, failures, args.outdir,
        extra={"input_dir": osp.abspath(args.input_dir), "layout": layout,
               "vae_ckpt": osp.abspath(args.vae_ckpt)},
        aggregate=aggregate,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    source = eval_args.validate_inputs(parser, args)
    eval_args.validate_windowing(parser, args)
    if source == "dataset_dir" and args.ronin_ckpt:
        parser.error("RoNIN needs a continuous recording; use --input or --input_dir")

    if batch.handle_offline(
        args, lambda: paths.resolve_input_dir(args.input_dir),
        lambda _item, dest, result: report.replot_signals(dest, result, args.n_plot),
    ):
        return
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = eval_args.resolve_device(args)
    print(f"Device: {device}")

    vae, config, vae_stats, ronin_net = load_models(args, device)
    kl_weight = float(getattr(vae, "kl_weight", config.model.params.get("kl_weight", 1e-6)))

    logdir = resolve_training_logdir(args.logdir, args.vae_ckpt)
    if logdir is not None:
        plots.plot_vae_training_curves(logdir, args.outdir)

    if source == "dataset_dir":
        evaluate_dataset_split(args, vae, vae_stats, device, kl_weight)
    elif source == "input_dir":
        evaluate_directory(args, vae, vae_stats, ronin_net, device, kl_weight)
    else:
        recording = paths.resolve_input(args.input)
        evaluate_recording(
            recording, args.outdir, vae, vae_stats, ronin_net, device, args, kl_weight,
        )

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
