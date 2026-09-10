#!/usr/bin/env python3
"""Evaluate a trained 1D IMU latent diffusion model.

``--mode traj`` conditions on trajectory velocity alone; ``--mode sim_cond``
additionally conditions on the VAE-encoded synthetic IMU latent and therefore
needs a synthetic parquet beside each recording.

What gets reported depends on the input:

    --input / --input_dir   the recording is generated end to end in
                            non-overlapping windows, giving signal-space
                            metrics, IMU overlays and - with --ronin_ckpt -
                            trajectory ATE/RTE plus optional generated-IMU
                            exports
    --dataset_dir --split   preprocessed windows: diffusion loss, per-window
                            DDIM generation metrics and overlays

Examples:
    python scripts/eval_ldm.py --mode sim_cond \\
        --ldm_ckpt logs/ldm_1d_sim_cond/ldm_world_sim_cond/checkpoints/best-019.ckpt \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --ldm_stats data/real_sim_paired_data_processed_ldm_training/stats.pt \\
        --input data/real_sim_paired_data_ldm/chest/john_chest_ios_corrected \\
        --ronin_ckpt ../juxta-ronin/models/base_from_scratch_198.pt \\
        --outdir outputs/ldm_sim_cond_eval

    python scripts/eval_ldm.py --mode traj \\
        --ldm_ckpt logs/ldm_1d/ldm_world_full_dataset_velbugfix/checkpoints/last.ckpt \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --dataset_dir data/real_sim_paired_data_processed_ldm_training --split val \\
        --outdir outputs/ldm_eval
"""

import argparse
import os
import os.path as osp
from contextlib import nullcontext


import numpy as np
import torch
from torch.utils.data import DataLoader

from ldm.data.imu_dataset import IMUDataset
from ldm.data.paired_imu_dataset import PairedIMUDataset
from ldm.evaluation import args as eval_args
from ldm.evaluation import (
    batch, generate, metrics, models, paths, plots, report, ronin, sequences, stats, windows,
)
from ldm.evaluation.constants import SAMPLE_RATE, WINDOW_SAMPLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a 1D IMU LDM: generation metrics, IMU overlays and "
                    "optional RoNIN trajectory metrics",
    )
    eval_args.add_input_args(parser, pair=True)
    eval_args.add_ldm_args(parser)
    eval_args.add_vae_args(parser, with_stats=False)
    eval_args.add_ronin_args(parser)
    eval_args.add_sampling_args(parser)
    parser.add_argument("--already_world", action="store_true",
                        help="HDF5 source already stores world-frame IMU")
    eval_args.add_windowing_args(parser)
    eval_args.add_output_args(parser, default_outdir="outputs/ldm_eval")
    eval_args.add_export_args(parser)
    eval_args.add_batch_args(parser)
    group = parser.add_argument_group("mixed sources (single input only)")
    group.add_argument("--imu_input", type=str, default=None,
                       help="Synthetic parquet initializer for traj --strength < 1 "
                            "(uses its world-frame columns)")
    group.add_argument("--cond_input", type=str, default=None,
                       help="Conditioning trajectory source (defaults to --input)")
    group.add_argument("--trim_to_match", action="store_true",
                       help="Truncate mismatched IMU / conditioning / synthetic "
                            "lengths to the shortest instead of failing")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Stop after this many batches (quick smoke runs)")
    parser.add_argument("--logdir", type=str, default=None,
                        help="Lightning log dir with tfevents for training curves "
                             "(default: guessed from --ldm_ckpt)")
    return parser


def validate(parser, args):
    """Cross-argument checks, then report which input flag was used."""
    if args.imu_input is not None and args.input is None:
        args.input = args.imu_input
    source = eval_args.validate_inputs(parser, args)
    if source != "input" and (args.imu_input or args.cond_input):
        parser.error("--imu_input / --cond_input only apply to a single --input")
    eval_args.validate_sampling(parser, args)
    eval_args.validate_windowing(parser, args)
    if source != "dataset_dir" and args.sample_rate != SAMPLE_RATE:
        parser.error("Recording LDM evaluation runs at 200 Hz; use --sample_rate 200")
    if source == "dataset_dir" and args.ronin_ckpt:
        parser.error("RoNIN needs continuous recordings; use --input or --input_dir")
    if args.sim_input and source != "input":
        parser.error("--sim_input applies only to a single --input")
    if args.imu_input and args.mode == "sim_cond":
        parser.error("--imu_input applies only to trajectory img2img; use --sim_input for sim_cond")
    return source


def resolve_training_logdir(logdir, ldm_ckpt):
    """Explicit ``--logdir``, else the run directory above the checkpoint."""
    if logdir is not None:
        return logdir
    guess = osp.abspath(osp.join(osp.dirname(ldm_ckpt), ".."))
    if osp.isdir(guess) and any("tfevents" in name for name in os.listdir(guess)):
        return guess
    return None


def resolve_export_path(flag, outdir, stem, suffix):
    """``--save_hdf5``/``--save_parquet`` value into a concrete output path."""
    if flag is None:
        return None
    return flag if flag else osp.join(outdir, f"{stem}_ldm_gen{suffix}")


def _as_model_windows(features, window):
    """Full-length [N, 6] RoNIN-order IMU into [B, 6, W] model-order windows."""
    n_windows = features.shape[0] // window
    keep = n_windows * window
    reordered = np.ascontiguousarray(windows.swap_imu_channels(features[:keep]))
    return torch.from_numpy(reordered).float().reshape(n_windows, window, 6).permute(0, 2, 1)


def window_metrics_from_sequences(features_orig, features_gen, ldm_stats, args, outdir):
    """Signal-space metrics and overlays from two full-length RoNIN-order arrays.

    The arrays are re-split into the model's windows and channel order, so
    ``gen_*`` here means the same thing as on the preprocessed-split path
    (standardized units) and ``physical_*`` reports m/s^2 and rad/s.
    """
    window = int(args.window_sec * args.sample_rate)
    if features_orig.shape[0] < window:
        return {}
    real_phys = _as_model_windows(features_orig, window)
    gen_phys = _as_model_windows(features_gen, window)
    real = windows.standardize(real_phys, ldm_stats["imu_mean"], ldm_stats["imu_std"])
    gen = windows.standardize(gen_phys, ldm_stats["imu_mean"], ldm_stats["imu_std"])

    title = f"LDM generation ({args.mode})"
    plots.plot_signal_overlays(
        real.numpy(), gen.numpy(), outdir, args.n_plot, sample_rate=args.sample_rate,
        title_prefix=title, input_label="real", output_label="generated",
    )
    plots.plot_signal_overlays(
        real_phys.numpy(), gen_phys.numpy(), osp.join(outdir, "physical"), args.n_plot,
        sample_rate=args.sample_rate, title_prefix=f"{title} [physical]",
        input_label="real", output_label="generated",
    )
    report.save_signal_plot_data(outdir, real.numpy(), gen.numpy(),
                                 real_phys.numpy(), gen_phys.numpy(), args.n_plot)
    diff = gen - real
    return {
        "sample_rate": args.sample_rate,
        "gen_mse": diff.pow(2).mean().item(),
        "gen_rmse": float(np.sqrt(diff.pow(2).mean().item())),
        "gen_mae": diff.abs().mean().item(),
        **metrics.channel_metrics(real, gen),
        **metrics.physical_metrics(real_phys, gen_phys),
    }


# ---------------------------------------------------------------------------
# Main flow, in call order
# ---------------------------------------------------------------------------

def load_models(args, device):
    """Load the LDM with its sampler and statistics, plus RoNIN when requested."""
    config_path = eval_args.resolve_ldm_config(args)
    print(f"Loading LDM ({args.mode}) from {args.ldm_ckpt}")
    print(f"  config: {config_path}")
    print(f"  first stage: {args.vae_ckpt}")
    ldm, sampler, _config = models.load_ldm(
        config_path, args.ldm_ckpt, device,
        vae_ckpt=args.vae_ckpt, scale_factor=args.scale_factor,
    )
    ldm_stats = stats.require_velocity(
        stats.resolve_stats(args.ldm_stats, args.dataset_dir, what="ldm_stats"),
        what="ldm_stats",
    )
    print(f"  scale_factor={float(ldm.scale_factor)}, "
          f"use_ema={ldm.use_ema and not args.no_ema}, stats={ldm_stats['path']}")

    ronin_net = None
    if args.ronin_ckpt:
        print("Loading RoNIN ...")
        ronin_net = models.load_ronin(
            args.ronin_ckpt, device, arch=args.ronin_arch,
            window_size=args.ronin_window, use_3d=args.ronin_3d,
            ronin_root=args.ronin_root,
        )
    return ldm, sampler, ldm_stats, ronin_net


@torch.no_grad()
def evaluate_split_windows(ldm, sampler, loader, device, args):
    """Diffusion loss, VAE baseline and DDIM generation over preprocessed windows.

    One pass so a batch's loss and its generation share the same encoded
    posterior sample. Returns (metrics, real, generated) with the two window
    stacks in standardized units.
    """
    shape = windows.latent_shape(ldm, args.latent_length)
    use_ema = not args.no_ema
    totals = {"diff_loss": 0.0, "diff_loss_ema": 0.0, "vae_mse": 0.0, "n": 0}
    all_real, all_gen = [], []

    for index, item in enumerate(loader):
        if args.max_batches is not None and index >= args.max_batches:
            break
        x = item["imu"].float().to(device)
        batch_size = x.shape[0]
        cond = {
            "velocity": item["velocity"].float().to(device),
            "physical_time": item["physical_time"].float().to(device),
        }
        if args.mode == "sim_cond":
            sim = item["sim_imu"].float().to(device)
            cond["sim_latent"] = ldm.scale_factor * ldm.encode_first_stage(sim).mode()

        posterior = ldm.encode_first_stage(x)
        z = ldm.scale_factor * posterior.sample()
        t = torch.randint(0, ldm.num_timesteps, (batch_size,), device=device).long()
        loss = ldm.p_losses(z, t, cond)
        totals["diff_loss"] += loss.item() * batch_size
        if ldm.use_ema:
            with ldm.ema_scope():
                totals["diff_loss_ema"] += ldm.p_losses(z, t, cond).item() * batch_size
        else:
            totals["diff_loss_ema"] += loss.item() * batch_size

        vae_recon = ldm.decode_first_stage(posterior.mode())
        totals["vae_mse"] += torch.nn.functional.mse_loss(vae_recon, x).item() * batch_size
        totals["n"] += batch_size

        context = ldm.ema_scope("eval") if (use_ema and ldm.use_ema) else nullcontext()
        with context:
            if args.strength < 1:
                if "sim_imu" not in item:
                    raise ValueError("Trajectory img2img requires paired windows containing sim_imu")
                z0 = ldm.scale_factor * ldm.encode_first_stage(
                    item["sim_imu"].float().to(device)
                ).mode()
                if args.strength == 0:
                    samples = z0
                else:
                    samples, _ = sampler.sample_img2img(
                        S=args.ddim_steps, x0=z0, conditioning=cond,
                        strength=args.strength, eta=args.ddim_eta, verbose=False,
                    )
            else:
                samples, _ = sampler.sample(
                    S=args.ddim_steps, batch_size=batch_size, shape=shape,
                    conditioning=cond, eta=args.ddim_eta, verbose=False,
                )
            gen = ldm.decode_first_stage(samples / ldm.scale_factor)
        all_real.append(x.cpu())
        all_gen.append(gen.cpu())

    if totals["n"] == 0:
        raise ValueError("No windows were evaluated")
    real = torch.cat(all_real, dim=0)
    gen = torch.cat(all_gen, dim=0)
    diff = gen - real
    n = totals["n"]
    return {
        "n_windows": n,
        "diff_loss": totals["diff_loss"] / n,
        "diff_loss_ema": totals["diff_loss_ema"] / n,
        "vae_mse": totals["vae_mse"] / n,
        "vae_rmse": float(np.sqrt(totals["vae_mse"] / n)),
        "gen_mse": diff.pow(2).mean().item(),
        "gen_rmse": float(np.sqrt(diff.pow(2).mean().item())),
        "gen_mae": diff.abs().mean().item(),
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
        "use_ema": use_ema and bool(ldm.use_ema),
        "scale_factor": float(ldm.scale_factor),
        **metrics.channel_metrics(real, gen),
    }, real, gen


def load_sequences(recording, args, cond_path=None):
    """Load the IMU, conditioning trajectory and synthetic IMU for one recording.

    Returns (dataset, features, ts, gt_pos, ts_imu, sim_features, n_samples).
    ``dataset`` is the RoNIN-shaped strided dataset over the IMU source, already
    trimmed so its ground truth matches the conditioning trajectory.
    """
    imu_path = recording.path
    cond_path = cond_path or imu_path
    mixed = osp.abspath(imu_path) != osp.abspath(cond_path)

    imu_ds = ronin.load_strided_dataset(
        imu_path, recording.dataset_type,
        step_size=args.ronin_step, window_size=args.ronin_window,
        already_world=args.already_world,
    )
    if mixed:
        cond_ds = ronin.load_strided_dataset(
            cond_path, None, step_size=args.ronin_step, window_size=args.ronin_window,
        )
        features, ts, gt_pos, ts_imu, n = ronin.align_imu_and_cond(
            imu_ds, cond_ds, trim=args.trim_to_match,
        )
    else:
        features = imu_ds.features[0]
        ts = imu_ds.ts_full[0]
        gt_pos = imu_ds.gt_pos_full[0]
        n = min(features.shape[0], len(ts), len(gt_pos))
        if n < features.shape[0]:
            print(f"  [info] Truncating features {features.shape[0]} -> {n} "
                  f"to match ts/gt_pos")
            ronin.trim_dataset_to(imu_ds, n)
            features = imu_ds.features[0]
            ts, gt_pos = ts[:n], gt_pos[:n]
        ts_imu = ts

    sim_features = initial_features = None
    if args.mode == "sim_cond" or args.strength < 1:
        sim_path = recording.sim_path
        if args.mode == "traj":
            sim_path = args.imu_input or (imu_path if recording.is_parquet else sim_path)
        if sim_path is None or not str(sim_path).endswith(".parquet"):
            raise ValueError("Synthetic world IMU is required: pass --sim_input, "
                             "--imu_input (traj), or a pair directory")
        synthetic = sequences.load_sim_on_timeline(sim_path, ts)
        if len(synthetic) < n:
            if not args.trim_to_match:
                raise ValueError("Synthetic IMU is shorter; use --trim_to_match")
            n = len(synthetic)
            ronin.trim_dataset_to(imu_ds, n)
            features = imu_ds.features[0]
            ts, gt_pos, ts_imu = ts[:n], gt_pos[:n], ts_imu[:n]
        if args.mode == "sim_cond":
            sim_features = synthetic[:n]
        else:
            initial_features = synthetic[:n]
    print(f"  {n} samples ({n / SAMPLE_RATE:.1f}s), {len(imu_ds)} RoNIN windows")
    return imu_ds, features, ts, gt_pos, ts_imu, sim_features, initial_features, n


def evaluate_trajectory(imu_ds, features_orig, features_gen, outdir, ronin_net,
                        device, args):
    """RoNIN ATE/RTE on the original IMU versus the generated IMU."""
    block, _results = ronin.compare_against_baseline(
        ronin_net, imu_ds, ronin.with_features(imu_ds, features_gen),
        outdir, device, args, other_key="ldm_gen", other_label="LDM gen",
        title=f"RoNIN trajectory: Original vs {args.mode} LDM-generated IMU",
    )
    return block


def export_generated(recording, features_gen, ts_imu, outdir, args, cond_path=None):
    """Write the generated world-frame IMU back into HDF5 and/or parquet copies."""
    exports = {}
    hdf5_out = resolve_export_path(args.save_hdf5, outdir, recording.stem, ".hdf5")
    if hdf5_out is not None:
        template = recording.path if not recording.is_parquet else cond_path
        if template is not None and not template.endswith(".parquet") \
                and osp.isfile(template):
            print("\nWriting generated IMU HDF5 (world-frame synced/acce+gyro) ...")
            exports["save_hdf5"] = osp.abspath(
                report.write_generated_hdf5(
                    template, hdf5_out, features_gen,
                    already_world=args.already_world and not recording.is_parquet,
                )
            )
        else:
            print(f"\n[warn] Skipping --save_hdf5; no HDF5 template for {recording.path}")

    parquet_out = resolve_export_path(args.save_parquet, outdir, recording.stem, ".parquet")
    if parquet_out is not None:
        if not recording.is_parquet:
            print("[warn] Skipping --save_parquet; the IMU source is not a parquet")
        else:
            print("\nWriting generated IMU parquet (world-frame columns) ...")
            exports["save_parquet"] = osp.abspath(report.write_generated_parquet(
                recording.path, parquet_out, features_gen,
                ts_imu[: features_gen.shape[0]],
            ))
    return exports


def evaluate_recording(recording, outdir, ldm, sampler, ldm_stats, ronin_net,
                       device, args, cond_path=None):
    """Generate one recording end to end and score the result."""
    print(f"  {recording.path}")
    generate.seed_everything(args.seed)
    imu_ds, features_orig, ts, gt_pos, ts_imu, sim_features, initial_features, n_samples = load_sequences(
        recording, args, cond_path=cond_path,
    )

    print("\nRunning LDM generation ...")
    features_gen = generate.ldm_generate(
        features_orig, ts, gt_pos, ldm, sampler, ldm_stats, device,
        sim_features=sim_features, initial_features=initial_features,
        ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
        use_ema=not args.no_ema, strength=args.strength,
        window=int(args.window_sec * args.sample_rate), latent_length=args.latent_length,
    )

    result = {
        "input": osp.abspath(recording.path),
        "imu_input": args.imu_input,
        "label": recording.label,
        "mode": args.mode,
        "ldm_ckpt": osp.abspath(args.ldm_ckpt),
        "vae_ckpt": osp.abspath(args.vae_ckpt),
        "ldm_stats": ldm_stats["path"],
        "n_samples": int(n_samples),
        "n_windows": n_samples // int(args.window_sec * args.sample_rate),
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
        "strength": args.strength,
        "scale_factor": float(ldm.scale_factor),
        "seed": args.seed,
    }
    if recording.sim_path:
        result["sim_input"] = osp.abspath(recording.sim_path)
    if cond_path and osp.abspath(cond_path) != osp.abspath(recording.path):
        result["cond_input"] = osp.abspath(cond_path)
        result["trim_to_match"] = bool(args.trim_to_match)

    result.update(window_metrics_from_sequences(
        features_orig, features_gen, ldm_stats, args, outdir,
    ))
    plots.plot_full_sequence_windows(
        features_orig, features_gen, outdir, n_plot=args.n_plot,
        window=int(args.window_sec * args.sample_rate), sample_rate=args.sample_rate,
        label_a="original", label_b="LDM gen",
    )
    if ronin_net is not None:
        result.update(evaluate_trajectory(
            imu_ds, features_orig, features_gen, outdir, ronin_net, device, args,
        ))
    result.update(export_generated(
        recording, features_gen, ts_imu, outdir, args, cond_path=cond_path,
    ))

    report.write_json(result, outdir, "metrics.json", what="Metrics")
    return result


def evaluate_dataset_split(args, ldm, sampler, ldm_stats, device):
    """Diffusion loss and per-window DDIM generation over a preprocessed split."""
    dataset_cls = PairedIMUDataset if args.mode == "sim_cond" or args.strength < 1 else IMUDataset
    dataset = dataset_cls(args.dataset_dir, ldm_stats["path"], split=args.split)
    print(f"Evaluating split:{args.split} ({len(dataset)} windows)")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    result = {
        "dataset_dir": osp.abspath(args.dataset_dir),
        "split": args.split,
        "source": f"split:{args.split}",
        "mode": args.mode,
        "ldm_ckpt": osp.abspath(args.ldm_ckpt),
        "vae_ckpt": osp.abspath(args.vae_ckpt),
        "ldm_stats": ldm_stats["path"],
    }
    split_metrics, real, gen = evaluate_split_windows(ldm, sampler, loader, device, args)
    result.update(split_metrics)

    real_phys = windows.inverse_standardize(real, ldm_stats["imu_mean"], ldm_stats["imu_std"])
    gen_phys = windows.inverse_standardize(gen, ldm_stats["imu_mean"], ldm_stats["imu_std"])
    result.update(metrics.physical_metrics(real_phys, gen_phys))
    result["sample_rate"] = args.sample_rate
    report.save_signal_plot_data(args.outdir, real.numpy(), gen.numpy(),
                                 real_phys.numpy(), gen_phys.numpy(), args.n_plot)

    title = f"LDM generation ({args.mode}, split:{args.split})"
    plots.plot_signal_overlays(
        real.numpy(), gen.numpy(), args.outdir, args.n_plot,
        sample_rate=args.sample_rate, title_prefix=title,
        input_label="gt", output_label="generated",
    )
    plots.plot_signal_overlays(
        real_phys.numpy(), gen_phys.numpy(), osp.join(args.outdir, "physical"),
        args.n_plot, sample_rate=args.sample_rate,
        title_prefix=f"{title} [physical]", input_label="gt", output_label="generated",
    )
    report.write_json(result, args.outdir, "metrics.json", what="Metrics", echo=True)
    return result


def evaluate_directory(args, ldm, sampler, ldm_stats, ronin_net, device):
    """Run the single-recording eval over every item under ``--input_dir``."""
    layout, recordings = paths.resolve_input_dir(
        args.input_dir, pairs_only=args.mode == "sim_cond",
    )
    print(f"Found {len(recordings)} recordings under {args.input_dir} ({layout} layout)")

    results, failures = batch.run_batch(
        recordings, args.outdir,
        lambda item, item_outdir: evaluate_recording(
            item, item_outdir, ldm, sampler, ldm_stats, ronin_net, device, args,
        ),
        kind="recording", force=args.force, dry_run=args.dry_run,
        plots_only=args.plots_only, strict=args.strict,
    )
    if args.dry_run:
        return None

    rows = [result for _name, result in results]
    aggregate = {
        "gen_rmse_mean": float(np.mean([r["gen_rmse"] for r in rows if "gen_rmse" in r]))
        if rows else None,
    }
    if ronin_net is not None:
        for side in ("original", "ldm_gen"):
            aggregate[side] = {
                f"{key}_mean": metrics.mean_metric(rows, side, key)
                for key in ("ate", "rte", "rte_short")
            }
    return batch.summarize(
        results, failures, args.outdir,
        extra={"input_dir": osp.abspath(args.input_dir), "layout": layout,
               "mode": args.mode, "ldm_ckpt": osp.abspath(args.ldm_ckpt)},
        aggregate=aggregate,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    source = validate(parser, args)

    if batch.handle_offline(
        args, lambda: paths.resolve_input_dir(args.input_dir, pairs_only=args.mode == "sim_cond"),
        lambda _item, dest, result: report.replot_signals(dest, result, args.n_plot),
    ):
        return
    generate.seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = eval_args.resolve_device(args)
    print(f"Device: {device}")

    ldm, sampler, ldm_stats, ronin_net = load_models(args, device)

    logdir = resolve_training_logdir(args.logdir, args.ldm_ckpt)
    if logdir is not None:
        plots.plot_ldm_training_curves(logdir, args.outdir)

    if source == "dataset_dir":
        evaluate_dataset_split(args, ldm, sampler, ldm_stats, device)
    elif source == "input_dir":
        evaluate_directory(args, ldm, sampler, ldm_stats, ronin_net, device)
    else:
        recording = paths.resolve_input(args.input, sim_path=args.sim_input)
        evaluate_recording(
            recording, args.outdir, ldm, sampler, ldm_stats, ronin_net, device, args,
            cond_path=args.cond_input,
        )

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
