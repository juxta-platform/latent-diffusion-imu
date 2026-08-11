"""Evaluate a trained 1D IMU VAE: metrics + reconstruction overlays.

Modes:
  1) Preprocessed split:
       --data_dir data/dataset_processed --split val
  2) Single real HDF5 or synthetic parquet (windowed on the fly):
       --input data/dataset/john_chest_ios_corrected.hdf5 \\
       --stats data/dataset_processed/stats.pt
       --input data/smplx_data_gen_0000.parquet --stats ...
       
Example usage:
  1) Single file -> python scripts/eval_vae_1d.py --config configs/imu/vae_1d.yaml --ckpt logs/vae_1d/checkpoints/best-000.ckpt 
        --input data/dataset/john_chest_ios_corrected.hdf5 --stats data/dataset_processed_overlapped/stats.pt --outdir outputs/vae_eval --n_plot -1
        
  2) Train or val list -> python scripts/eval_vae_1d.py --config configs/imu/vae_1d.yaml --ckpt logs/vae_1d/checkpoints/best-000.ckpt --data_dir data/dataset_processed_overlapped 
        --outdir outputs/vae_eval --n_plot 8 --logdir logs/vae_1d/lightning_logs/version_0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ldm.data.imu_dataset import IMUDataset, SyntheticIMUDataset, resolve_stats
from ldm.util import instantiate_from_config

CHANNEL_NAMES = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


def _load_preprocess_process_file():
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "preprocess_imu.py")
    spec = importlib.util.spec_from_file_location("preprocess_imu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.process_file


def inverse_standardize_imu(dataset, imu):
    """Undo IMU standardization using dataset mean/std."""
    return imu * dataset.imu_std.to(imu.device) + dataset.imu_mean.to(imu.device)


class HDF5WindowDataset(Dataset):
    """Window a real HDF5 recording and standardize with training stats."""

    def __init__(self, hdf5_path, stats_path, window_sec=10, sample_rate=200,
                 latent_length=100, local_frame=False):
        process_file = _load_preprocess_process_file()
        self.stats = torch.load(stats_path, weights_only=True)
        self.imu_mean = self.stats["imu_mean"].view(6, 1)
        self.imu_std = self.stats["imu_std"].view(6, 1)
        window_samples = int(window_sec * sample_rate)
        self.windows = process_file(hdf5_path, window_samples, latent_length,
                                    local_frame=local_frame)
        if not self.windows:
            raise ValueError(f"No full windows found in {hdf5_path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data = self.windows[idx]
        imu = (data["imu"] - self.imu_mean) / self.imu_std
        return {
            "imu": imu,
            "imu_raw": data["imu"],
            "velocity": data["velocity"],
            "physical_time": data["physical_time"],
        }


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, config


def resolve_stats_path(stats, data_dir, model=None):
    """Resolve stats.pt path; falls back to model-embedded stats if available."""
    if stats is not None:
        return stats
    if data_dir is not None:
        path = os.path.join(data_dir, "stats.pt")
        if os.path.isfile(path):
            return path
    if model is not None and hasattr(model, 'stats_set') and bool(model.stats_set):
        return None  # signal caller to use model-embedded stats
    raise ValueError("Provide --stats or --data_dir containing stats.pt (or use a ckpt with embedded stats)")


def _stats_path_or_model(args, model=None):
    """Get stats_path; returns None when model-embedded stats should be used."""
    return resolve_stats_path(args.stats, args.data_dir, model=model)


def build_dataset(args, model=None):
    """Build dataset from --input (hdf5/parquet) or preprocessed --data_dir/--split."""
    stats_path = _stats_path_or_model(args, model=model)
    local_frame = getattr(args, 'local_frame', False)

    if stats_path is None:
        imu_mean, imu_std = resolve_stats(model=model)
        tmp_stats = {'imu_mean': imu_mean.view(-1), 'imu_std': imu_std.view(-1),
                     'vel_mean': torch.zeros(2), 'vel_std': torch.ones(2)}
        tmp_path = os.path.join(args.outdir, '_tmp_stats.pt')
        torch.save(tmp_stats, tmp_path)
        stats_path = tmp_path

    if args.input is not None:
        ext = os.path.splitext(args.input)[1].lower()
        if ext in (".hdf5", ".h5"):
            dataset = HDF5WindowDataset(
                args.input,
                stats_path,
                window_sec=args.window_sec,
                sample_rate=args.sample_rate,
                latent_length=args.latent_length,
                local_frame=local_frame,
            )
            source = f"hdf5:{os.path.basename(args.input)}"
        elif ext == ".parquet":
            dataset = SyntheticIMUDataset(
                args.input,
                stats_path,
                window_sec=args.window_sec,
                sample_rate=args.sample_rate,
                latent_length=args.latent_length,
            )
            source = f"parquet:{os.path.basename(args.input)}"
        else:
            raise ValueError(f"--input must be .hdf5/.h5/.parquet, got {ext}")
        return dataset, source

    if args.data_dir is None:
        raise ValueError("Provide --data_dir (preprocessed) or --input (hdf5/parquet)")

    dataset = IMUDataset(args.data_dir, stats_path, split=args.split)
    source = f"split:{args.split}"
    return dataset, source


def plot_training_curves(logdir, outdir):
    """Plot loss curves from a Lightning TensorBoard event file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed; skipping training-curve plots")
        return

    ea = EventAccumulator(logdir)
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    if not scalars:
        print(f"No scalar tags found in {logdir}")
        return

    def series(tag):
        if tag not in scalars:
            return None
        events = ea.Scalars(tag)
        return np.array([e.step for e in events]), np.array([e.value for e in events])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, name in zip(axes, ["total_loss", "rec_loss", "kl_loss"]):
        train_ep = series(f"train/{name}_epoch")
        val = series(f"val/{name}")
        if train_ep is not None:
            ax.plot(train_ep[0], train_ep[1], label="train (epoch)", alpha=0.85)
        if val is not None:
            ax.plot(val[0], val[1], label="val", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if name != "kl_loss":
            ax.set_yscale("log")

    path = os.path.join(outdir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def plot_overlays(inputs, recons, outdir, n_plot, sample_rate=200, title_prefix="VAE reconstruction"):
    """Overlay input vs reconstruction for a few windows."""
    os.makedirs(os.path.join(outdir, "overlays"), exist_ok=True)
    t = np.arange(inputs.shape[-1]) / sample_rate
    n = inputs.shape[0] if n_plot < 0 else min(n_plot, inputs.shape[0])

    for i in range(n):
        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        for c, ax in enumerate(axes):
            ax.plot(t, inputs[i, c], label="input", linewidth=0.8, alpha=0.9)
            ax.plot(t, recons[i, c], label="recon", linewidth=0.8, alpha=0.9)
            rmse = np.sqrt(np.mean((inputs[i, c] - recons[i, c]) ** 2))
            ax.set_ylabel(CHANNEL_NAMES[c], fontsize=9)
            ax.set_title(f"{CHANNEL_NAMES[c]}  (RMSE={rmse:.4e})", fontsize=9, loc="left")
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"{title_prefix} — sample {i}", fontsize=12)
        path = os.path.join(outdir, "overlays", f"sample_{i:03d}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)

    rmse = np.sqrt(((inputs[:n] - recons[:n]) ** 2).mean(axis=(0, 2)))
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.bar(CHANNEL_NAMES, rmse)
    ax.set_ylabel("RMSE")
    ax.set_title(f"Per-channel RMSE over {n} plotted samples")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(outdir, "per_channel_rmse.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {n} overlay plots under {os.path.join(outdir, 'overlays')}")


@torch.no_grad()
def evaluate(model, loader, device, kl_weight, use_posterior_mean, max_batches=None):
    totals = {
        "total_loss": 0.0,
        "rec_loss": 0.0,
        "kl_loss": 0.0,
        "mae": 0.0,
        "n": 0,
    }
    per_ch_mse = torch.zeros(6, device=device)
    per_ch_mae = torch.zeros(6, device=device)
    all_inputs = []
    all_recons = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch["imu"].float().to(device)
        recon, posterior = model(x, sample_posterior=not use_posterior_mean)

        rec = torch.nn.functional.mse_loss(recon, x)
        kl = model._kl_loss(posterior)
        total = rec + kl_weight * kl
        mae = torch.nn.functional.l1_loss(recon, x)

        bs = x.shape[0]
        totals["total_loss"] += total.item() * bs
        totals["rec_loss"] += rec.item() * bs
        totals["kl_loss"] += kl.item() * bs
        totals["mae"] += mae.item() * bs
        totals["n"] += bs

        diff = recon - x
        per_ch_mse += diff.pow(2).mean(dim=(0, 2)) * bs
        per_ch_mae += diff.abs().mean(dim=(0, 2)) * bs

        all_inputs.append(x.cpu())
        all_recons.append(recon.cpu())

    n = max(totals["n"], 1)
    metrics = {
        "n_samples": totals["n"],
        "total_loss": totals["total_loss"] / n,
        "rec_loss": totals["rec_loss"] / n,
        "rmse": float(np.sqrt(totals["rec_loss"] / n)),
        "kl_loss": totals["kl_loss"] / n,
        "mae": totals["mae"] / n,
        "per_channel_mse": {
            name: (per_ch_mse[i] / n).item() for i, name in enumerate(CHANNEL_NAMES)
        },
        "per_channel_rmse": {
            name: float(torch.sqrt(per_ch_mse[i] / n).item()) for i, name in enumerate(CHANNEL_NAMES)
        },
        "per_channel_mae": {
            name: (per_ch_mae[i] / n).item() for i, name in enumerate(CHANNEL_NAMES)
        },
        "use_posterior_mean": use_posterior_mean,
        "kl_weight": kl_weight,
    }
    inputs = torch.cat(all_inputs, dim=0)
    recons = torch.cat(all_recons, dim=0)
    return metrics, inputs, recons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/imu/vae_1d.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Real .hdf5/.h5 or synthetic .parquet file to window and reconstruct",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Preprocessed dataset dir (train/val + stats.pt). Also used to find stats with --input",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Path to stats.pt (defaults to data_dir/stats.pt)",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--outdir", type=str, default="outputs/vae_eval")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--n_plot",
        type=int,
        default=8,
        help="Number of overlay plots (-1 = all windows)",
    )
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--sample_rate", type=int, default=200)
    parser.add_argument("--latent_length", type=int, default=100)
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="Use stochastic posterior sample instead of mean (default: mean)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Lightning log version dir (with events.out.tfevents.*) for training curves",
    )
    parser.add_argument("--local_frame", action="store_true",
                        help="Use local device-frame IMU (no rotation) for HDF5 windowing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model(args.config, args.ckpt, device)
    kl_weight = float(getattr(model, "kl_weight", config.model.params.get("kl_weight", 1e-6)))
    use_posterior_mean = not args.sample_posterior

    dataset, source = build_dataset(args, model=model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Evaluating {source} ({len(dataset)} windows) on {device} ...")
    metrics, inputs, recons = evaluate(
        model, loader, device, kl_weight, use_posterior_mean, args.max_batches
    )

    inputs_phys = inverse_standardize_imu(dataset, inputs)
    recons_phys = inverse_standardize_imu(dataset, recons)
    phys_mse = (inputs_phys - recons_phys).pow(2).mean().item()
    phys_mae = (inputs_phys - recons_phys).abs().mean().item()
    metrics["physical_mse"] = phys_mse
    metrics["physical_rmse"] = float(np.sqrt(phys_mse))
    metrics["physical_mae"] = phys_mae
    metrics["ckpt"] = os.path.abspath(args.ckpt)
    metrics["source"] = source
    if args.input is not None:
        metrics["input"] = os.path.abspath(args.input)
    else:
        metrics["split"] = args.split

    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    n_save = inputs.shape[0] if args.n_plot < 0 else min(args.n_plot, inputs.shape[0])
    torch.save(
        {
            "imu": inputs[:n_save],
            "recon": recons[:n_save],
            "imu_physical": inputs_phys[:n_save],
            "recon_physical": recons_phys[:n_save],
            "source": source,
        },
        os.path.join(args.outdir, "recon_samples.pt"),
    )

    title = f"VAE recon ({source})"
    plot_overlays(
        inputs.numpy(),
        recons.numpy(),
        args.outdir,
        args.n_plot,
        sample_rate=args.sample_rate,
        title_prefix=title,
    )
    plot_overlays(
        inputs_phys.numpy(),
        recons_phys.numpy(),
        os.path.join(args.outdir, "physical"),
        args.n_plot,
        sample_rate=args.sample_rate,
        title_prefix=f"{title} [physical]",
    )

    if args.logdir is not None:
        plot_training_curves(args.logdir, args.outdir)
    else:
        guess = os.path.abspath(os.path.join(os.path.dirname(args.ckpt), ".."))
        events = [f for f in os.listdir(guess) if "tfevents" in f] if os.path.isdir(guess) else []
        if events:
            print(f"Found event files in {guess}; plotting training curves")
            plot_training_curves(guess, args.outdir)

    print(f"Done. Results in {args.outdir}")


if __name__ == "__main__":
    main()
