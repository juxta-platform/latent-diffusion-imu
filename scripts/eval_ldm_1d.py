"""Evaluate a trained 1D IMU LDM: diffusion loss + trajectory-conditioned generation.

Modes:
  1) Preprocessed split:
       --data_dir data/dataset_processed --split val
  2) Single real HDF5 or synthetic parquet (windowed on the fly):
       --input data/dataset/john_chest_ios_corrected.hdf5 \\
       --stats data/dataset_processed/stats.pt

Example usage:
  1) Single file -> python scripts/eval_ldm_1d.py --config configs/imu/ldm_1d.yaml \\
        --ckpt logs/ldm_1d/.../best-000.ckpt \\
        --first_stage_ckpt logs/vae_1d/.../best-000.ckpt \\
        --input data/dataset/john_chest_ios_corrected.hdf5 \\
        --stats data/dataset_processed/stats.pt --outdir outputs/ldm_eval --n_plot -1

  2) Val split -> python scripts/eval_ldm_1d.py --config configs/imu/ldm_1d.yaml \\
        --ckpt logs/ldm_1d/.../best-000.ckpt \\
        --first_stage_ckpt logs/vae_1d/.../best-000.ckpt \\
        --data_dir data/dataset_processed --outdir outputs/ldm_eval --n_plot 8 \\
        --logdir logs/ldm_1d/.../lightning_logs/version_0
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ldm.data.imu_dataset import IMUDataset, SyntheticIMUDataset
from ldm.models.diffusion.ddim_1d import DDIMSampler1D
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
        self.vel_mean = self.stats["vel_mean"].view(2, 1)
        self.vel_std = self.stats["vel_std"].view(2, 1)
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
        velocity = (data["velocity"] - self.vel_mean) / self.vel_std
        return {
            "imu": imu,
            "imu_raw": data["imu"],
            "velocity": velocity,
            "physical_time": data["physical_time"],
        }


def load_model(config_path, ckpt_path, device, first_stage_ckpt=None, scale_factor=None):
    config = OmegaConf.load(config_path)
    if first_stage_ckpt is not None:
        config.model.params.first_stage_ckpt = first_stage_ckpt
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    if scale_factor is not None:
        model.register_buffer("scale_factor", torch.tensor(float(scale_factor)))
    model = model.to(device).eval()
    return model, config


def resolve_stats_path(stats, data_dir):
    if stats is not None:
        return stats
    if data_dir is not None:
        path = os.path.join(data_dir, "stats.pt")
        if os.path.isfile(path):
            return path
    raise ValueError("Provide --stats or --data_dir containing stats.pt")


def build_dataset(args):
    """Build dataset from --input (hdf5/parquet) or preprocessed --data_dir/--split."""
    if args.input is not None:
        stats_path = resolve_stats_path(args.stats, args.data_dir)
        ext = os.path.splitext(args.input)[1].lower()
        if ext in (".hdf5", ".h5"):
            dataset = HDF5WindowDataset(
                args.input,
                stats_path,
                window_sec=args.window_sec,
                sample_rate=args.sample_rate,
                latent_length=args.latent_length,
                local_frame=getattr(args, 'local_frame', False),
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

    stats_path = resolve_stats_path(args.stats, args.data_dir)
    dataset = IMUDataset(args.data_dir, stats_path, split=args.split)
    source = f"split:{args.split}"
    return dataset, source


def latent_shape(model):
    """Infer (z_channels, latent_length) from first-stage config defaults."""
    z_ch = getattr(model.first_stage_model, "embed_dim", 8)
    return (int(z_ch), 100)


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

    def steps_to_epochs(steps, epoch_map):
        """Map TensorBoard global steps to epoch indices via the logged 'epoch' scalar."""
        if epoch_map is None or len(steps) == 0:
            return steps
        ep_steps, ep_vals = epoch_map
        return np.interp(steps.astype(float), ep_steps.astype(float), ep_vals.astype(float))

    train_ep = series("train/loss_epoch")
    train_step = series("train/loss_step")
    val = series("val/loss")
    val_ema = series("val/loss_ema")
    epoch_map = series("epoch")
    if train_ep is None and train_step is None and val is None:
        print(f"No loss tags found in {logdir}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    if train_ep is not None:
        ax.plot(steps_to_epochs(train_ep[0], epoch_map), train_ep[1],
                label="train", color="C0", alpha=0.85, linewidth=1.2)
    if val is not None:
        ax.plot(steps_to_epochs(val[0], epoch_map), val[1],
                label="val", color="C1", alpha=0.9, linewidth=1.4)
    if val_ema is not None:
        ax.plot(steps_to_epochs(val_ema[0], epoch_map), val_ema[1],
                label="val (EMA)", color="C2", alpha=0.9, linewidth=1.4)
    ax.set_title("Epoch loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if train_step is not None:
        ax.plot(train_step[0], train_step[1], label="train (step)", color="C0", alpha=0.55, linewidth=0.9)
    if val is not None:
        ax.plot(val[0], val[1], label="val", color="C1", alpha=0.9, linewidth=1.4)
    if val_ema is not None:
        ax.plot(val_ema[0], val_ema[1], label="val (EMA)", color="C2", alpha=0.9, linewidth=1.4)
    ax.set_title("Step loss")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def plot_overlays(inputs, gens, outdir, n_plot, sample_rate=200, title_prefix="LDM generation"):
    """Overlay GT vs generated IMU for a few windows."""
    os.makedirs(os.path.join(outdir, "overlays"), exist_ok=True)
    t = np.arange(inputs.shape[-1]) / sample_rate
    n = inputs.shape[0] if n_plot < 0 else min(n_plot, inputs.shape[0])

    for i in range(n):
        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        for c, ax in enumerate(axes):
            ax.plot(t, inputs[i, c], label="gt", linewidth=0.8, alpha=0.9)
            ax.plot(t, gens[i, c], label="generated", linewidth=0.8, alpha=0.9)
            rmse = np.sqrt(np.mean((inputs[i, c] - gens[i, c]) ** 2))
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

    rmse = np.sqrt(((inputs[:n] - gens[:n]) ** 2).mean(axis=(0, 2)))
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
def evaluate(model, sampler, loader, device, ddim_steps, ddim_eta, max_batches=None, use_ema=True):
    totals = {
        "diff_loss": 0.0,
        "diff_loss_ema": 0.0,
        "gen_mse": 0.0,
        "gen_mae": 0.0,
        "vae_mse": 0.0,
        "n": 0,
    }
    per_ch_mse = torch.zeros(6, device=device)
    per_ch_mae = torch.zeros(6, device=device)
    all_inputs = []
    all_gens = []
    all_vae_recons = []
    shape = latent_shape(model)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch["imu"].float().to(device)
        velocity = batch["velocity"].float().to(device)
        physical_time = batch["physical_time"].float().to(device)
        cond = {"velocity": velocity, "physical_time": physical_time}
        bs = x.shape[0]

        # Diffusion noise-prediction loss (same as training val)
        posterior = model.encode_first_stage(x)
        z = model.scale_factor * posterior.sample()
        t = torch.randint(0, model.num_timesteps, (bs,), device=device).long()
        loss = model.p_losses(z, t, cond)
        totals["diff_loss"] += loss.item() * bs

        if model.use_ema:
            with model.ema_scope():
                loss_ema = model.p_losses(z, t, cond)
            totals["diff_loss_ema"] += loss_ema.item() * bs
        else:
            totals["diff_loss_ema"] += loss.item() * bs

        # VAE reconstruction baseline
        z_mode = posterior.mode()
        vae_recon = model.decode_first_stage(z_mode)
        vae_mse = torch.nn.functional.mse_loss(vae_recon, x)
        totals["vae_mse"] += vae_mse.item() * bs

        # Trajectory-conditioned DDIM generation
        ctx = model.ema_scope("eval") if (use_ema and model.use_ema) else nullcontext()
        with ctx:
            samples, _ = sampler.sample(
                S=ddim_steps,
                batch_size=bs,
                shape=shape,
                conditioning=cond,
                eta=ddim_eta,
                verbose=False,
            )
            gen = model.decode_first_stage(samples / model.scale_factor)

        gen_mse = torch.nn.functional.mse_loss(gen, x)
        gen_mae = torch.nn.functional.l1_loss(gen, x)
        totals["gen_mse"] += gen_mse.item() * bs
        totals["gen_mae"] += gen_mae.item() * bs
        totals["n"] += bs

        diff = gen - x
        per_ch_mse += diff.pow(2).mean(dim=(0, 2)) * bs
        per_ch_mae += diff.abs().mean(dim=(0, 2)) * bs

        all_inputs.append(x.cpu())
        all_gens.append(gen.cpu())
        all_vae_recons.append(vae_recon.cpu())

    n = max(totals["n"], 1)
    metrics = {
        "n_samples": totals["n"],
        "diff_loss": totals["diff_loss"] / n,
        "diff_loss_ema": totals["diff_loss_ema"] / n,
        "gen_mse": totals["gen_mse"] / n,
        "gen_rmse": float(np.sqrt(totals["gen_mse"] / n)),
        "gen_mae": totals["gen_mae"] / n,
        "vae_mse": totals["vae_mse"] / n,
        "vae_rmse": float(np.sqrt(totals["vae_mse"] / n)),
        "per_channel_mse": {
            name: (per_ch_mse[i] / n).item() for i, name in enumerate(CHANNEL_NAMES)
        },
        "per_channel_rmse": {
            name: float(torch.sqrt(per_ch_mse[i] / n).item()) for i, name in enumerate(CHANNEL_NAMES)
        },
        "per_channel_mae": {
            name: (per_ch_mae[i] / n).item() for i, name in enumerate(CHANNEL_NAMES)
        },
        "ddim_steps": ddim_steps,
        "ddim_eta": ddim_eta,
        "use_ema": use_ema and bool(model.use_ema),
        "scale_factor": float(model.scale_factor),
    }
    inputs = torch.cat(all_inputs, dim=0)
    gens = torch.cat(all_gens, dim=0)
    vae_recons = torch.cat(all_vae_recons, dim=0)
    return metrics, inputs, gens, vae_recons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/imu/ldm_1d.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="LDM checkpoint path")
    parser.add_argument(
        "--first_stage_ckpt",
        type=str,
        default=None,
        help="VAE checkpoint (overrides config first_stage_ckpt)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Real .hdf5/.h5 or synthetic .parquet file to window and evaluate",
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
    parser.add_argument("--outdir", type=str, default="outputs/ldm_eval")
    parser.add_argument("--batch_size", type=int, default=8)
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
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help="Override model.scale_factor (loaded from ckpt buffer when present)",
    )
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA weights for sampling")
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

    model, config = load_model(
        args.config,
        args.ckpt,
        device,
        first_stage_ckpt=args.first_stage_ckpt,
        scale_factor=args.scale_factor,
    )
    sampler = DDIMSampler1D(model)
    print(f"scale_factor={model.scale_factor}, use_ema={model.use_ema and not args.no_ema}")

    dataset, source = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Evaluating {source} ({len(dataset)} windows) on {device} ...")
    metrics, inputs, gens, vae_recons = evaluate(
        model,
        sampler,
        loader,
        device,
        args.ddim_steps,
        args.ddim_eta,
        args.max_batches,
        use_ema=not args.no_ema,
    )

    inputs_phys = inverse_standardize_imu(dataset, inputs)
    gens_phys = inverse_standardize_imu(dataset, gens)
    phys_mse = (inputs_phys - gens_phys).pow(2).mean().item()
    phys_mae = (inputs_phys - gens_phys).abs().mean().item()
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
            "generated": gens[:n_save],
            "vae_recon": vae_recons[:n_save],
            "imu_physical": inputs_phys[:n_save],
            "generated_physical": gens_phys[:n_save],
            "source": source,
        },
        os.path.join(args.outdir, "gen_samples.pt"),
    )

    title = f"LDM gen ({source})"
    plot_overlays(
        inputs.numpy(),
        gens.numpy(),
        args.outdir,
        args.n_plot,
        sample_rate=args.sample_rate,
        title_prefix=title,
    )
    plot_overlays(
        inputs_phys.numpy(),
        gens_phys.numpy(),
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
