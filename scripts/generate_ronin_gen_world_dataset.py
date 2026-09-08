#!/usr/bin/env python3
"""Generate a RoNIN ``gen_world`` HDF5 dataset from real/sim pair folders.

Modes:
  sim_cond  — sim-conditioned LDM (eval_ldm_1d_sim_cond). Conditions on
              trajectory velocity plus synthetic parquet IMU latent.
              Generates IMU from noise; real.hdf5 is optional.
  traj      — trajectory-conditioned LDM (eval_ldm_1d / eval_ronin_ldm).
              Conditions only on velocities from real.hdf5 tango_pos;
              skips synthetic parquets.

Example (sim-cond, existing behavior):
  python scripts/generate_ronin_gen_world_dataset.py \\
      --mode sim_cond \\
      --pairs_root data/real_sim_imu_pairs_ronin_dataset \\
      --ckpt logs/ldm_1d_sim_cond/ldm_world_sim_cond/checkpoints/last.ckpt \\
      --first_stage_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
      --stats data/real_sim_pairs_processed/stats.pt \\
      --outdir data/ronin_ldm_gen_world

Example (traj-only LDM, no parquet):
  python scripts/generate_ronin_gen_world_dataset.py \\
      --mode traj \\
      --pairs_root data/real_sim_imu_pairs_ronin_dataset \\
      --ckpt logs/ldm_1d/ldm_world_full_dataset_velbugfix/checkpoints/last.ckpt \\
      --first_stage_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
      --stats data/dataset_full_processed_velbugfix/stats.pt \\
      --outdir data/ronin_ldm_gen_world_traj
"""

import argparse
import csv
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from ldm.models.diffusion.ddim_1d import DDIMSampler1D
from ldm.util import instantiate_from_config


def load_model(config_path, ckpt_path, first_stage_ckpt, device, scale_factor=None):
    config = OmegaConf.load(config_path)
    config.model.params.first_stage_ckpt = first_stage_ckpt
    model = instantiate_from_config(config.model)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    if scale_factor is not None:
        model.register_buffer("scale_factor", torch.tensor(float(scale_factor)))
    return model.to(device).eval()


def load_stats(path):
    stats = torch.load(path, weights_only=True)
    return {
        name: stats[name].float().view(1, -1, 1)
        for name in ("imu_mean", "imu_std", "vel_mean", "vel_std")
    }


def resolve_manifest_path(value, repo_root, pair_dir):
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [repo_root / path, pair_dir / path.name]
    return next((p for p in candidates if p.is_file()), candidates[0])


def path_is_file(path):
    return path is not None and path.is_file()


def resolve_pair_files(row, pairs_root, repo_root, sim_root=None):
    pair_id = row["pair_id"].strip()
    pair_dir = pairs_root / pair_id
    trajectory = resolve_manifest_path(row.get("trajectory_txt"), repo_root, pair_dir)
    real = resolve_manifest_path(row.get("real_hdf5"), repo_root, pair_dir)
    if real is None:
        fallback_real = pair_dir / "real.hdf5"
        real = fallback_real if fallback_real.is_file() else None

    sim_candidates = []
    if sim_root is not None:
        sim_candidates.extend([
            sim_root / pair_id / "synthetic.parquet",
            sim_root / pair_id / ".run" / "synthetic_0000.parquet",
        ])
    manifest_sim = resolve_manifest_path(row.get("synthetic_parquet"), repo_root, pair_dir)
    if manifest_sim is not None:
        sim_candidates.append(manifest_sim)
    sim_candidates.extend([
        pair_dir / "synthetic.parquet",
        pair_dir / ".run" / "synthetic_0000.parquet",
    ])
    synthetic = next((p for p in sim_candidates if p.is_file()), sim_candidates[0])
    return pair_id, trajectory, synthetic, real


def read_manifest(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("status", "").strip() == "generated"]


def flat_stem(pair_id):
    """Map carry/seq pair ids to a flat basename for gen_world lists."""
    return pair_id.strip().replace("/", "_").replace("\\", "_")


def load_real_hdf5(real_path):
    """Load time / tango_pos / game_rv and world-frame IMU from real.hdf5.

    real.hdf5 stores device-frame IMU; rotate via game_rv so an unchanged
    tail matches LDM world-frame output (acce, gyro).
    """
    with h5py.File(real_path, "r") as f:
        time = np.copy(f["synced/time"]).astype(np.float64)
        position = np.copy(f["synced/tango_pos"]).astype(np.float64)
        orientation = np.copy(f["synced/game_rv"]).astype(np.float64)
        real_acce = np.copy(f["synced/acce"]).astype(np.float64)
        real_gyro = np.copy(f["synced/gyro"]).astype(np.float64)

    rotation = Rotation.from_quat(orientation[:, [1, 2, 3, 0]])
    real_acce = rotation.apply(real_acce)
    real_gyro = rotation.apply(real_gyro)
    real_imu = np.concatenate([real_acce, real_gyro], axis=1).astype(np.float32)

    n = min(len(time), len(position), len(orientation), len(real_imu))
    time, position, orientation, real_imu = (
        time[:n], position[:n], orientation[:n], real_imu[:n]
    )
    if n < 2 or np.any(np.diff(time) <= 0):
        raise ValueError(f"{real_path} has non-increasing or insufficient timestamps")
    return time, position, orientation, real_imu


def load_pair(trajectory_path, synthetic_path, real_path=None):
    """Load trajectory + synthetic IMU; real.hdf5 is optional for LDM generation.

    Without real.hdf5, sequence length comes from trajectory/sim only and the
    generated remainder shorter than one window is trimmed (no real IMU tail).
    """
    trajectory = np.loadtxt(trajectory_path, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] < 8:
        raise ValueError(
            f"{trajectory_path} must have columns: time, xyz, quaternion(wxyz)"
        )

    sim = pd.read_parquet(synthetic_path)
    columns = [
        "accel_world_x", "accel_world_y", "accel_world_z",
        "gyro_world_x", "gyro_world_y", "gyro_world_z",
    ]
    missing = [name for name in columns if name not in sim.columns]
    if missing:
        raise ValueError(f"{synthetic_path} missing columns: {missing}")

    lengths = [len(trajectory), len(sim)]
    real_imu = None
    if real_path is not None:
        _, _, _, real_imu = load_real_hdf5(real_path)
        lengths.append(len(real_imu))

    n = min(lengths)
    time = trajectory[:n, 0]
    position = trajectory[:n, 1:4]
    orientation = trajectory[:n, 4:8]
    sim_imu = sim.loc[:, columns].iloc[:n].to_numpy(dtype=np.float32)
    if real_imu is not None:
        real_imu = real_imu[:n]

    if n < 2 or np.any(np.diff(time) <= 0):
        raise ValueError(f"{trajectory_path} has non-increasing or insufficient timestamps")
    return time, position, orientation, sim_imu, real_imu


def window_starts(n_samples, window, stride):
    if n_samples < window:
        return []
    return list(range(0, n_samples - window + 1, stride))


def build_conditioning(time, position, starts, window, latent_length, stats, device):
    velocities = []
    physical_times = []
    for start in starts:
        end = start + window
        t = time[start:end]
        pos = position[start:end, :2]
        velocity = np.diff(pos, axis=0) / np.maximum(np.diff(t)[:, None], 1e-8)
        velocity = np.concatenate([velocity[:1], velocity], axis=0)
        velocities.append(torch.from_numpy(velocity.T.copy()).float())
        t_rel = (t - t[0]) / max(t[-1] - t[0], 1e-8)
        physical_times.append(torch.from_numpy(t_rel.copy()).float().unsqueeze(0))

    velocity = F.interpolate(
        torch.stack(velocities), size=latent_length, mode="linear", align_corners=True
    )
    physical_time = F.interpolate(
        torch.stack(physical_times), size=latent_length, mode="linear", align_corners=True
    )
    velocity = (
        velocity - stats["vel_mean"]
    ) / stats["vel_std"]
    return {
        "velocity": velocity.to(device),
        "physical_time": physical_time.to(device),
    }


@torch.no_grad()
def generate_sequence(
    model,
    sampler,
    stats,
    time,
    position,
    sim_imu,
    real_imu,
    device,
    window,
    stride,
    latent_length,
    batch_size,
    ddim_steps,
    ddim_eta,
    use_ema,
    trim_tail,
):
    n_samples = len(time)
    starts = window_starts(n_samples, window, stride)
    if not starts:
        raise ValueError(f"sequence has {n_samples} samples, fewer than window {window}")

    # LDM fills complete windows from noise. If real IMU is available and
    # --trim_tail is off, keep world-frame real IMU for a short remainder;
    # otherwise initialize from zeros and trim the remainder.
    if real_imu is None:
        output = np.zeros((n_samples, 6), dtype=np.float32)
        retain_real_tail = False
    else:
        if len(real_imu) != n_samples:
            raise ValueError(
                f"real IMU length {len(real_imu)} != trajectory length {n_samples}"
            )
        output = real_imu.copy()
        retain_real_tail = not trim_tail
    imu_mean = stats["imu_mean"].to(device)
    imu_std = stats["imu_std"].to(device)
    z_channels = int(getattr(model.first_stage_model, "embed_dim", 8))
    shape = (z_channels, latent_length)
    use_sim = sim_imu is not None

    context = model.ema_scope("dataset generation") if use_ema and model.use_ema else nullcontext()
    with context:
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            conditioning = build_conditioning(
                time, position, batch_starts, window, latent_length, stats, device
            )
            if use_sim:
                sim_windows = np.stack([
                    sim_imu[start:start + window].T for start in batch_starts
                ])
                sim_windows = torch.from_numpy(sim_windows).float().to(device)
                sim_normalized = (sim_windows - imu_mean) / imu_std
                conditioning["sim_latent"] = (
                    model.scale_factor * model.encode_first_stage(sim_normalized).mode()
                )
            samples, _ = sampler.sample(
                S=ddim_steps,
                batch_size=len(batch_starts),
                shape=shape,
                conditioning=conditioning,
                eta=ddim_eta,
                verbose=False,
            )
            generated = model.decode_first_stage(samples / model.scale_factor)
            generated = (generated * imu_std + imu_mean).cpu().numpy()
            if generated.shape[-1] != window:
                raise ValueError(
                    f"model decoded {generated.shape[-1]} samples; expected {window}"
                )

            for item, start in enumerate(batch_starts):
                end = start + window
                output[start:end] = generated[item].T

            done = min(offset + len(batch_starts), len(starts))
            print(f"    generated {done}/{len(starts)} windows", end="\r", flush=True)
    print()

    tail = n_samples - (starts[-1] + window)
    if tail > 0:
        if retain_real_tail:
            print(f"    retained {tail} trailing real IMU samples")
        else:
            output = output[:-tail]
            print(f"    trimmed {tail} trailing samples")
    return output.astype(np.float32)


def write_hdf5(path, time, position, orientation, generated, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp_path, "w") as f:
        synced = f.create_group("synced")
        synced.create_dataset("time", data=time, compression="gzip")
        synced.create_dataset("tango_pos", data=position, compression="gzip")
        synced.create_dataset("game_rv", data=orientation, compression="gzip")
        synced.create_dataset("acce", data=generated[:, :3], compression="gzip")
        synced.create_dataset("gyro", data=generated[:, 3:], compression="gzip")
        f.attrs["imu_frame"] = "world"
        f.attrs["imu_channel_order"] = "acce_xyz,gyro_xyz"
        f.attrs["generator"] = metadata.get("generator", "sim-conditioned LDM")
        f.attrs["generation_metadata"] = json.dumps(metadata)
    os.replace(tmp_path, path)


def output_is_valid(path):
    try:
        with h5py.File(path, "r") as f:
            lengths = [
                len(f[name]) for name in (
                    "synced/time", "synced/tango_pos", "synced/game_rv",
                    "synced/acce", "synced/gyro",
                )
            ]
            return min(lengths) > 0 and len(set(lengths)) == 1
    except (OSError, KeyError):
        return False


def write_lists(outdir, pair_ids, val_fraction, seed):
    pair_ids = sorted(pair_ids)
    shuffled = pair_ids.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    if val_fraction > 0 and len(shuffled) > 1:
        n_val = min(max(n_val, 1), len(shuffled) - 1)
    val = set(shuffled[:n_val])
    train = [pair_id for pair_id in pair_ids if pair_id not in val]
    val = [pair_id for pair_id in pair_ids if pair_id in val]

    for name, values in (("all.txt", pair_ids), ("train.txt", train), ("val.txt", val)):
        with (outdir / name).open("w") as f:
            f.write("\n".join(values))
            if values:
                f.write("\n")
    return len(train), len(val)


def main():
    parser = argparse.ArgumentParser(
        description="Generate world-frame HDF5s for ronin_resnet.py --dataset gen_world"
    )
    parser.add_argument(
        "--mode",
        choices=["sim_cond", "traj"],
        default="sim_cond",
        help="sim_cond: trajectory + synthetic IMU latent (eval_ldm_1d_sim_cond). "
             "traj: trajectory velocities from real.hdf5 only (eval_ldm_1d).",
    )
    parser.add_argument("--pairs_root", type=Path, required=True)
    parser.add_argument("--sim_root", type=Path, default=None,
                        help="Optional mirrored root containing synthetic parquet files "
                             "(sim_cond mode only)")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config", default=None,
                        help="LDM config. Defaults to ldm_1d_sim_cond.yaml (sim_cond) "
                             "or ldm_1d.yaml (traj)")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--first_stage_ckpt", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--data_dir", default=None,
                        help="Used to resolve <data_dir>/stats.pt when --stats is omitted")
    parser.add_argument("--sample_rate", type=int, default=200)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--stride_sec", type=float, default=10.0)
    parser.add_argument("--latent_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--scale_factor", type=float, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--trim_tail", action="store_true",
                        help="Trim a remainder shorter than one LDM window instead "
                             "of retaining world-frame real IMU. Always applied when "
                             "real.hdf5 is absent.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Stop at the first missing or failed pair")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only discover and validate pair inputs")
    args = parser.parse_args()

    if args.config is None:
        args.config = (
            "configs/imu/ldm_1d.yaml" if args.mode == "traj"
            else "configs/imu/ldm_1d_sim_cond.yaml"
        )

    if not 0 <= args.val_fraction < 1:
        parser.error("--val_fraction must be in [0, 1)")
    window = int(round(args.window_sec * args.sample_rate))
    stride = int(round(args.stride_sec * args.sample_rate))
    if window <= 0 or stride <= 0 or stride > window:
        parser.error("require 0 < --stride_sec <= --window_sec")

    pairs_root = args.pairs_root.resolve()
    manifest_path = pairs_root / "manifest.csv"
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")
    repo_root = pairs_root.parent.parent
    rows = read_manifest(manifest_path)
    resolved = [
        (row, *resolve_pair_files(row, pairs_root, repo_root, args.sim_root))
        for row in rows
    ]

    def pair_ready(trajectory, synthetic, real):
        if args.mode == "traj":
            # Traj mode conditions on velocities from real.hdf5 tango_pos.
            return path_is_file(real)
        # sim_cond generates IMU from noise using trajectory + synthetic latent.
        # real.hdf5 is optional (only used to retain a short real IMU tail).
        return path_is_file(trajectory) and path_is_file(synthetic)

    runnable = [
        item for item in resolved
        if pair_ready(item[2], item[3], item[4])
    ]
    print(
        f"Mode={args.mode}. Found {len(rows)} generated manifest entries; "
        f"{len(runnable)} have required inputs"
    )

    missing = []
    for _, pair_id, trajectory, synthetic, real in resolved:
        if pair_ready(trajectory, synthetic, real):
            continue
        missing.append({
            "pair_id": pair_id,
            "trajectory": None if trajectory is None else str(trajectory),
            "trajectory_exists": path_is_file(trajectory),
            "synthetic": None if synthetic is None else str(synthetic),
            "synthetic_exists": path_is_file(synthetic),
            "real": None if real is None else str(real),
            "real_exists": path_is_file(real),
        })
    if missing:
        print(f"Missing inputs for {len(missing)} pairs")
        for item in missing[:10]:
            print(f"  {item['pair_id']}: trajectory={item['trajectory_exists']}, "
                  f"synthetic={item['synthetic_exists']}, real={item['real_exists']}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        if args.strict:
            raise FileNotFoundError("missing pair inputs (see report above)")
    if args.dry_run:
        return
    if not runnable:
        raise FileNotFoundError(
            "No runnable pairs. For sim_cond, each pair needs trajectory.txt and "
            "synthetic.parquet (real.hdf5 optional). For traj, each pair needs "
            "real.hdf5."
        )

    stats_path = args.stats or (
        os.path.join(args.data_dir, "stats.pt") if args.data_dir else None
    )
    if stats_path is None:
        parser.error("provide --stats or --data_dir containing stats.pt")

    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model(
        args.config, args.ckpt, args.first_stage_ckpt, device, args.scale_factor
    )
    sampler = DDIMSampler1D(model)
    stats = load_stats(stats_path)
    successful = []
    failures = []
    generator_name = (
        "trajectory-conditioned LDM" if args.mode == "traj"
        else "sim-conditioned LDM"
    )

    for index, (row, pair_id, trajectory_path, synthetic_path, real_path) in enumerate(runnable, 1):
        stem = flat_stem(pair_id)
        output_path = args.outdir / f"{stem}.hdf5"
        print(f"[{index}/{len(runnable)}] {pair_id} -> {stem}.hdf5")
        if output_path.is_file() and output_is_valid(output_path) and not args.overwrite:
            print("    already complete")
            successful.append(stem)
            continue
        try:
            if args.mode == "traj":
                time, position, orientation, real_imu = load_real_hdf5(real_path)
                sim_imu = None
            else:
                time, position, orientation, sim_imu, real_imu = load_pair(
                    trajectory_path, synthetic_path,
                    real_path if path_is_file(real_path) else None,
                )
            generated = generate_sequence(
                model, sampler, stats, time, position, sim_imu, real_imu, device,
                window, stride, args.latent_length, args.batch_size,
                args.ddim_steps, args.ddim_eta, not args.no_ema, args.trim_tail,
            )
            output_len = len(generated)
            has_real = path_is_file(real_path)
            metadata = {
                "pair_id": pair_id,
                "output_stem": stem,
                "mode": args.mode,
                "generator": generator_name,
                "synthetic_source": (
                    None if args.mode == "traj" else str(synthetic_path.resolve())
                ),
                "real_source": str(real_path.resolve()) if has_real else None,
                "trajectory_source": (
                    str(real_path.resolve()) if args.mode == "traj"
                    else str(trajectory_path.resolve())
                ),
                "ldm_config": str(Path(args.config).resolve()),
                "ldm_checkpoint": str(Path(args.ckpt).resolve()),
                "first_stage_checkpoint": str(Path(args.first_stage_ckpt).resolve()),
                "stats": str(Path(stats_path).resolve()),
                "seed": args.seed,
                "ddim_steps": args.ddim_steps,
                "ddim_eta": args.ddim_eta,
                "tail_mode": (
                    "real" if has_real and not args.trim_tail else "trim"
                ),
            }
            write_hdf5(
                output_path, time[:output_len], position[:output_len],
                orientation[:output_len], generated, metadata
            )
            successful.append(stem)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failures.append({"pair_id": pair_id, "error": repr(exc)})
            if args.strict:
                raise

    n_train, n_val = write_lists(
        args.outdir, successful, args.val_fraction, args.seed
    )
    with (args.outdir / "generation_summary.json").open("w") as f:
        json.dump({
            "mode": args.mode,
            "config": args.config,
            "ckpt": str(Path(args.ckpt).resolve()),
            "successful": successful,
            "missing": missing,
            "failures": failures,
            "train_sequences": n_train,
            "val_sequences": n_val,
        }, f, indent=2)
    print(f"Done: {len(successful)} HDF5s ({n_train} train, {n_val} val), "
          f"{len(failures)} failed")


if __name__ == "__main__":
    main()
