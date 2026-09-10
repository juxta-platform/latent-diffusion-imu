"""Resolution of the ``stats.pt`` files behind ``--vae_stats`` / ``--ldm_stats``.

A stats file holds ``imu_mean``/``imu_std`` (6 channels, ``[accel, gyro]``
order) and, for datasets built for the LDM, ``vel_mean``/``vel_std``. VAE
checkpoints may instead carry the IMU statistics as buffers, in which case a
temporary file is materialized so consumers that need a path still work.

Resolution order for a given flag: explicit path > ``<dataset_dir>/stats.pt`` >
statistics embedded in the checkpoint.
"""

import os

import torch

from ldm.evaluation.constants import STATS_NAME

IMU_KEYS = ("imu_mean", "imu_std")
VELOCITY_KEYS = ("vel_mean", "vel_std")


def resolve_stats_path(stats_path=None, dataset_dir=None, required=True, what="stats"):
    """First existing stats path among the explicit flag and the dataset dir."""
    if stats_path is not None:
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"{what} not found: {stats_path}")
        return stats_path
    if dataset_dir is not None:
        candidate = os.path.join(dataset_dir, STATS_NAME)
        if os.path.isfile(candidate):
            return candidate
    if required:
        raise ValueError(
            f"Provide --{what} or a --dataset_dir containing {STATS_NAME}"
        )
    return None


def load_stats(path):
    """Load a stats file into ``[C, 1]`` tensors plus the originating path.

    Velocity statistics are optional; VAE-only datasets omit them.
    """
    raw = torch.load(path, weights_only=True)
    missing = [key for key in IMU_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    stats = {
        "path": os.path.abspath(path),
        "imu_mean": raw["imu_mean"].float().view(-1, 1),
        "imu_std": raw["imu_std"].float().view(-1, 1),
    }
    if all(key in raw for key in VELOCITY_KEYS):
        stats["vel_mean"] = raw["vel_mean"].float().view(-1, 1)
        stats["vel_std"] = raw["vel_std"].float().view(-1, 1)
    return stats


def stats_from_model(model, outdir):
    """Materialize checkpoint-embedded IMU statistics as a stats dict.

    Writes ``<outdir>/_embedded_stats.pt`` so downstream consumers that take a
    path (the synthetic parquet dataset, for instance) keep working. Velocity
    statistics are neutral because a VAE never conditions on velocity.
    """
    if not (hasattr(model, "stats_set") and bool(model.stats_set)):
        return None
    imu_mean = model.imu_mean.detach().cpu().view(-1)
    imu_std = model.imu_std.detach().cpu().view(-1)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "_embedded_stats.pt")
    torch.save(
        {
            "imu_mean": imu_mean,
            "imu_std": imu_std,
            "vel_mean": torch.zeros(2),
            "vel_std": torch.ones(2),
        },
        path,
    )
    return load_stats(path)


def resolve_stats(stats_path=None, dataset_dir=None, model=None, outdir=None,
                  what="vae_stats"):
    """Load statistics from a flag, a dataset dir, or the checkpoint itself."""
    path = resolve_stats_path(stats_path, dataset_dir, required=False, what=what)
    if path is not None:
        return load_stats(path)
    if model is not None and outdir is not None:
        embedded = stats_from_model(model, outdir)
        if embedded is not None:
            print(f"  {what}: using statistics embedded in the checkpoint")
            return embedded
    raise ValueError(
        f"Provide --{what}, or a --dataset_dir containing {STATS_NAME}, "
        f"or use a checkpoint with embedded statistics"
    )


def require_velocity(stats, what="ldm_stats"):
    """Raise unless the stats carry the velocity statistics the LDM needs."""
    if all(key in stats for key in VELOCITY_KEYS):
        return stats
    raise ValueError(
        f"{stats['path']} has no {list(VELOCITY_KEYS)}. LDM conditioning needs the "
        f"stats.pt the LDM was trained with; pass it via --{what}."
    )
