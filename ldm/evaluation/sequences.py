"""Recording loaders, IMU frame conversion and window construction.

Three on-disk recording formats feed the pipeline:

* Real HDF5 (``synced/{time,acce,gyro,linacce,tango_pos,game_rv}``) storing
  device-frame IMU that is rotated to world frame with ``game_rv``.
* Simulator parquet with ``accel_local_*``/``accel_world_*`` column families.
* Generated "gen_world" HDF5, whose ``synced/acce|gyro`` are already world
  frame and must not be rotated again (``already_world``).

Every loader takes ``imu_frame`` explicitly rather than defaulting, because
which frame a model wants is a property of the model, not of the file: the VAE
and LDM are trained on world frame, the carrying classifier on whichever frame
its checkpoint records.

This module owns everything from a path to a window: the raw loaders, the
window builders :func:`process_file` / :func:`process_pair`, and the
``Dataset`` wrappers around them. :mod:`ldm.evaluation.windows` holds the
tensor-level operations that apply to windows once built.
"""

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ldm.evaluation.constants import (
    ACCEL_LOCAL_COLS,
    ACCEL_WORLD_COLS,
    GYRO_LOCAL_COLS,
    GYRO_WORLD_COLS,
    HDF5_EXTS,
    LATENT_LENGTH,
    PHONE_ROT_COLS,
    SAMPLE_RATE,
    STATS_NAME,
)

IMU_FRAMES = ("local", "world")


def check_imu_frame(imu_frame):
    """Validate an ``imu_frame`` argument, returning it unchanged."""
    if imu_frame not in IMU_FRAMES:
        raise ValueError(
            f"imu_frame must be one of {IMU_FRAMES}, got {imu_frame!r}"
        )
    return imu_frame


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

class Hdf5Recording(NamedTuple):
    """One synced HDF5 recording. ``imu`` is [N, 6] in ``[accel, gyro]`` order."""

    time: np.ndarray         # [N]
    imu: np.ndarray          # [N, 6]
    pos: np.ndarray          # [N, 3] tango_pos
    orientation: np.ndarray  # [N, 4] game_rv, (w, x, y, z)


def hdf5_stores_world_imu(path):
    """Whether an HDF5's ``synced/acce|gyro`` are already world frame.

    gen_world exports written by :func:`ldm.evaluation.report.write_gen_world_hdf5`
    record ``imu_frame='world'`` and keep the source recording's (non-identity)
    ``game_rv``, so rotating them again would corrupt the signal. Real
    recordings carry no such attribute and are device frame.
    """
    import h5py

    with h5py.File(path, "r") as f:
        return str(f.attrs.get("imu_frame", "")) == "world"


def load_hdf5(path, imu_frame, already_world=False, remove_gravity=False,
              validate_time=False):
    """Load a synced HDF5 recording in the requested IMU frame.

    The stored ``synced/acce|gyro`` are device frame for real recordings and
    world frame for gen_world exports. So the IMU is rotated by ``game_rv``
    only for ``imu_frame='world'`` on a real recording; ``imu_frame='local'``
    and world-frame storage both return the stored IMU untouched.

    Whether the file is already world frame is read from its ``imu_frame``
    attribute, so callers cannot silently double-rotate a generated recording.
    ``already_world=True`` forces that interpretation for exports predating the
    attribute.
    """
    import h5py
    from scipy.spatial.transform import Rotation

    check_imu_frame(imu_frame)
    already_world = already_world or hdf5_stores_world_imu(path)
    if already_world and imu_frame == "local":
        raise ValueError(
            f"{path} stores world-frame IMU, so it cannot be loaded as "
            f"imu_frame='local'; there is no rotation back to the device frame"
        )

    with h5py.File(path, "r") as f:
        time = np.copy(f["synced/time"]).astype(np.float64)
        pos = np.copy(f["synced/tango_pos"]).astype(np.float64)
        orientation = np.copy(f["synced/game_rv"]).astype(np.float64)  # (w,x,y,z)
        gyro = np.copy(f["synced/gyro"]).astype(np.float64)
        acce = np.copy(
            f["synced/linacce" if remove_gravity else "synced/acce"]
        ).astype(np.float64)

    if imu_frame == "world" and not already_world:
        # HDF5 stores (w, x, y, z); scipy expects (x, y, z, w)
        rot = Rotation.from_quat(orientation[:, [1, 2, 3, 0]])
        acce, gyro = rot.apply(acce), rot.apply(gyro)

    imu = np.concatenate([acce, gyro], axis=1)
    if not validate_time:
        return Hdf5Recording(time, imu, pos, orientation)

    n = min(len(time), len(imu), len(pos), len(orientation))
    if n < 2 or np.any(np.diff(time[:n]) <= 0):
        raise ValueError(f"{path} has non-increasing or insufficient timestamps")
    return Hdf5Recording(time[:n], imu[:n], pos[:n], orientation[:n])


def load_hdf5_world(path):
    """A real recording as world-frame IMU plus its trajectory, length-aligned.

    Thin wrapper over :func:`load_hdf5` for the gen_world export path, where an
    untouched tail has to match the LDM's float32 world-frame output.

    Returns (time [N], position [N,3], orientation [N,4] wxyz, imu [N,6]).
    """
    rec = load_hdf5(path, imu_frame="world", validate_time=True)
    return rec.time, rec.pos, rec.orientation, rec.imu.astype(np.float32)


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

def _export_world_to_sim(v, heading_rad=0.0):
    """Invert juxta-simulator-rs map_world_frame (heading, then Y-up -> Z-up)."""
    ch, sh = np.cos(heading_rad), np.sin(heading_rad)
    ex, ey, ez = v[:, 0], v[:, 1], v[:, 2]
    bx = ch * ex + sh * ey
    by = -sh * ex + ch * ey
    bz = ez
    # inv map_world_axes: (bx,by,bz)=(sx,-sz,sy) -> (sx,sy,sz)=(bx,bz,-by)
    return np.stack([bx, bz, -by], axis=1)


def world_imu_to_local_parquet(accel_w, gyro_w, phone_rot_xyzw, heading_rad=0.0):
    """Rotate exported world-frame parquet IMU back to the phone-local frame.

    Matches the simulator: ``accel_local = phone_rot^-1 * accel_world_sim``,
    after undoing the export axis remap and heading applied to world columns.
    """
    from scipy.spatial.transform import Rotation

    accel_sim = _export_world_to_sim(accel_w, heading_rad)
    gyro_sim = _export_world_to_sim(gyro_w, heading_rad)
    rot = Rotation.from_quat(phone_rot_xyzw)
    return rot.inv().apply(accel_sim), rot.inv().apply(gyro_sim)


def local_imu_to_world_parquet(accel, gyro, phone_rot_xyzw, heading_rad=0.0):
    """Phone-local IMU to exported Z-up world axes, including heading."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_quat(phone_rot_xyzw)
    ch, sh = np.cos(heading_rad), np.sin(heading_rad)

    def exported(v):
        sx, sy, sz = rot.apply(v).T
        return np.column_stack([ch * sx + sh * sz, sh * sx - ch * sz, sy])

    return exported(accel), exported(gyro)


def imu_to_world(path, imu, imu_frame, world_heading=0.0):
    """Convert model-order IMU (raw or reconstructed) on the source time grid."""
    import os
    import h5py
    import pandas as pd
    from scipy.spatial.transform import Rotation

    check_imu_frame(imu_frame)
    if imu_frame == "world":
        return np.asarray(imu)
    if os.path.splitext(str(path))[1].lower() in HDF5_EXTS:
        with h5py.File(path, "r") as f:
            orientation = np.asarray(f["synced/game_rv"])[:len(imu)]
        rot = Rotation.from_quat(orientation[:, [1, 2, 3, 0]])
        return np.concatenate([rot.apply(imu[:, :3]), rot.apply(imu[:, 3:])], axis=1)
    df = pd.read_parquet(path)
    orientation = df[PHONE_ROT_COLS].to_numpy()[:len(imu)]
    return np.concatenate(local_imu_to_world_parquet(
        imu[:, :3], imu[:, 3:], orientation, world_heading,
    ), axis=1)


def load_parquet_imu(path, imu_frame, parquet_from_world=False,
                     world_heading=0.0, n_samples=None, verbose=True):
    """Load [N, 6] ``[accel, gyro]`` IMU from a simulator parquet.

    ``imu_frame='world'`` reads the ``*_world_*`` columns directly, including
    the synthetic IMU used for LDM conditioning or img2img initialization.
    ``imu_frame='local'`` reads ``*_local_*``, or rotates world -> local via
    ``phone_rot`` when ``parquet_from_world`` is set or local columns are absent.

    ``n_samples`` truncates the result, for aligning against a shorter pair.
    """
    import pandas as pd

    check_imu_frame(imu_frame)
    path = str(path)
    path = path if path.endswith(".parquet") else path + ".parquet"
    df = pd.read_parquet(path)
    has_local = all(c in df.columns for c in ACCEL_LOCAL_COLS + GYRO_LOCAL_COLS)
    has_world = all(c in df.columns for c in ACCEL_WORLD_COLS + GYRO_WORLD_COLS)
    has_rot = all(c in df.columns for c in PHONE_ROT_COLS)

    def result(accel, gyro):
        imu = np.concatenate([accel, gyro], axis=1)
        return imu if n_samples is None else imu[:n_samples]

    if imu_frame == "world":
        if not has_world:
            raise ValueError(f"{path}: missing world IMU columns for imu_frame=world")
        if verbose:
            print("  parquet IMU: world columns")
        return result(df[ACCEL_WORLD_COLS].values.astype(np.float64),
                      df[GYRO_WORLD_COLS].values.astype(np.float64))

    if not (parquet_from_world or not has_local):
        if verbose:
            print("  parquet IMU: local columns")
        return result(df[ACCEL_LOCAL_COLS].values.astype(np.float64),
                      df[GYRO_LOCAL_COLS].values.astype(np.float64))

    if not has_world:
        raise ValueError(
            f"{path}: need world IMU columns to rotate to local "
            f"(parquet_from_world={parquet_from_world}, has_local={has_local})"
        )
    if not has_rot:
        raise ValueError(f"{path}: missing phone_rot_*; cannot rotate world IMU to local")

    if verbose:
        print(f"  parquet IMU: world->local via phone_rot (heading={world_heading:.4f} rad)"
              + ("; local columns ignored" if has_local else ""))
    return result(*world_imu_to_local_parquet(
        df[ACCEL_WORLD_COLS].values.astype(np.float64),
        df[GYRO_WORLD_COLS].values.astype(np.float64),
        df[PHONE_ROT_COLS].values.astype(np.float64),
        heading_rad=world_heading,
    ))


def load_parquet_position(path):
    """Load horizontal trajectory and timestamps from a simulator parquet.

    Prefers ``agent_pos_x``/``agent_pos_z`` (z-up horizontal plane). Returns
    (pos [N,2] or None, time [N] or None).
    """
    import pandas as pd

    df = pd.read_parquet(path)
    time = df["time"].values.astype(np.float64) if "time" in df.columns else None
    if "agent_pos_x" in df.columns and "agent_pos_z" in df.columns:
        pos = df[["agent_pos_x", "agent_pos_z"]].values.astype(np.float64)
    elif "agent_pos_x" in df.columns and "agent_pos_y" in df.columns:
        pos = df[["agent_pos_x", "agent_pos_y"]].values.astype(np.float64)
    else:
        return None, time
    return pos, time


def load_sim_on_timeline(path, time):
    """World sim IMU aligned by elapsed time, returning only the covered span."""
    import pandas as pd

    imu = load_parquet_imu(str(path), "world")
    sim_time = pd.read_parquet(path)["time"].to_numpy(dtype=np.float64)
    keep = np.r_[True, np.diff(sim_time) != 0]
    sim_time, imu = sim_time[keep], imu[keep]
    if len(sim_time) < 2 or np.any(np.diff(sim_time) <= 0):
        raise ValueError(f"{path} requires increasing timestamps")
    sim_time = sim_time - sim_time[0]
    target = np.asarray(time, dtype=np.float64)
    target = target - target[0]
    target = target[target <= sim_time[-1] + 1e-4]
    return _interpolate_linear(imu, sim_time, target)


# ---------------------------------------------------------------------------
# Frame-aware trajectory loading (classifier / generation entry point)
# ---------------------------------------------------------------------------

def load_trajectory_imu(path, imu_frame, already_world=False,
                        parquet_from_world=False, world_heading=0.0):
    """Load one recording as [N, 6] ``[accel, gyro]`` IMU plus pos and time.

    Returns (imu [N,6], pos [N,2|3] or None, time [N] or None).
    """
    import os

    check_imu_frame(imu_frame)
    if already_world and imu_frame == "local":
        raise ValueError("Generated world-frame IMU cannot be used with a local-frame model")
    ext = os.path.splitext(path)[1].lower()

    if ext in HDF5_EXTS:
        rec = load_hdf5(path, imu_frame=imu_frame, already_world=already_world)
        if already_world:
            print("  hdf5 IMU: world (synced as-is, no game_rv)")
        else:
            print(f"  hdf5 IMU: "
                  f"{'local' if imu_frame == 'local' else 'world (via game_rv)'}")
        return rec.imu, rec.pos, rec.time

    if ext == ".parquet":
        imu = load_parquet_imu(
            path, imu_frame=imu_frame, parquet_from_world=parquet_from_world,
            world_heading=world_heading,
        )
        pos, time = load_parquet_position(path)
        return imu, pos, time

    raise ValueError(f"Unsupported file type: {ext}")


def load_activity_imu(path, classifier_imu, imu_frame, already_world=False,
                      parquet_from_world=False, world_heading=0.0):
    """Device-frame IMU used to derive stationary labels, when obtainable.

    Falls back to the classifier-frame IMU whenever the local frame cannot be
    recovered (gen_world HDF5, parquet without ``phone_rot``).
    """
    import os

    if imu_frame == "local":
        return classifier_imu

    ext = os.path.splitext(path)[1].lower()
    if ext in HDF5_EXTS and not already_world and not hdf5_stores_world_imu(path):
        return load_hdf5(path, imu_frame="local").imu
    if ext == ".parquet" and not already_world:
        try:
            return load_parquet_imu(
                path, imu_frame="local", parquet_from_world=parquet_from_world,
                world_heading=world_heading,
            )
        except (KeyError, ValueError) as exc:
            print(f"  [warn] could not load local IMU for activity labels ({exc}); "
                  "using classifier-frame IMU")
    return classifier_imu


def load_pair_sequence(trajectory_path, synthetic_path, real_path=None):
    """Load a pair folder's trajectory plus synthetic IMU, real IMU optional.

    Returns (time, position, orientation, sim_imu [N,6], real_imu [N,6] or None)
    with both IMUs in ``[accel, gyro]`` world-frame order. Without ``real_path``
    the length comes from the trajectory and synthetic parquet alone.
    """
    import pandas as pd

    trajectory = np.loadtxt(trajectory_path, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] < 8:
        raise ValueError(
            f"{trajectory_path} must have columns: time, xyz, quaternion(wxyz)"
        )

    sim = pd.read_parquet(synthetic_path)
    columns = ACCEL_WORLD_COLS + GYRO_WORLD_COLS
    missing = [name for name in columns if name not in sim.columns]
    if missing:
        raise ValueError(f"{synthetic_path} missing columns: {missing}")

    lengths = [len(trajectory), len(sim)]
    real_imu = None
    if real_path is not None:
        _, _, _, real_imu = load_hdf5_world(real_path)
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


# ---------------------------------------------------------------------------
# Window builders (shared with the preprocessing scripts)
# ---------------------------------------------------------------------------

def window_conditioning(time, pos_2d, start, end, latent_length=LATENT_LENGTH):
    """Latent-length velocity [2, L] and normalized time [1, L] for one window.

    The single definition of the LDM's conditioning signals, shared by the
    preprocessing window builders here and by
    :func:`ldm.evaluation.windows.build_conditioning` at inference time, so
    training-time and generation-time conditioning cannot drift apart.

    Velocity is left in physical units; callers standardize with the LDM's
    velocity statistics.
    """
    # Relative time avoids float32 precision collapse on large absolute timestamps
    t_rel = np.asarray(time[start:end], dtype=np.float64) - float(time[start])
    pos = np.asarray(pos_2d[start:end], dtype=np.float64)[:, :2]

    dt = torch.tensor(np.diff(t_rel), dtype=torch.float32)
    # Subtract absolute positions in float64 before converting to model dtype.
    dp = torch.tensor(np.diff(pos, axis=0).T, dtype=torch.float32)
    vel = dp / dt.unsqueeze(0).clamp_min(1e-8)
    vel = torch.cat([vel[:, :1], vel], dim=1)  # [2, W] pad first
    velocity = F.interpolate(
        vel.unsqueeze(0), size=latent_length, mode="linear", align_corners=True
    ).squeeze(0)  # [2, L]

    # Normalize in float64 before casting; float32 loses resolution here.
    t_norm = torch.tensor(t_rel / (t_rel[-1] + 1e-8), dtype=torch.float32)
    physical_time = F.interpolate(
        t_norm.view(1, 1, -1), size=latent_length, mode="linear", align_corners=True
    ).view(1, latent_length)

    return velocity, physical_time


def _window_payload(imu, pos_2d, time, start, end, latent_length):
    """Build one ``{imu, velocity, physical_time}`` window from raw arrays."""
    w_imu = torch.tensor(imu[start:end].T, dtype=torch.float32)  # [6, W]
    velocity, physical_time = window_conditioning(
        time, pos_2d, start, end, latent_length
    )
    return w_imu, velocity, physical_time


def process_file(path, window_samples, latent_length, imu_frame, stride=None,
                 remove_gravity=False, already_world=False):
    """Window one HDF5 recording into ``{imu, velocity, physical_time}`` dicts.

    ``stride`` defaults to ``window_samples`` (non-overlapping windows).
    """
    if stride is None:
        stride = window_samples
    rec = load_hdf5(path, imu_frame=imu_frame, already_world=already_world,
                    remove_gravity=remove_gravity)

    imu = rec.imu                    # [N, 6]
    time = rec.time
    pos_2d = rec.pos[:, [0, 1]]

    windows = []
    for start in range(0, len(imu) - window_samples + 1, stride):
        w_imu, velocity, physical_time = _window_payload(
            imu, pos_2d, time, start, start + window_samples, latent_length
        )
        windows.append({
            "imu": w_imu,
            "velocity": velocity,
            "physical_time": physical_time,
        })
    return windows


def process_pair(real_path, sim_path, window_samples, latent_length, stride=None,
                 remove_gravity=False):
    """Window an aligned real HDF5 / synthetic parquet pair.

    Real IMU is rotated to world frame via ``game_rv``; sim IMU uses the
    parquet's world-frame columns directly. Returns dicts that additionally
    carry ``sim_imu``.
    """
    import pandas as pd

    if stride is None:
        stride = window_samples

    rec = load_hdf5(real_path, imu_frame="world", remove_gravity=remove_gravity)
    df = pd.read_parquet(sim_path)
    sim = np.concatenate([
        df[ACCEL_WORLD_COLS].values, df[GYRO_WORLD_COLS].values,
    ], axis=1)

    # Use the minimum length to absorb minor length mismatches
    n = min(len(rec.imu), len(sim))
    real_imu = rec.imu[:n]
    sim_imu = sim[:n]
    pos_2d = rec.pos[:n, [0, 1]]
    time = rec.time[:n]

    windows = []
    for start in range(0, n - window_samples + 1, stride):
        end = start + window_samples
        w_imu, velocity, physical_time = _window_payload(
            real_imu, pos_2d, time, start, end, latent_length
        )
        windows.append({
            "imu": w_imu,
            "sim_imu": torch.tensor(sim_imu[start:end].T, dtype=torch.float32),
            "velocity": velocity,
            "physical_time": physical_time,
        })
    return windows


def compute_stats(windows):
    """Per-channel mean/std over a list of window dicts (real IMU only)."""
    all_imu = torch.stack([w["imu"] for w in windows])
    all_vel = torch.stack([w["velocity"] for w in windows])
    return {
        "imu_mean": all_imu.mean(dim=(0, 2)),
        "imu_std": all_imu.std(dim=(0, 2)) + 1e-8,
        "vel_mean": all_vel.mean(dim=(0, 2)),
        "vel_std": all_vel.std(dim=(0, 2)) + 1e-8,
    }


def split_by_source(windows_by_source, val_fraction, seed):
    """Split source names into (train, val), keeping a source whole.

    Splitting per source rather than per window keeps windows from one
    recording out of both halves, which would otherwise leak through the
    overlap between neighbouring windows. A single source lands in both, so
    overfit runs still have a validation set.
    """
    import random

    names = sorted(windows_by_source)
    if not names:
        raise ValueError("no windows to split; check the input paths and lengths")
    if len(names) == 1:
        return names, names

    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    n_val = min(max(1, int(len(shuffled) * val_fraction)), len(shuffled) - 1)
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def write_window_dataset(output_dir, windows_by_source, val_fraction, seed,
                         source_label="files"):
    """Split, save ``train/`` + ``val/`` windows, ``stats.pt`` and ``split.txt``.

    The tail shared by every preprocessing script. Statistics come from the
    training split only, so validation windows never influence the
    normalization the model is trained under.

    Returns the computed statistics.
    """
    import os

    train_names, val_names = split_by_source(windows_by_source, val_fraction, seed)
    train_windows = [w for name in train_names for w in windows_by_source[name]]
    val_windows = [w for name in val_names for w in windows_by_source[name]]
    print(f"Split: {len(train_names)} train {source_label} "
          f"({len(train_windows)} windows), "
          f"{len(val_names)} val {source_label} ({len(val_windows)} windows)")

    stats = compute_stats(train_windows)
    for key in ("imu_mean", "imu_std", "vel_mean", "vel_std"):
        print(f"{key:9s} {stats[key].numpy()}")

    for split, split_windows in (("train", train_windows), ("val", val_windows)):
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i, window in enumerate(split_windows):
            torch.save(window, os.path.join(split_dir, f"window_{i:04d}.pt"))
    torch.save(stats, os.path.join(output_dir, STATS_NAME))

    split_path = os.path.join(output_dir, "split.txt")
    with open(split_path, "w") as f:
        f.write(f"# seed={seed} val_fraction={val_fraction}\n")
        f.write(f"# {len(train_names)} train {source_label}, "
                f"{len(val_names)} val {source_label}\n")
        for split, names in (("train", train_names), ("val", val_names)):
            f.write(f"\n[{split}]\n")
            f.write("".join(f"{name}\n" for name in names))
    print(f"Wrote split list to {split_path}")
    print(f"Saved to {output_dir}")
    return stats


# ---------------------------------------------------------------------------
# On-the-fly window datasets
# ---------------------------------------------------------------------------

class HDF5WindowDataset(Dataset):
    """Non-overlapping windows of one HDF5 recording, standardized for eval."""

    def __init__(self, hdf5_path, stats, imu_frame, window_sec=10.0,
                 sample_rate=SAMPLE_RATE, latent_length=LATENT_LENGTH,
                 already_world=False, stride_sec=None):
        self.imu_mean = stats["imu_mean"]
        self.imu_std = stats["imu_std"]
        self.vel_mean = stats.get("vel_mean")
        self.vel_std = stats.get("vel_std")
        window_samples = int(window_sec * sample_rate)
        stride = window_samples if stride_sec is None else int(stride_sec * sample_rate)
        self.windows = process_file(
            hdf5_path, window_samples, latent_length, imu_frame,
            stride=stride, already_world=already_world,
        )
        if not self.windows:
            raise ValueError(f"No full windows found in {hdf5_path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data = self.windows[idx]
        item = {
            "imu": (data["imu"] - self.imu_mean) / self.imu_std,
            "imu_raw": data["imu"],
            "physical_time": data["physical_time"],
        }
        if self.vel_mean is None:
            item["velocity"] = data["velocity"]
        else:
            item["velocity"] = (data["velocity"] - self.vel_mean) / self.vel_std
        return item

    def inverse_standardize_imu(self, imu):
        return imu * self.imu_std.to(imu.device) + self.imu_mean.to(imu.device)


class PairedWindowDataset(Dataset):
    """Non-overlapping windows of an aligned real HDF5 / synthetic parquet pair."""

    def __init__(self, real_path, sim_path, stats, window_sec=10.0,
                 sample_rate=SAMPLE_RATE, latent_length=LATENT_LENGTH,
                 stride_sec=None):
        self.imu_mean = stats["imu_mean"]
        self.imu_std = stats["imu_std"]
        self.vel_mean = stats["vel_mean"]
        self.vel_std = stats["vel_std"]
        window_samples = int(window_sec * sample_rate)
        stride = window_samples if stride_sec is None else int(stride_sec * sample_rate)
        self.windows = process_pair(
            real_path, sim_path, window_samples, latent_length, stride=stride,
        )
        if not self.windows:
            raise ValueError(f"No full windows found in pair {real_path} / {sim_path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data = self.windows[idx]
        return {
            "imu": (data["imu"] - self.imu_mean) / self.imu_std,
            "sim_imu": (data["sim_imu"] - self.imu_mean) / self.imu_std,
            "imu_raw": data["imu"],
            "velocity": (data["velocity"] - self.vel_mean) / self.vel_std,
            "physical_time": data["physical_time"],
        }

    def inverse_standardize_imu(self, imu):
        return imu * self.imu_std.to(imu.device) + self.imu_mean.to(imu.device)


class RecordingWindowDataset(Dataset):
    """VAE windows in its training frame, retaining the source timeline."""

    def __init__(self, path, stats, imu_frame, window_sec=10.0,
                 sample_rate=SAMPLE_RATE, already_world=False,
                 parquet_from_world=False, world_heading=0.0):
        self.imu, self.pos, self.time = load_trajectory_imu(
            path, imu_frame, already_world, parquet_from_world, world_heading,
        )
        self.window = int(window_sec * sample_rate)
        if self.window < 2:
            raise ValueError("a VAE window must contain at least two samples")
        self.imu_mean, self.imu_std = stats["imu_mean"], stats["imu_std"]
        if len(self) == 0:
            raise ValueError(f"No full windows found in {path}")

    def __len__(self):
        return len(self.imu) // self.window

    def __getitem__(self, index):
        start = index * self.window
        raw = torch.tensor(self.imu[start:start + self.window].T, dtype=torch.float32)
        return {"imu": (raw - self.imu_mean) / self.imu_std, "imu_raw": raw}


def resolve_window_dataset(path, stats, imu_frame, sim_path=None, window_sec=10.0,
                           sample_rate=SAMPLE_RATE, latent_length=LATENT_LENGTH,
                           already_world=False, parquet_from_world=False,
                           world_heading=0.0):
    """VAE windows in the requested frame, or paired world-frame LDM windows."""
    import os

    if sim_path is not None:
        if imu_frame != "world":
            raise ValueError("Paired LDM windows require world-frame IMU")
        dataset = PairedWindowDataset(
            path, sim_path, stats, window_sec=window_sec,
            sample_rate=sample_rate, latent_length=latent_length,
        )
        return dataset, f"pair:{os.path.basename(os.path.dirname(path))}"

    ext = os.path.splitext(path)[1].lower()
    if ext in HDF5_EXTS or ext == ".parquet":
        dataset = RecordingWindowDataset(
            path, stats, imu_frame, window_sec, sample_rate, already_world,
            parquet_from_world, world_heading,
        )
        kind = "parquet" if ext == ".parquet" else "hdf5"
        return dataset, f"{kind}:{os.path.basename(path)}"
    raise ValueError(f"--input must be .hdf5/.h5/.parquet, got {ext}")


# ---------------------------------------------------------------------------
# RoNIN-shaped sequence adapters
# ---------------------------------------------------------------------------

def _interpolate_linear(data, ts_in, ts_out):
    """Linearly interpolate an [N, D] array from ts_in onto ts_out."""
    out = np.zeros((len(ts_out), data.shape[1]), dtype=data.dtype)
    for d in range(data.shape[1]):
        out[:, d] = np.interp(ts_out, ts_in, data[:, d])
    return out


class ParquetSequence:
    """Simulator parquet as a RoNIN sequence (2D velocity target).

    Reimplements SimParquetSequence without importing the full RoNIN module
    graph. Features are ``[gyro, accel]`` world frame at ``SAMPLE_RATE``.
    """

    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        self.w = kwargs.get("interval", 1)
        self.window_size = kwargs.get("window_size", 200)
        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        import pandas as pd

        pq = path if path.endswith(".parquet") else path + ".parquet"
        df = pd.read_parquet(pq)
        df = df.drop_duplicates(subset="time", keep="first").reset_index(drop=True)

        ts_in = df["time"].values.astype(np.float64)
        gyro = df[GYRO_WORLD_COLS].values.astype(np.float64)
        accel = df[ACCEL_WORLD_COLS].values.astype(np.float64)
        # agent_pos_{x,z} is already the 2D horizontal position in the z-up frame
        pos = df[["agent_pos_x", "agent_pos_z"]].values.astype(np.float64)

        dt_med = np.median(np.diff(ts_in))
        freq = 1.0 / dt_med if dt_med > 0 else float("nan")
        if abs(freq - SAMPLE_RATE) < 1.0 and np.allclose(np.diff(ts_in), dt_med, atol=1e-4):
            ts = ts_in
        else:
            ts = np.arange(ts_in[0], ts_in[-1] - 1e-6, 1.0 / SAMPLE_RATE)
            gyro = _interpolate_linear(gyro, ts_in, ts)
            accel = _interpolate_linear(accel, ts_in, ts)
            pos = _interpolate_linear(pos, ts_in, ts)

        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (pos[self.w:] - pos[:-self.w]) / dt

        self.features = np.concatenate([gyro, accel], axis=1)
        # Keep the full timeline for LDM conditioning; RoNIN targets stay truncated.
        self.ts_full = ts
        self.gt_pos_full = pos
        n = max(1, len(glob_v) - self.window_size)
        self.ts = ts[:n]
        self.targets = glob_v[:n, :2]
        # Pad to 3 cols for the RoNIN aux layout (ts + ori + pos = 8)
        self.gt_pos = np.concatenate([pos[:n], np.zeros((n, 1))], axis=1)
        self.orientations = np.zeros((n, 4))
        self.orientations[:, 0] = 1.0

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        n = min(len(self.ts), len(self.orientations), len(self.gt_pos))
        return np.concatenate(
            [self.ts[:n, None], self.orientations[:n], self.gt_pos[:n]], axis=1
        )

    def get_meta(self):
        return "parquet"


class HDF5Sequence:
    """Real or gen_world HDF5 as a RoNIN sequence (2D velocity target).

    Reimplements HybridGlobSpeedSequence for real recordings, whose
    device-frame IMU is rotated to the global frame with ``game_rv``. gen_world
    exports already store world-frame IMU alongside the source recording's
    ``game_rv``, so for those the rotation is skipped, matching juxta-ronin's
    GenWorldSequence.
    """

    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        self.w = kwargs.get("interval", 1)
        self.window_size = kwargs.get("window_size", 200)
        self.already_world = kwargs.get("already_world", False)
        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        import h5py
        import quaternion

        hdf5_path = path if path.endswith(HDF5_EXTS) else path + ".hdf5"

        with h5py.File(hdf5_path, "r") as f:
            already_world = self.already_world or str(f.attrs.get("imu_frame", "")) == "world"
            ts = np.copy(f["synced/time"]).astype(np.float64)
            gyro = np.copy(f["synced/gyro"]).astype(np.float64)
            acce = np.copy(f["synced/acce"]).astype(np.float64)
            tango_pos = np.copy(f["synced/tango_pos"]).astype(np.float64)
            ori = np.copy(f["synced/game_rv"]).astype(np.float64)

        if already_world:
            glob_gyro, glob_acce = gyro, acce
        else:
            # game_rv is (w, x, y, z), matching RoNIN change_cf / numpy-quaternion.
            ori_q = quaternion.from_float_array(ori)
            nz = np.zeros((len(acce), 1))
            gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
            acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))
            glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
            glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        n_min = min(len(ts) - self.w, len(tango_pos) - self.w)
        dt = (ts[self.w:self.w + n_min] - ts[:n_min])[:, None]
        glob_v = (tango_pos[self.w:self.w + n_min] - tango_pos[:n_min]) / dt

        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)
        self.ts = ts
        self.ts_full = ts
        self.targets = glob_v[:, :2]
        self.orientations = ori
        self.gt_pos = tango_pos
        self.gt_pos_full = tango_pos

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        n = min(len(self.ts), len(self.orientations), len(self.gt_pos))
        return np.concatenate(
            [self.ts[:n, None], self.orientations[:n], self.gt_pos[:n]], axis=1
        )

    def get_meta(self):
        return "hdf5"


class ArraySequence(HDF5Sequence):
    """Already-world model-order IMU and trajectory, resampled for RoNIN."""

    def __init__(self, data_path=None, *, time, position, imu, **kwargs):
        super().__init__(None, **kwargs)
        if time is None or position is None:
            raise ValueError("RoNIN evaluation requires timestamps and a trajectory")
        ts = np.asarray(time, dtype=np.float64)
        pos = np.asarray(position, dtype=np.float64)
        imu = np.asarray(imu, dtype=np.float64)
        if len(ts) != len(imu) or len(pos) != len(imu):
            raise ValueError("IMU, timestamps and trajectory must have equal lengths")
        keep = np.r_[True, np.diff(ts) != 0]
        ts, pos, imu = ts[keep], pos[keep], imu[keep]
        if len(ts) < 2 or np.any(np.diff(ts) <= 0):
            raise ValueError("RoNIN requires increasing timestamps")
        if not np.allclose(np.diff(ts), 1.0 / SAMPLE_RATE, atol=1e-4, rtol=0):
            uniform = np.arange(ts[0], ts[-1] - 1e-6, 1.0 / SAMPLE_RATE)
            pos = _interpolate_linear(pos, ts, uniform)
            imu = _interpolate_linear(imu, ts, uniform)
            ts = uniform
        if len(ts) <= self.w:
            raise ValueError("Sequence is shorter than a RoNIN window")
        if pos.shape[1] == 2:
            pos = np.column_stack([pos, np.zeros(len(pos))])
        self.features = imu[:, [3, 4, 5, 0, 1, 2]]
        self.ts = self.ts_full = ts
        self.gt_pos = self.gt_pos_full = pos
        self.orientations = np.tile([1., 0., 0., 0.], (len(ts), 1))
        self.targets = ((pos[self.w:] - pos[:-self.w]) /
                        (ts[self.w:] - ts[:-self.w])[:, None])[:, :2]
