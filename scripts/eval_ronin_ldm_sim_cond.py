"""Evaluate RoNIN trajectory on original vs sim-conditioned LDM-generated IMU.

Same features as eval_ronin_ldm.py, but the LDM is additionally conditioned on
a VAE-encoded synthetic IMU latent (alongside trajectory). Requires a paired
synthetic IMU source (--sim_input or --pair_dir).

Example usage (pair folder):
    python scripts/eval_ronin_ldm_sim_cond.py \
        --pair_dir data/real_sim_imu_pairs/chest/john_chest_ios_corrected \
        --ldm_ckpt logs/ldm_1d_sim_cond/.../best-000.ckpt \
        --first_stage_ckpt logs/vae_1d/.../best-000.ckpt \
        --ronin_ckpt path/to/ronin/checkpoint_latest.pt \
        --stats data/real_sim_pairs_processed/stats.pt \
        --outdir outputs/ronin_ldm_sim_cond_eval

Example usage (explicit paths):
    python scripts/eval_ronin_ldm_sim_cond.py \
        --imu_input data/real_sim_imu_pairs/chest/john_chest_ios_corrected/real.hdf5 \
        --sim_input data/real_sim_imu_pairs/chest/john_chest_ios_corrected/synthetic.parquet \
        --ldm_ckpt logs/ldm_1d_sim_cond/.../best-000.ckpt \
        --first_stage_ckpt logs/vae_1d/.../best-000.ckpt \
        --ronin_ckpt path/to/ronin/checkpoint_latest.pt \
        --stats data/real_sim_pairs_processed/stats.pt \
        --outdir outputs/ronin_ldm_sim_cond_eval
"""

import argparse
import copy
import json
import os
import sys
from contextlib import nullcontext
from os import path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quaternion
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))

from ldm.models.diffusion.ddim_1d import DDIMSampler1D
from ldm.util import instantiate_from_config

RONIN_ROOT = osp.normpath(osp.join(osp.dirname(__file__), "..", "..", "juxta-ronin"))

SAMPLE_RATE = 200
LDM_WINDOW = 2000  # 10s at 200 Hz
LATENT_LENGTH = 100


# ---------------------------------------------------------------------------
# Lightweight data loaders (avoid importing full RoNIN module graph)
# ---------------------------------------------------------------------------

def _interpolate_linear(data, ts_in, ts_out):
    """Linearly interpolate [N, D] array from ts_in to ts_out."""
    out = np.zeros((len(ts_out), data.shape[1]), dtype=data.dtype)
    for d in range(data.shape[1]):
        out[:, d] = np.interp(ts_out, ts_in, data[:, d])
    return out


class _ParquetSequence:
    """Minimal reimplementation of SimParquetSequence (2D velocity target)."""

    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        self.w = kwargs.get("interval", 1)
        self.window_size = kwargs.get("window_size", 200)
        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        pq = path if path.endswith(".parquet") else path + ".parquet"
        df = pd.read_parquet(pq)
        df = df.drop_duplicates(subset="time", keep="first").reset_index(drop=True)

        ts_in = df["time"].values.astype(np.float64)
        gyro = df[["gyro_world_x", "gyro_world_y", "gyro_world_z"]].values.astype(np.float64)
        accel = df[["accel_world_x", "accel_world_y", "accel_world_z"]].values.astype(np.float64)
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
        # Keep full timeline for LDM conditioning; RoNIN targets stay truncated.
        self.ts_full = ts
        self.gt_pos_full = pos  # [N, 2] horizontal plane
        n_full = len(glob_v)
        n = max(1, n_full - self.window_size)
        self.ts = ts[:n]
        self.targets = glob_v[:n, :2]
        # Pad to 3 cols for RoNIN aux layout (ts + ori + pos = 8)
        self.gt_pos = np.concatenate([pos[:n], np.zeros((n, 1))], axis=1)
        self.orientations = np.zeros((n, 4))
        self.orientations[:, 0] = 1.0

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        n = min(len(self.ts), len(self.orientations), len(self.gt_pos))
        return np.concatenate([self.ts[:n, None], self.orientations[:n], self.gt_pos[:n]], axis=1)

    def get_meta(self):
        return "parquet (eval_ronin_ldm)"


class _HDF5Sequence:
    """Minimal reimplementation of HybridGlobSpeedSequence (2D velocity target)."""

    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        self.w = kwargs.get("interval", 1)
        self.window_size = kwargs.get("window_size", 200)
        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        hdf5_path = path if path.endswith(".hdf5") else path + ".hdf5"

        with h5py.File(hdf5_path, "r") as f:
            ts = np.copy(f["synced/time"]).astype(np.float64)
            gyro = np.copy(f["synced/gyro"]).astype(np.float64)
            acce = np.copy(f["synced/acce"]).astype(np.float64)
            tango_pos = np.copy(f["synced/tango_pos"]).astype(np.float64)
            ori = np.copy(f["synced/game_rv"]).astype(np.float64)

        # Rotate device-frame IMU to global frame via game_rv quaternions.
        # game_rv is stored as (w, x, y, z) -- matches RoNIN change_cf / numpy-quaternion.
        ori_q = quaternion.from_float_array(ori)
        nz = np.zeros((len(acce), 1))
        gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        n_min = min(len(ts) - self.w, len(tango_pos) - self.w)
        dt = (ts[self.w:self.w + n_min] - ts[:n_min])[:, None]
        pos_diff = tango_pos[self.w:self.w + n_min] - tango_pos[:n_min]
        glob_v = pos_diff / dt

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
        return np.concatenate([self.ts[:n, None], self.orientations[:n], self.gt_pos[:n]], axis=1)

    def get_meta(self):
        return "hdf5 (eval_ronin_ldm)"


class _StridedDataset(Dataset):
    """Strided windowing dataset compatible with RoNIN's run_test loop."""

    def __init__(self, seq_type, root_dir, data_list, step_size=10, window_size=200, **kwargs):
        super().__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.window_size = window_size
        self.step_size = step_size
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.ts_full, self.gt_pos_full = [], []
        self.features, self.targets = [], []

        for i, data in enumerate(data_list):
            seq = seq_type(osp.join(root_dir, data), interval=window_size,
                           window_size=window_size, **kwargs)
            feat = seq.get_feature()
            targ = seq.get_target()
            aux = seq.get_aux()

            self.features.append(feat)
            self.targets.append(targ)
            self.ts.append(aux[:, 0])
            self.orientations.append(aux[:, 1:5])
            self.gt_pos.append(aux[:, 5:])
            self.ts_full.append(getattr(seq, "ts_full", aux[:, 0]))
            self.gt_pos_full.append(getattr(seq, "gt_pos_full", aux[:, 5:]))

            self.index_map += [[i, j] for j in range(0, targ.shape[0], step_size)]

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item]
        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]
        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


# ---------------------------------------------------------------------------
# RoNIN model (inline to avoid importing tensorboardX / yaspin)
# ---------------------------------------------------------------------------

def _build_ronin_2d(arch, window_size):
    """Build 2D RoNIN ResNet1D (adds juxta-ronin/source to sys.path)."""
    sys.path.insert(0, RONIN_ROOT)
    from source.model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule

    _fc_config = {"fc_dim": 512, "in_dim": window_size // 32 + 1,
                  "dropout": 0.5, "trans_planes": 128}
    in_ch, out_ch = 6, 2
    if arch == "resnet18":
        groups = [2, 2, 2, 2]
    elif arch == "resnet50":
        groups = [3, 4, 6, 3]
        _fc_config["fc_dim"] = 1024
    elif arch == "resnet101":
        groups = [3, 4, 23, 3]
        _fc_config["fc_dim"] = 1024
    else:
        raise ValueError(f"Unknown arch {arch}")
    return ResNet1D(in_ch, out_ch, BasicBlock1D, groups,
                    base_plane=64, output_block=FCOutputModule,
                    kernel_size=3, **_fc_config)


def _build_ronin_3d(arch, window_size):
    sys.path.insert(0, RONIN_ROOT)
    from source.model_resnet1d_3d import ResNet1D, BasicBlock1D

    inter_dim = window_size // 32 + 1
    in_ch, out_ch = 6, 3
    if arch == "resnet18":
        groups = [2, 2, 2, 2]
    elif arch == "resnet50":
        groups = [3, 4, 6, 3]
    elif arch == "resnet101":
        groups = [3, 4, 23, 3]
    else:
        raise ValueError(f"Unknown arch {arch}")
    return ResNet1D(BasicBlock1D, in_ch, out_ch, groups, inter_dim)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_ldm(config_path, ckpt_path, device, first_stage_ckpt=None, scale_factor=None):
    config = OmegaConf.load(config_path)
    if first_stage_ckpt is not None:
        config.model.params.first_stage_ckpt = first_stage_ckpt
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load LDM weights from {ckpt_path} into architecture from "
            f"{config_path}. Pass the matching config via --ldm_config / --config.\n"
            f"Original error: {e}"
        ) from e
    if scale_factor is not None:
        model.register_buffer("scale_factor", torch.tensor(float(scale_factor)))
    return model.to(device).eval()


def load_vae_stats(stats_path):
    stats = torch.load(stats_path, weights_only=True)
    return (
        stats["imu_mean"].view(6, 1),
        stats["imu_std"].view(6, 1),
        stats["vel_mean"].view(2, 1),
        stats["vel_std"].view(2, 1),
    )


def ldm_to_ronin_channels(feat):
    """[accel(3), gyro(3)] -> [gyro(3), accel(3)]"""
    return feat[:, [3, 4, 5, 0, 1, 2]]


def ronin_to_ldm_channels(feat):
    """[gyro(3), accel(3)] -> [accel(3), gyro(3)]"""
    return feat[:, [3, 4, 5, 0, 1, 2]]


def write_generated_hdf5(src_path, out_path, features_gen_ronin):
    """Copy src HDF5 and replace synced/acce + synced/gyro with generated IMU.

    features_gen_ronin is world-frame [N, 6] in RoNIN order [gyro, accel];
    stored directly into synced/gyro and synced/acce (no local-frame rotation).
    All other datasets (including linacce) are left unchanged. Trailing
    samples beyond N keep the original IMU.
    """
    import shutil

    src_path = src_path if src_path.endswith((".hdf5", ".h5")) else src_path + ".hdf5"
    N = features_gen_ronin.shape[0]

    shutil.copy2(src_path, out_path)
    with h5py.File(out_path, "a") as f:
        if "synced" not in f or "acce" not in f["synced"] or "gyro" not in f["synced"]:
            raise ValueError(f"{src_path} missing synced/acce or synced/gyro")
        n_file = f["synced/acce"].shape[0]
        n_write = min(N, n_file)
        if n_write < N:
            print(f"  [warn] Truncating generated IMU {N} -> {n_write} to fit HDF5 length")

        gyro_world = features_gen_ronin[:n_write, :3]
        acce_world = features_gen_ronin[:n_write, 3:6]
        f["synced/acce"][:n_write] = acce_world
        f["synced/gyro"][:n_write] = gyro_world

    print(f"Wrote generated world-frame IMU HDF5 to {out_path} "
          f"({n_write}/{n_file} samples replaced)")
    return out_path


def write_generated_parquet(src_path, out_path, features_gen_ronin, gen_ts):
    """Copy src parquet and replace world-frame IMU columns with generated IMU.

    features_gen_ronin is world-frame [N, 6] in RoNIN order [gyro, accel],
    sampled on gen_ts. Values are written onto the (deduped) parquet time grid
    via linear interpolation when the grids differ (e.g. after 200 Hz resample).
    Local IMU columns and all other fields are left unchanged.
    """
    src_path = src_path if src_path.endswith(".parquet") else src_path + ".parquet"
    df = pd.read_parquet(src_path)
    df = df.drop_duplicates(subset="time", keep="first").reset_index(drop=True)
    t_df = df["time"].values.astype(np.float64)
    gen_ts = np.asarray(gen_ts, dtype=np.float64)
    N = features_gen_ronin.shape[0]
    gen_ts = gen_ts[:N]
    gyro_gen = features_gen_ronin[:N, :3]
    accel_gen = features_gen_ronin[:N, 3:6]

    same_grid = (
        len(gen_ts) == len(t_df)
        and np.allclose(gen_ts, t_df, atol=1e-4, rtol=0.0)
    )
    if same_grid:
        gyro_out, accel_out = gyro_gen, accel_gen
    else:
        print(f"  [info] Interpolating generated IMU ({N} @ gen_ts) onto "
              f"parquet time grid ({len(t_df)} samples)")
        # Only fill where parquet time falls inside the generated span
        gyro_out = np.column_stack([
            np.interp(t_df, gen_ts, gyro_gen[:, d], left=np.nan, right=np.nan)
            for d in range(3)
        ])
        accel_out = np.column_stack([
            np.interp(t_df, gen_ts, accel_gen[:, d], left=np.nan, right=np.nan)
            for d in range(3)
        ])
        # Keep original IMU outside the generated span
        mask = np.isfinite(gyro_out[:, 0])
        for d, col in enumerate(["gyro_world_x", "gyro_world_y", "gyro_world_z"]):
            vals = df[col].values.astype(np.float64)
            vals[mask] = gyro_out[mask, d]
            gyro_out[:, d] = vals
        for d, col in enumerate(["accel_world_x", "accel_world_y", "accel_world_z"]):
            vals = df[col].values.astype(np.float64)
            vals[mask] = accel_out[mask, d]
            accel_out[:, d] = vals

    df = df.copy()
    df["gyro_world_x"] = gyro_out[:, 0]
    df["gyro_world_y"] = gyro_out[:, 1]
    df["gyro_world_z"] = gyro_out[:, 2]
    df["accel_world_x"] = accel_out[:, 0]
    df["accel_world_y"] = accel_out[:, 1]
    df["accel_world_z"] = accel_out[:, 2]
    df.to_parquet(out_path, index=False)
    print(f"Wrote generated world-frame IMU parquet to {out_path} "
          f"({N} gen samples -> {len(df)} rows)")
    return out_path


def _window_conditioning(ts, gt_pos, start, end, vel_mean, vel_std, device):
    """Build LDM velocity + physical_time conditioning for one 10s window.

    Uses x/y from gt_pos (first two columns, horizontal plane in z-up frame).
    """
    t_win = ts[start:end]
    pos = gt_pos[start:end]
    pos_2d = pos[:, :2]

    t_rel = t_win - t_win[0]
    w_pos = torch.tensor(pos_2d.T, dtype=torch.float32)  # [2, 2000]
    dt = torch.tensor(np.diff(t_win), dtype=torch.float32)
    dp = w_pos[:, 1:] - w_pos[:, :-1]
    vel = dp / dt.unsqueeze(0).clamp_min(1e-8)
    vel = torch.cat([vel[:, :1], vel], dim=1)  # [2, 2000]

    vel_resampled = F.interpolate(
        vel.unsqueeze(0), size=LATENT_LENGTH, mode="linear", align_corners=True
    ).squeeze(0)  # [2, 100]
    velocity = ((vel_resampled - vel_mean.cpu()) / vel_std.cpu()).unsqueeze(0).to(device)

    t_norm = torch.tensor(t_rel / (t_rel[-1] + 1e-8), dtype=torch.float32)
    physical_time = F.interpolate(
        t_norm.view(1, 1, -1), size=LATENT_LENGTH, mode="linear", align_corners=True
    ).to(device)  # [1, 1, 100]

    return {"velocity": velocity, "physical_time": physical_time}


def load_sim_imu_ldm(path, n_samples=None):
    """Load synthetic parquet world-frame IMU in LDM channel order [accel, gyro].

    Returns numpy [N, 6].
    """
    path = path if path.endswith(".parquet") else path + ".parquet"
    df = pd.read_parquet(path)
    accel = df[["accel_world_x", "accel_world_y", "accel_world_z"]].values.astype(np.float64)
    gyro = df[["gyro_world_x", "gyro_world_y", "gyro_world_z"]].values.astype(np.float64)
    imu = np.concatenate([accel, gyro], axis=1)  # [N, 6] LDM order
    if n_samples is not None:
        imu = imu[:n_samples]
    return imu


@torch.no_grad()
def generate_features_ldm(
    features,
    ts,
    gt_pos,
    model,
    sampler,
    imu_mean,
    imu_std,
    vel_mean,
    vel_std,
    device,
    sim_features_ldm,
    ddim_steps=50,
    ddim_eta=0.0,
    use_ema=True,
    strength=1.0,
):
    """Generate IMU via sim-conditioned LDM in non-overlapping 10s windows.

    Args:
        features: numpy [N, 6] in RoNIN channel order [gyro, accel]
        ts: numpy [N] timestamps aligned with features/gt_pos
        gt_pos: numpy [N, 2|3] ground-truth position
        sim_features_ldm: numpy [N, 6] sim IMU in LDM order [accel, gyro]
        strength: 0 = VAE recon of input IMU; (0,1) = encode + partial
            DDIM denoising (img2img); >=1 = full generation from noise.
    Returns:
        generated: numpy [N, 6] in RoNIN channel order
    """
    if not (0.0 <= strength):
        raise ValueError(f"strength must be >= 0, got {strength}")
    if sim_features_ldm is None:
        raise ValueError("sim_features_ldm is required for sim-conditioned LDM")

    N = features.shape[0]
    if sim_features_ldm.shape[0] < N:
        raise ValueError(
            f"sim IMU length {sim_features_ldm.shape[0]} < features length {N}"
        )
    n_windows = N // LDM_WINDOW
    gen = features.copy()

    if n_windows == 0:
        print(f"  [warn] Sequence too short for a full 10s window ({N} samples)")
        return gen

    mean_d = imu_mean.to(device)
    std_d = imu_std.to(device)
    vel_mean_d = vel_mean.to(device)
    vel_std_d = vel_std.to(device)
    z_ch = int(getattr(model.first_stage_model, "embed_dim", 8))
    shape = (z_ch, LATENT_LENGTH)
    print(f"  LDM strength={strength} "
          f"({'VAE recon' if strength == 0.0 else 'full gen' if strength >= 1.0 else 'img2img'})")

    ctx = model.ema_scope("eval") if (use_ema and model.use_ema) else nullcontext()
    with ctx:
        for i in range(n_windows):
            s, e = i * LDM_WINDOW, (i + 1) * LDM_WINDOW
            # Align conditioning length with available ts/gt_pos
            n_cond = min(e, len(ts), len(gt_pos), sim_features_ldm.shape[0])
            if n_cond - s < LDM_WINDOW:
                print(f"  [warn] Window {i} truncated for conditioning; keeping original IMU")
                continue

            cond = _window_conditioning(ts, gt_pos, s, e, vel_mean_d, vel_std_d, device)

            # Encode sim IMU window as additional conditioning
            sim_win = torch.tensor(sim_features_ldm[s:e].T, dtype=torch.float32).unsqueeze(0).to(device)
            sim_norm = (sim_win - mean_d) / std_d
            sim_latent = model.scale_factor * model.encode_first_stage(sim_norm).mode()
            cond["sim_latent"] = sim_latent

            if strength < 1.0:
                # Encode input window (RoNIN [gyro,accel] -> LDM [accel,gyro])
                imu_win = torch.tensor(features[s:e], dtype=torch.float32)
                imu_ldm = ronin_to_ldm_channels(imu_win).T.unsqueeze(0).to(device)  # [1,6,2000]
                imu_norm = (imu_ldm - mean_d) / std_d
                z0 = model.scale_factor * model.encode_first_stage(imu_norm).mode()

                if strength == 0.0:
                    samples = z0
                else:
                    samples, _ = sampler.sample_img2img(
                        S=ddim_steps,
                        x0=z0,
                        conditioning=cond,
                        strength=strength,
                        eta=ddim_eta,
                        verbose=False,
                    )
            else:
                samples, _ = sampler.sample(
                    S=ddim_steps,
                    batch_size=1,
                    shape=shape,
                    conditioning=cond,
                    eta=ddim_eta,
                    verbose=False,
                )

            imu_std_out = model.decode_first_stage(samples / model.scale_factor)
            imu_phys = imu_std_out * std_d + mean_d  # [1, 6, 2000] LDM channel order
            out_ldm = imu_phys.squeeze(0).T.cpu()  # [2000, 6]
            out_ronin = ldm_to_ronin_channels(out_ldm)
            gen[s:e] = out_ronin.numpy()

    tail = N - n_windows * LDM_WINDOW
    if tail > 0:
        print(f"  [info] {tail} trailing samples kept unchanged (< 10s window)")
    return gen


@torch.no_grad()
def run_test(network, loader, device):
    """Run inference on a DataLoader, return (targets, preds) numpy arrays."""
    network.eval()
    all_t, all_p = [], []
    for feat, targ, _, _ in loader:
        pred = network(feat.to(device)).cpu().numpy()
        all_t.append(targ.numpy())
        all_p.append(pred)
    if not all_t:
        raise ValueError(
            "RoNIN DataLoader produced 0 windows. The sequence is shorter than "
            f"window_size after loading/trimming (need at least one full window)."
        )
    return np.concatenate(all_t), np.concatenate(all_p)


def recon_traj_2d(dataset, preds, seq_id=0):
    """Reconstruct 2D trajectory from predicted velocities (matches ronin_resnet)."""
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int64)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-6], ts[ind], [ts[-1] + 1e-6]])
    return interp1d(ts_ext, pos, axis=0)(ts)


def recon_traj_3d(dataset, preds, seq_id=0):
    """Reconstruct 3D trajectory from predicted displacements."""
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int64)
    if len(ind) < 2:
        return np.zeros((len(ts), 3))
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    window_dur = dataset.window_size / SAMPLE_RATE
    vel = preds / window_dur
    pos = np.zeros([preds.shape[0] + 2, 3])
    pos[0] = dataset.gt_pos[seq_id][0]
    pos[1:-1] = np.cumsum(vel * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-6], ts[ind], [ts[-1] + 1e-6]])
    pos_interp = np.zeros((len(ts), 3))
    for d in range(3):
        pos_interp[:, d] = interp1d(ts_ext, pos[:, d], kind="linear",
                                     fill_value="extrapolate")(ts)
    return pos_interp


def compute_ate(est, gt):
    return float(np.sqrt(np.mean((est - gt) ** 2)))


def compute_rte(est, gt, delta):
    if delta <= 0 or delta >= est.shape[0]:
        return 0.0
    err = est[delta:] + gt[:-delta] - est[:-delta] - gt[delta:]
    return float(np.sqrt(np.mean(err ** 2)))


def compute_ate_rte(est, gt, pred_per_min=12000):
    ate = compute_ate(est, gt)
    if est.shape[0] < pred_per_min:
        ratio = pred_per_min / est.shape[0]
        rte = compute_rte(est, gt, est.shape[0] - 1) * ratio
    else:
        rte = compute_rte(est, gt, pred_per_min)
    return ate, rte


def run_ronin_pipeline(network, dataset, device, use_3d=False):
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    targets, preds = run_test(network, loader, device)

    if use_3d:
        pos_pred = recon_traj_3d(dataset, preds)
        pos_gt = dataset.gt_pos[0]
    else:
        pos_pred = recon_traj_2d(dataset, preds)[:, :2]
        pos_gt = dataset.gt_pos[0][:, :2]

    ate, rte = compute_ate_rte(pos_pred, pos_gt, SAMPLE_RATE * 60)
    return {
        "preds": preds, "targets": targets,
        "pos_pred": pos_pred, "pos_gt": pos_gt,
        "ate": ate, "rte": rte,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectories(res_orig, res_recon, outdir, use_3d=False):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    ax = axes[0]
    ax.plot(res_orig["pos_gt"][:, 0], res_orig["pos_gt"][:, 1], "k-", lw=1.2, label="GT")
    ax.plot(res_orig["pos_pred"][:, 0], res_orig["pos_pred"][:, 1], "b-", lw=1.0, alpha=0.85, label="RoNIN")
    ax.set_title(f"Original IMU\nATE={res_orig['ate']:.3f}  RTE={res_orig['rte']:.3f}")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(res_recon["pos_gt"][:, 0], res_recon["pos_gt"][:, 1], "k-", lw=1.2, label="GT")
    ax.plot(res_recon["pos_pred"][:, 0], res_recon["pos_pred"][:, 1], "r-", lw=1.0, alpha=0.85, label="RoNIN (LDM gen)")
    ax.set_title(f"LDM-Generated IMU\nATE={res_recon['ate']:.3f}  RTE={res_recon['rte']:.3f}")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(res_orig["pos_gt"][:, 0], res_orig["pos_gt"][:, 1], "k-", lw=1.2, label="GT")
    ax.plot(res_orig["pos_pred"][:, 0], res_orig["pos_pred"][:, 1], "b-", lw=0.9, alpha=0.7, label="Original")
    ax.plot(res_recon["pos_pred"][:, 0], res_recon["pos_pred"][:, 1], "r-", lw=0.9, alpha=0.7, label="LDM gen")
    ax.set_title("Overlay")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

    fig.suptitle("RoNIN trajectory: Original vs sim-cond LDM-generated IMU", fontsize=13)
    fig.tight_layout()
    path = osp.join(outdir, "trajectory_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot to {path}")


def plot_position_error(res_orig, res_recon, outdir):
    err_orig = np.linalg.norm(res_orig["pos_pred"] - res_orig["pos_gt"], axis=1)
    err_recon = np.linalg.norm(res_recon["pos_pred"] - res_recon["pos_gt"], axis=1)
    t_o = np.arange(len(err_orig)) / SAMPLE_RATE
    t_r = np.arange(len(err_recon)) / SAMPLE_RATE

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_o, err_orig, "b-", lw=0.8, alpha=0.85, label=f"Original (ATE={res_orig['ate']:.3f})")
    ax.plot(t_r, err_recon, "r-", lw=0.8, alpha=0.85, label=f"LDM gen (ATE={res_recon['ate']:.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position error (m)")
    ax.set_title("Cumulative position error vs ground truth")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = osp.join(outdir, "position_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved position error plot to {path}")


def plot_imu_windows(feat_orig, feat_gen, outdir, n_plot=4):
    n_windows = feat_orig.shape[0] // LDM_WINDOW
    n_plot = min(n_plot, n_windows)
    if n_plot == 0:
        return

    overlay_dir = osp.join(outdir, "imu_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    ch_names = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
    t = np.arange(LDM_WINDOW) / SAMPLE_RATE

    for wi in range(n_plot):
        s, e = wi * LDM_WINDOW, (wi + 1) * LDM_WINDOW
        orig, gen = feat_orig[s:e], feat_gen[s:e]

        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        for c, ax in enumerate(axes):
            ax.plot(t, orig[:, c], lw=0.7, alpha=0.9, label="original")
            ax.plot(t, gen[:, c], lw=0.7, alpha=0.9, label="LDM gen")
            rmse = np.sqrt(np.mean((orig[:, c] - gen[:, c]) ** 2))
            ax.set_ylabel(ch_names[c], fontsize=9)
            ax.set_title(f"{ch_names[c]}  (RMSE={rmse:.4e})", fontsize=9, loc="left")
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"IMU window {wi} — original vs LDM generation", fontsize=12)
        fig.savefig(osp.join(overlay_dir, f"window_{wi:03d}.png"), dpi=140)
        plt.close(fig)

    print(f"Saved {n_plot} IMU overlay plots to {overlay_dir}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def detect_dataset_type(path, override=None):
    """Return 'sim_parquet' or 'hybrid' from path extension or override."""
    if override is not None:
        return override
    ext = osp.splitext(path)[1].lower()
    if ext == ".parquet":
        return "sim_parquet"
    if ext in (".hdf5", ".h5"):
        return "hybrid"
    raise ValueError(
        f"Cannot auto-detect dataset type for {path}. "
        f"Pass --dataset / --imu_dataset / --cond_dataset."
    )


def split_input_path(path):
    """Return (root_dir, data_name_without_ext, resolved_path_with_ext)."""
    path = path.rstrip("/")
    root_dir = osp.split(path)[0]
    data_name = osp.split(path)[1]
    resolved = path
    for suffix in (".parquet", ".hdf5", ".h5"):
        if data_name.endswith(suffix):
            data_name = data_name[:-len(suffix)]
            break
    else:
        # No extension on the provided path; leave data_name as-is (loaders append).
        pass
    return root_dir, data_name, resolved


def load_strided_dataset(path, dataset_type, step_size, window_size):
    """Load a single recording into a _StridedDataset."""
    root_dir, data_name, _ = split_input_path(path)
    seq_type = _ParquetSequence if dataset_type == "sim_parquet" else _HDF5Sequence
    return _StridedDataset(
        seq_type, root_dir, [data_name],
        step_size=step_size, window_size=window_size,
    )


def align_imu_and_cond(imu_ds, cond_ds, trim=False):
    """Take IMU features from imu_ds and traj conditioning from cond_ds.

    Requires equal sample counts unless trim=True (then truncate both to min).
    Returns (features, ts_cond, gt_pos_cond, ts_imu, n_samples) and mutates
    imu_ds so RoNIN GT / timestamps come from the conditioning trajectory.
    """
    features = np.asarray(imu_ds.features[0])
    ts_imu = np.asarray(imu_ds.ts_full[0], dtype=np.float64).copy()
    ts_cond = np.asarray(cond_ds.ts_full[0], dtype=np.float64)
    gt_cond = np.asarray(cond_ds.gt_pos_full[0], dtype=np.float64)

    n_imu = features.shape[0]
    n_cond = min(len(ts_cond), len(gt_cond))
    if n_imu != n_cond:
        if not trim:
            raise ValueError(
                f"IMU and conditioning lengths must match: "
                f"IMU features={n_imu} ({n_imu / SAMPLE_RATE:.2f}s), "
                f"cond traj={n_cond} ({n_cond / SAMPLE_RATE:.2f}s). "
                f"Pass --trim_to_match to truncate to the shorter length, "
                f"or trim/resample sources beforehand."
            )
        N = min(n_imu, n_cond)
        print(f"  [trim] Truncating to common length {N} samples "
              f"({N / SAMPLE_RATE:.2f}s); IMU was {n_imu}, cond was {n_cond}")
    else:
        N = n_imu

    features = features[:N]
    ts_imu = ts_imu[:N]
    ts = ts_cond[:N]
    gt_pos = gt_cond[:N]

    # Point RoNIN evaluation GT at the conditioning trajectory
    imu_ds.features[0] = features
    imu_ds.ts_full[0] = ts
    imu_ds.gt_pos_full[0] = gt_pos

    n_ronin_ts = min(len(imu_ds.ts[0]), N)
    n_ronin_pos = min(len(imu_ds.gt_pos[0]), N)
    imu_ds.ts[0] = ts[:n_ronin_ts]
    pos_cols = imu_ds.gt_pos[0].shape[1]
    if gt_pos.shape[1] >= pos_cols:
        imu_ds.gt_pos[0] = gt_pos[:n_ronin_pos, :pos_cols]
    else:
        pad = np.zeros((n_ronin_pos, pos_cols), dtype=gt_pos.dtype)
        pad[:, :gt_pos.shape[1]] = gt_pos[:n_ronin_pos]
        imu_ds.gt_pos[0] = pad

    if len(imu_ds.orientations) > 0:
        imu_ds.orientations[0] = imu_ds.orientations[0][:n_ronin_ts]
    if len(imu_ds.targets) > 0:
        # Keep targets whose RoNIN window still fits in the truncated features
        max_start = max(0, N - imu_ds.window_size)
        imu_ds.targets[0] = imu_ds.targets[0][: max_start + 1]

    # Drop RoNIN windows that would read past the truncated feature length
    imu_ds.index_map = [
        [seq_id, frame_id]
        for seq_id, frame_id in imu_ds.index_map
        if seq_id == 0
        and frame_id + imu_ds.window_size <= N
        and frame_id < len(imu_ds.targets[0])
    ]

    return features, ts, gt_pos, ts_imu, N


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RoNIN on original vs sim-conditioned LDM-generated IMU"
    )
    parser.add_argument("--pair_dir", type=str, default=None,
                        help="Pair folder with real.hdf5 + synthetic.parquet "
                             "(sets imu/cond to real.hdf5 and sim to synthetic.parquet)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to .parquet or .hdf5 (used for both IMU and "
                             "trajectory conditioning unless --imu_input / --cond_input set)")
    parser.add_argument("--imu_input", type=str, default=None,
                        help="IMU source (.parquet or .hdf5). Defaults to --input")
    parser.add_argument("--sim_input", type=str, default=None,
                        help="Synthetic IMU parquet for sim_latent conditioning "
                             "(required unless --pair_dir is set)")
    parser.add_argument("--cond_input", type=str, default=None,
                        help="Conditioning trajectory source (.parquet or .hdf5). "
                             "Defaults to --input / --imu_input. Length must match "
                             "--imu_input unless --trim_to_match is set")
    parser.add_argument("--trim_to_match", action="store_true",
                        help="If IMU and cond lengths differ, truncate both to the "
                             "shorter length (prefix) instead of erroring")
    parser.add_argument("--ldm_config", "--config", type=str,
                        default="configs/imu/ldm_1d_sim_cond.yaml",
                        help="LDM model config (must match the checkpoint architecture)")
    parser.add_argument("--ldm_ckpt", type=str, required=True)
    parser.add_argument("--first_stage_ckpt", type=str, required=True,
                        help="VAE checkpoint used as LDM first stage")
    parser.add_argument("--ronin_ckpt", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True, help="Path to stats.pt")
    parser.add_argument("--outdir", type=str, default="outputs/ronin_ldm_sim_cond_eval")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset type for --input (sim_parquet / hybrid)")
    parser.add_argument("--imu_dataset", type=str, default=None,
                        help="Override dataset type for --imu_input")
    parser.add_argument("--cond_dataset", type=str, default=None,
                        help="Override dataset type for --cond_input")
    parser.add_argument("--use_3d", action="store_true",
                        help="Use 3D RoNIN model (ronin_resnet_3d) instead of 2D")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Sim-to-real strength: 0=VAE recon, (0,1)=partial denoising from "
             "encoded input IMU, >=1=full generation from noise (default)",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help="Override model.scale_factor (loaded from ckpt buffer when present)",
    )
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA weights for sampling")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--n_imu_plot", type=int, default=4,
                        help="Number of IMU window overlay plots")
    parser.add_argument(
        "--hdf5_out",
        type=str,
        nargs="?",
        const=None,
        default=None,
        help="Output path for HDF5 with generated world-frame IMU "
             "(default: <outdir>/<imu_stem>_ldm_gen.hdf5). "
             "Written by default when an HDF5 template is available "
             "(IMU source, else cond source). Pass empty string to disable.",
    )
    parser.add_argument(
        "--parquet_out",
        type=str,
        nargs="?",
        const=None,
        default=None,
        help="Output path for parquet with generated world-frame IMU "
             "(default: <outdir>/<imu_stem>_ldm_gen.parquet). "
             "Only written when IMU source is parquet. Pass empty string to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_trajectories", action="store_true",
        help="Save pos_pred/pos_gt arrays for original and generated as "
             "<outdir>/trajectories.npz (for downstream aggregation)",
    )
    args = parser.parse_args()

    # Resolve pair_dir convenience -> imu / sim / cond paths
    if args.pair_dir is not None:
        pair_dir = args.pair_dir.rstrip("/")
        real_hdf5 = osp.join(pair_dir, "real.hdf5")
        sim_parquet = osp.join(pair_dir, "synthetic.parquet")
        if not osp.isfile(real_hdf5):
            raise ValueError(f"Missing real.hdf5 in {pair_dir}")
        if not osp.isfile(sim_parquet):
            raise ValueError(f"Missing synthetic.parquet in {pair_dir}")
        if args.imu_input is None and args.input is None:
            args.imu_input = real_hdf5
            args.input = real_hdf5
        if args.cond_input is None:
            args.cond_input = real_hdf5
        if args.sim_input is None:
            args.sim_input = sim_parquet
        if args.imu_dataset is None and args.dataset is None:
            args.imu_dataset = "hybrid"
        if args.cond_dataset is None and args.dataset is None:
            args.cond_dataset = "hybrid"

    imu_path = args.imu_input or args.input
    cond_path = args.cond_input or args.input
    sim_path = args.sim_input
    if imu_path is None or cond_path is None:
        raise ValueError(
            "Provide --pair_dir, or --input, or both --imu_input and --cond_input "
            "(each defaults to --input when omitted)"
        )
    if sim_path is None:
        raise ValueError(
            "Provide --sim_input (synthetic parquet) or --pair_dir for sim-latent conditioning"
        )

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Legacy --dataset applies to --input when used as the shared source
    imu_override = args.imu_dataset
    cond_override = args.cond_dataset
    if args.dataset is not None:
        if args.imu_input is None and imu_override is None:
            imu_override = args.dataset
        if args.cond_input is None and cond_override is None:
            cond_override = args.dataset

    imu_type = detect_dataset_type(imu_path, imu_override)
    cond_type = detect_dataset_type(cond_path, cond_override)
    mixed = osp.abspath(imu_path) != osp.abspath(cond_path)

    print(f"IMU source:  {imu_path} ({imu_type})")
    print(f"Cond source: {cond_path} ({cond_type})"
          + ("  [mixed]" if mixed else "  [same as IMU]"))
    print(f"Sim source:  {sim_path}")
    print(f"Device:  {device}")
    print(f"Strength:{args.strength}")

    # ---- Load models ----
    print("\nLoading LDM ...")
    ldm = load_ldm(
        args.ldm_config,
        args.ldm_ckpt,
        device,
        first_stage_ckpt=args.first_stage_ckpt,
        scale_factor=args.scale_factor,
    )
    sampler = DDIMSampler1D(ldm)
    imu_mean, imu_std, vel_mean, vel_std = load_vae_stats(args.stats)
    print(f"  scale_factor={ldm.scale_factor}, use_ema={ldm.use_ema and not args.no_ema}")

    print("Loading RoNIN ...")
    ckpt = torch.load(args.ronin_ckpt, map_location="cpu")
    if args.use_3d:
        ronin_net = _build_ronin_3d(args.arch, args.window_size)
    else:
        ronin_net = _build_ronin_2d(args.arch, args.window_size)
    ronin_net.load_state_dict(ckpt["model_state_dict"])
    ronin_net.eval().to(device)

    # ---- Load data ----
    print("Loading data ...")
    imu_ds = load_strided_dataset(
        imu_path, imu_type, args.step_size, args.window_size
    )
    if mixed:
        cond_ds = load_strided_dataset(
            cond_path, cond_type, args.step_size, args.window_size
        )
        features_orig, ts, gt_pos, ts_imu, N = align_imu_and_cond(
            imu_ds, cond_ds, trim=args.trim_to_match
        )
        dataset_orig = imu_ds
    else:
        dataset_orig = imu_ds
        features_orig = dataset_orig.features[0]
        ts = dataset_orig.ts_full[0]
        gt_pos = dataset_orig.gt_pos_full[0]
        N = min(features_orig.shape[0], len(ts), len(gt_pos))
        if N < features_orig.shape[0]:
            print(f"  [info] Truncating features {features_orig.shape[0]} -> {N} "
                  f"to match ts/gt_pos")
            features_orig = features_orig[:N]
            dataset_orig.features[0] = features_orig
            ts = ts[:N]
            gt_pos = gt_pos[:N]
        ts_imu = ts

    sim_features_ldm = load_sim_imu_ldm(sim_path, n_samples=None)
    if sim_features_ldm.shape[0] < N:
        if not args.trim_to_match:
            raise ValueError(
                f"Sim IMU length {sim_features_ldm.shape[0]} < IMU length {N}. "
                f"Pass --trim_to_match to truncate."
            )
        N = sim_features_ldm.shape[0]
        print(f"  [trim] Truncating to sim length {N}")
        features_orig = features_orig[:N]
        ts = ts[:N]
        gt_pos = gt_pos[:N]
        ts_imu = ts_imu[:N]
        dataset_orig.features[0] = features_orig
    elif sim_features_ldm.shape[0] > N:
        sim_features_ldm = sim_features_ldm[:N]

    n_ldm_windows = N // LDM_WINDOW
    print(f"  {N} samples ({N / SAMPLE_RATE:.1f}s), {n_ldm_windows} full 10s LDM windows, "
          f"{len(dataset_orig)} RoNIN windows")

    if len(dataset_orig) == 0 or N < args.window_size:
        raise ValueError(
            f"Sequence too short for RoNIN evaluation after loading/trimming: "
            f"N={N} samples ({N / SAMPLE_RATE:.2f}s), window_size={args.window_size}, "
            f"RoNIN windows={len(dataset_orig)}. "
            f"IMU features and conditioning lengths were mismatched "
            f"(see [trim] message above) or the IMU source is nearly empty. "
            f"Check that --imu_input is a full-length recording aligned with --cond_input."
        )

    # ---- LDM generation ----
    print("\nRunning sim-conditioned LDM generation ...")
    features_gen = generate_features_ldm(
        features_orig,
        ts,
        gt_pos,
        ldm,
        sampler,
        imu_mean,
        imu_std,
        vel_mean,
        vel_std,
        device,
        sim_features_ldm=sim_features_ldm,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        use_ema=not args.no_ema,
        strength=args.strength,
    )

    dataset_gen = copy.deepcopy(dataset_orig)
    dataset_gen.features[0] = features_gen

    # ---- RoNIN on both ----
    print("Running RoNIN on original IMU ...")
    res_orig = run_ronin_pipeline(ronin_net, dataset_orig, device, use_3d=args.use_3d)
    print(f"  Original   — ATE: {res_orig['ate']:.4f}, RTE: {res_orig['rte']:.4f}")

    print("Running RoNIN on LDM-generated IMU ...")
    res_gen = run_ronin_pipeline(ronin_net, dataset_gen, device, use_3d=args.use_3d)
    print(f"  LDM gen    — ATE: {res_gen['ate']:.4f}, RTE: {res_gen['rte']:.4f}")

    # ---- Metrics ----
    metrics = {
        "imu_input": osp.abspath(imu_path),
        "cond_input": osp.abspath(cond_path),
        "sim_input": osp.abspath(sim_path),
        "pair_dir": osp.abspath(args.pair_dir) if args.pair_dir else None,
        "mixed": mixed,
        "trim_to_match": bool(args.trim_to_match),
        "imu_dataset_type": imu_type,
        "cond_dataset_type": cond_type,
        "ldm_ckpt": osp.abspath(args.ldm_ckpt),
        "first_stage_ckpt": osp.abspath(args.first_stage_ckpt),
        "ronin_ckpt": osp.abspath(args.ronin_ckpt),
        "n_samples": int(N),
        "n_ldm_windows": n_ldm_windows,
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
        "strength": args.strength,
        "scale_factor": float(ldm.scale_factor),
        "original": {"ate": res_orig["ate"], "rte": res_orig["rte"]},
        "ldm_gen": {"ate": res_gen["ate"], "rte": res_gen["rte"]},
        "delta": {
            "ate": res_gen["ate"] - res_orig["ate"],
            "rte": res_gen["rte"] - res_orig["rte"],
        },
    }
    metrics_path = osp.join(args.outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # ---- Save trajectory arrays (opt-in) ----
    if args.save_trajectories:
        traj_path = osp.join(args.outdir, "trajectories.npz")
        np.savez_compressed(
            traj_path,
            pos_gt=res_orig["pos_gt"],
            pos_pred_orig=res_orig["pos_pred"],
            pos_pred_gen=res_gen["pos_pred"],
        )
        print(f"Trajectories saved to {traj_path}")

    # ---- Plots ----
    plot_trajectories(res_orig, res_gen, args.outdir, use_3d=args.use_3d)
    plot_position_error(res_orig, res_gen, args.outdir)
    plot_imu_windows(features_orig, features_gen, args.outdir, n_plot=args.n_imu_plot)

    # ---- Export generated IMU (HDF5 by default; parquet when IMU is parquet) ----
    stem = osp.splitext(osp.basename(imu_path.rstrip("/")))[0]
    wrote_export = False

    if args.hdf5_out != "":
        if imu_type != "sim_parquet":
            src_hdf5 = (
                imu_path if imu_path.endswith((".hdf5", ".h5")) else imu_path + ".hdf5"
            )
        elif cond_type != "sim_parquet":
            src_hdf5 = (
                cond_path if cond_path.endswith((".hdf5", ".h5")) else cond_path + ".hdf5"
            )
        else:
            src_hdf5 = None
        if src_hdf5 is not None and osp.isfile(src_hdf5):
            if args.hdf5_out is None:
                hdf5_out = osp.join(args.outdir, f"{stem}_ldm_gen.hdf5")
            else:
                hdf5_out = args.hdf5_out
            print("\nWriting generated IMU HDF5 (world-frame synced/acce+gyro) ...")
            write_generated_hdf5(src_hdf5, hdf5_out, features_gen)
            metrics["hdf5_out"] = osp.abspath(hdf5_out)
            wrote_export = True
        elif src_hdf5 is not None:
            print(f"\n[warn] Skipping HDF5 export; template not found: {src_hdf5}")

    if imu_type == "sim_parquet" and args.parquet_out != "":
        if args.parquet_out is None:
            parquet_out = osp.join(args.outdir, f"{stem}_ldm_gen.parquet")
        else:
            parquet_out = args.parquet_out
        print("\nWriting generated IMU parquet (world-frame columns) ...")
        src_pq = imu_path if imu_path.endswith(".parquet") else imu_path + ".parquet"
        write_generated_parquet(src_pq, parquet_out, features_gen, ts_imu[: features_gen.shape[0]])
        metrics["parquet_out"] = osp.abspath(parquet_out)
        wrote_export = True

    if wrote_export:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
