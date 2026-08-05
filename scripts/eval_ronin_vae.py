"""Evaluate RoNIN trajectory on original vs VAE-reconstructed IMU.

Loads a single parquet or hdf5 recording, runs RoNIN inference on the
original IMU, then reconstructs the IMU through a trained VAE (in
non-overlapping 10s windows) and runs RoNIN again. Produces side-by-side
trajectory plots and ATE/RTE metrics.

Example usage (parquet):
    python scripts/eval_ronin_vae.py \
        --input data/smplx_data_gen_0000.parquet \
        --vae_ckpt logs/vae_1d/checkpoints/best-000.ckpt \
        --ronin_ckpt path/to/ronin/checkpoint_latest.pt \
        --stats data/dataset_processed_overlapped/stats.pt \
        --outdir outputs/ronin_vae_eval

Example usage (hdf5):
    python scripts/eval_ronin_vae.py \
        --input data/dataset/john_chest_ios_corrected.hdf5 \
        --dataset hybrid \
        --vae_ckpt logs/vae_1d/checkpoints/best-000.ckpt \
        --ronin_ckpt path/to/ronin/checkpoint_latest.pt \
        --stats data/dataset_processed_overlapped/stats.pt \
        --outdir outputs/ronin_vae_eval
"""

import argparse
import copy
import json
import os
import sys
from os import path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quaternion
import torch
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))

from ldm.util import instantiate_from_config

RONIN_ROOT = osp.normpath(osp.join(osp.dirname(__file__), "..", "..", "juxta-ronin"))

SAMPLE_RATE = 200
VAE_WINDOW = 2000  # 10s at 200 Hz


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
        n_full = len(glob_v)
        n = max(1, n_full - self.window_size)
        self.ts = ts[:n]
        self.targets = glob_v[:n, :2]
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
        return "parquet (eval_ronin_vae)"


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
        self.targets = glob_v[:, :2]
        self.orientations = ori
        self.gt_pos = tango_pos

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        n = min(len(self.ts), len(self.orientations), len(self.gt_pos))
        return np.concatenate([self.ts[:n, None], self.orientations[:n], self.gt_pos[:n]], axis=1)

    def get_meta(self):
        return "hdf5 (eval_ronin_vae)"


# Minimal reimplementation of StridedSequenceDataset
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

def load_vae(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load VAE weights from {ckpt_path} into architecture from "
            f"{config_path}. Pass the matching config via --vae_config / --config "
            f"(e.g. configs/imu/vae_1d_downsample221_kl0.yaml for downsample221_kl0).\n"
            f"Original error: {e}"
        ) from e
    return model.to(device).eval()


def load_vae_stats(stats_path):
    stats = torch.load(stats_path, weights_only=True)
    return stats["imu_mean"].view(6, 1), stats["imu_std"].view(6, 1)


def ronin_to_vae_channels(feat):
    """[gyro(3), accel(3)] -> [accel(3), gyro(3)]"""
    return feat[:, [3, 4, 5, 0, 1, 2]]


def vae_to_ronin_channels(feat):
    """[accel(3), gyro(3)] -> [gyro(3), accel(3)]"""
    return feat[:, [3, 4, 5, 0, 1, 2]]


@torch.no_grad()
def reconstruct_features_vae(features, vae, imu_mean, imu_std, device):
    """Run VAE on full feature array in non-overlapping 10s windows.

    Args:
        features: numpy [N, 6] in RoNIN channel order [gyro, accel]
    Returns:
        reconstructed: numpy [N, 6] in RoNIN channel order
    """
    N = features.shape[0]
    n_windows = N // VAE_WINDOW
    recon = features.copy()

    if n_windows == 0:
        print(f"  [warn] Sequence too short for a full 10s window ({N} samples)")
        return recon

    mean_d = imu_mean.to(device)
    std_d = imu_std.to(device)

    for i in range(n_windows):
        s, e = i * VAE_WINDOW, (i + 1) * VAE_WINDOW
        window = torch.tensor(features[s:e], dtype=torch.float32)  # [2000, 6]
        window_vae = ronin_to_vae_channels(window)  # swap to VAE order
        x = window_vae.T.unsqueeze(0).to(device)    # [1, 6, 2000]

        x_std = (x - mean_d) / std_d
        x_recon, _ = vae(x_std, sample_posterior=False)
        x_recon = x_recon * std_d + mean_d

        out_vae = x_recon.squeeze(0).T.cpu()  # [2000, 6] VAE order
        out_ronin = vae_to_ronin_channels(out_vae)  # swap back
        recon[s:e] = out_ronin.numpy()

    tail = N - n_windows * VAE_WINDOW
    if tail > 0:
        print(f"  [info] {tail} trailing samples kept unchanged (< 10s window)")
    return recon


@torch.no_grad()
def run_test(network, loader, device):
    """Run inference on a DataLoader, return (targets, preds) numpy arrays."""
    network.eval()
    all_t, all_p = [], []
    for feat, targ, _, _ in loader:
        pred = network(feat.to(device)).cpu().numpy()
        all_t.append(targ.numpy())
        all_p.append(pred)
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
    ax.plot(res_recon["pos_pred"][:, 0], res_recon["pos_pred"][:, 1], "r-", lw=1.0, alpha=0.85, label="RoNIN (VAE recon)")
    ax.set_title(f"VAE-Reconstructed IMU\nATE={res_recon['ate']:.3f}  RTE={res_recon['rte']:.3f}")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(res_orig["pos_gt"][:, 0], res_orig["pos_gt"][:, 1], "k-", lw=1.2, label="GT")
    ax.plot(res_orig["pos_pred"][:, 0], res_orig["pos_pred"][:, 1], "b-", lw=0.9, alpha=0.7, label="Original")
    ax.plot(res_recon["pos_pred"][:, 0], res_recon["pos_pred"][:, 1], "r-", lw=0.9, alpha=0.7, label="VAE recon")
    ax.set_title("Overlay")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

    fig.suptitle("RoNIN trajectory: Original vs VAE-reconstructed IMU", fontsize=13)
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
    ax.plot(t_r, err_recon, "r-", lw=0.8, alpha=0.85, label=f"VAE recon (ATE={res_recon['ate']:.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position error (m)")
    ax.set_title("Cumulative position error vs ground truth")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = osp.join(outdir, "position_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved position error plot to {path}")


def plot_imu_windows(feat_orig, feat_recon, outdir, n_plot=4):
    n_windows = feat_orig.shape[0] // VAE_WINDOW
    n_plot = min(n_plot, n_windows)
    if n_plot == 0:
        return

    overlay_dir = osp.join(outdir, "imu_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    ch_names = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
    t = np.arange(VAE_WINDOW) / SAMPLE_RATE

    for wi in range(n_plot):
        s, e = wi * VAE_WINDOW, (wi + 1) * VAE_WINDOW
        orig, recon = feat_orig[s:e], feat_recon[s:e]

        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        for c, ax in enumerate(axes):
            ax.plot(t, orig[:, c], lw=0.7, alpha=0.9, label="original")
            ax.plot(t, recon[:, c], lw=0.7, alpha=0.9, label="VAE recon")
            rmse = np.sqrt(np.mean((orig[:, c] - recon[:, c]) ** 2))
            ax.set_ylabel(ch_names[c], fontsize=9)
            ax.set_title(f"{ch_names[c]}  (RMSE={rmse:.4e})", fontsize=9, loc="left")
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"IMU window {wi} — original vs VAE reconstruction", fontsize=12)
        fig.savefig(osp.join(overlay_dir, f"window_{wi:03d}.png"), dpi=140)
        plt.close(fig)

    print(f"Saved {n_plot} IMU overlay plots to {overlay_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RoNIN trajectory on original vs VAE-reconstructed IMU"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to .parquet or .hdf5 input file")
    parser.add_argument("--vae_config", "--config", type=str,
                        default="configs/imu/vae_1d.yaml",
                        help="VAE model config (must match the checkpoint architecture)")
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--ronin_ckpt", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True, help="Path to stats.pt")
    parser.add_argument("--outdir", type=str, default="outputs/ronin_vae_eval")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset type (sim_parquet / hybrid). Auto-detected from extension.")
    parser.add_argument("--use_3d", action="store_true",
                        help="Use 3D RoNIN model (ronin_resnet_3d) instead of 2D")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--n_imu_plot", type=int, default=4,
                        help="Number of IMU window overlay plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ext = osp.splitext(args.input)[1].lower()
    if args.dataset is not None:
        dataset_type = args.dataset
    elif ext == ".parquet":
        dataset_type = "sim_parquet"
    elif ext in (".hdf5", ".h5"):
        dataset_type = "hybrid"
    else:
        raise ValueError(f"Cannot auto-detect dataset type for {ext}. Use --dataset.")

    print(f"Input:   {args.input}")
    print(f"Dataset: {dataset_type}")
    print(f"Device:  {device}")

    # ---- Load models ----
    print("\nLoading VAE ...")
    vae = load_vae(args.vae_config, args.vae_ckpt, device)
    imu_mean, imu_std = load_vae_stats(args.stats)

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
    input_path = args.input.rstrip("/")
    root_dir = osp.split(input_path)[0]
    data_name = osp.split(input_path)[1]
    # Strip extension (seq loaders append it internally)
    for suffix in (".parquet", ".hdf5", ".h5"):
        if data_name.endswith(suffix):
            data_name = data_name[:-len(suffix)]
            break

    if dataset_type == "sim_parquet":
        seq_type = _ParquetSequence
    else:
        seq_type = _HDF5Sequence

    dataset_orig = _StridedDataset(
        seq_type, root_dir, [data_name],
        step_size=args.step_size, window_size=args.window_size,
    )
    features_orig = dataset_orig.features[0]
    N = features_orig.shape[0]
    n_vae_windows = N // VAE_WINDOW
    print(f"  {N} samples ({N / SAMPLE_RATE:.1f}s), {n_vae_windows} full 10s VAE windows, "
          f"{len(dataset_orig)} RoNIN windows")

    # ---- VAE reconstruction ----
    print("\nRunning VAE reconstruction ...")
    features_recon = reconstruct_features_vae(features_orig, vae, imu_mean, imu_std, device)

    dataset_recon = copy.deepcopy(dataset_orig)
    dataset_recon.features[0] = features_recon

    # ---- RoNIN on both ----
    print("Running RoNIN on original IMU ...")
    res_orig = run_ronin_pipeline(ronin_net, dataset_orig, device, use_3d=args.use_3d)
    print(f"  Original   — ATE: {res_orig['ate']:.4f}, RTE: {res_orig['rte']:.4f}")

    print("Running RoNIN on VAE-reconstructed IMU ...")
    res_recon = run_ronin_pipeline(ronin_net, dataset_recon, device, use_3d=args.use_3d)
    print(f"  VAE recon  — ATE: {res_recon['ate']:.4f}, RTE: {res_recon['rte']:.4f}")

    # ---- Metrics ----
    metrics = {
        "input": osp.abspath(args.input),
        "dataset_type": dataset_type,
        "vae_ckpt": osp.abspath(args.vae_ckpt),
        "ronin_ckpt": osp.abspath(args.ronin_ckpt),
        "n_samples": int(N),
        "n_vae_windows": n_vae_windows,
        "original": {"ate": res_orig["ate"], "rte": res_orig["rte"]},
        "vae_recon": {"ate": res_recon["ate"], "rte": res_recon["rte"]},
        "delta": {
            "ate": res_recon["ate"] - res_orig["ate"],
            "rte": res_recon["rte"] - res_orig["rte"],
        },
    }
    metrics_path = osp.join(args.outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # ---- Plots ----
    plot_trajectories(res_orig, res_recon, args.outdir, use_3d=args.use_3d)
    plot_position_error(res_orig, res_recon, args.outdir)
    plot_imu_windows(features_orig, features_recon, args.outdir, n_plot=args.n_imu_plot)

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
