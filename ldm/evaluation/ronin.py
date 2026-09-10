"""RoNIN windowing, inference and trajectory reconstruction.

Kept free of the RoNIN training package so the eval scripts do not inherit its
tensorboardX / yaspin dependencies; only the model definitions are imported,
lazily, by :mod:`ldm.evaluation.models`.
"""

import copy
import os.path as osp

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset

from ldm.evaluation.constants import SAMPLE_RATE, TRAJECTORY_EXTS
from ldm.evaluation.metrics import compute_ate_rte, compute_rte_at_delta
from ldm.evaluation.sequences import ArraySequence, HDF5Sequence, ParquetSequence


class StridedDataset(Dataset):
    """Strided windowing over one or more sequences, as RoNIN's loop expects."""

    def __init__(self, seq_type, root_dir, data_list, step_size=10, window_size=200,
                 **kwargs):
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


def detect_dataset_type(path, override=None):
    """``sim_parquet`` or ``hybrid`` from a path's extension, or an override."""
    if override is not None:
        return override
    ext = osp.splitext(path)[1].lower()
    if ext == ".parquet":
        return "sim_parquet"
    if ext in (".hdf5", ".h5"):
        return "hybrid"
    raise ValueError(f"Cannot auto-detect the sequence type for {path}")


def split_input_path(path):
    """Keep the real extension, especially .h5 versus .hdf5."""
    return osp.split(str(path).rstrip("/"))


def load_strided_dataset(path, dataset_type=None, step_size=10, window_size=200,
                         already_world=False):
    """Load one recording into a :class:`StridedDataset`."""
    root_dir, data_name = split_input_path(path)
    seq_type = ParquetSequence if detect_dataset_type(path, dataset_type) == "sim_parquet" \
        else HDF5Sequence
    return StridedDataset(
        seq_type, root_dir, [data_name], step_size=step_size, window_size=window_size,
        already_world=already_world,
    )


def dataset_from_arrays(time, position, imu, step_size=10, window_size=200):
    return StridedDataset(
        ArraySequence, "", [""], step_size=step_size, window_size=window_size,
        time=time, position=position, imu=imu,
    )


def with_features(dataset, features):
    """Copy of ``dataset`` whose first sequence uses ``features`` instead.

    Reconstructed and generated IMU are scored on exactly the same windows and
    ground truth as the original, so only the feature array changes.
    """
    replaced = copy.deepcopy(dataset)
    replaced.features[0] = features
    return replaced


def align_imu_and_cond(imu_ds, cond_ds, trim=False):
    """Take IMU from one dataset and trajectory conditioning from another.

    Requires equal sample counts unless ``trim`` truncates both to the shorter
    one. Mutates ``imu_ds`` so RoNIN's ground truth and timestamps come from the
    conditioning trajectory, and drops windows that would read past the end.

    Returns (features, ts_cond, gt_pos_cond, ts_imu, n_samples).
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
                f"IMU and conditioning lengths must match: IMU features={n_imu} "
                f"({n_imu / SAMPLE_RATE:.2f}s), cond traj={n_cond} "
                f"({n_cond / SAMPLE_RATE:.2f}s). Pass --trim_to_match to truncate "
                f"to the shorter length, or trim/resample the sources beforehand."
            )
        n = min(n_imu, n_cond)
        print(f"  [trim] Truncating to common length {n} samples "
              f"({n / SAMPLE_RATE:.2f}s); IMU was {n_imu}, cond was {n_cond}")
    else:
        n = n_imu

    features = features[:n]
    ts_imu = ts_imu[:n]
    ts = ts_cond[:n]
    gt_pos = gt_cond[:n]

    imu_ds.features[0] = features
    imu_ds.ts_full[0] = ts
    imu_ds.gt_pos_full[0] = gt_pos

    n_ronin_ts = min(len(imu_ds.ts[0]), n)
    n_ronin_pos = min(len(imu_ds.gt_pos[0]), n)
    imu_ds.ts[0] = ts[:n_ronin_ts]
    pos_cols = imu_ds.gt_pos[0].shape[1]
    if gt_pos.shape[1] >= pos_cols:
        imu_ds.gt_pos[0] = gt_pos[:n_ronin_pos, :pos_cols]
    else:
        padded = np.zeros((n_ronin_pos, pos_cols), dtype=gt_pos.dtype)
        padded[:, :gt_pos.shape[1]] = gt_pos[:n_ronin_pos]
        imu_ds.gt_pos[0] = padded

    if imu_ds.orientations:
        imu_ds.orientations[0] = imu_ds.orientations[0][:n_ronin_ts]
    if imu_ds.targets:
        # Mixed sources must use the conditioning trajectory's velocity too.
        w = imu_ds.window_size
        count = min(len(imu_ds.targets[0]), max(0, n - w))
        imu_ds.targets[0] = ((gt_pos[w:w + count, :2] - gt_pos[:count, :2]) /
                             (ts[w:w + count] - ts[:count])[:, None])

    imu_ds.index_map = [
        [seq_id, frame_id]
        for seq_id, frame_id in imu_ds.index_map
        if seq_id == 0
        and frame_id + imu_ds.window_size <= n
        and frame_id < len(imu_ds.targets[0])
    ]

    return features, ts, gt_pos, ts_imu, n


def trim_dataset_to(dataset, n):
    """Truncate a single-sequence dataset in place to ``n`` samples."""
    dataset.features[0] = dataset.features[0][:n]
    dataset.ts_full[0] = np.asarray(dataset.ts_full[0])[:n]
    dataset.gt_pos_full[0] = np.asarray(dataset.gt_pos_full[0])[:n]
    dataset.ts[0] = np.asarray(dataset.ts[0])[:n]
    dataset.gt_pos[0] = np.asarray(dataset.gt_pos[0])[:n]
    dataset.orientations[0] = np.asarray(dataset.orientations[0])[:n]
    # Targets are forward differences over window_size, requiring the endpoint.
    dataset.targets[0] = dataset.targets[0][:max(0, n - dataset.window_size)]
    dataset.index_map = [
        [seq_id, frame_id]
        for seq_id, frame_id in dataset.index_map
        if seq_id == 0
        and frame_id + dataset.window_size <= n
        and frame_id < len(dataset.targets[0])
    ]
    return dataset


@torch.no_grad()
def run_inference(network, loader, device):
    """Run RoNIN over a DataLoader, returning (targets, preds) numpy arrays."""
    network.eval()
    all_targets, all_preds = [], []
    for feat, targ, _, _ in loader:
        all_preds.append(network(feat.to(device)).cpu().numpy())
        all_targets.append(targ.numpy())
    if not all_targets:
        raise ValueError(
            "RoNIN DataLoader produced 0 windows; the sequence is shorter than "
            "window_size after loading/trimming"
        )
    return np.concatenate(all_targets), np.concatenate(all_preds)


def recon_traj_2d(dataset, preds, seq_id=0):
    """Integrate predicted 2D velocities into a trajectory (as ronin_resnet)."""
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int64)
    if len(ind) < 2:
        raise ValueError("RoNIN trajectory reconstruction requires at least two prediction windows")
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-6], ts[ind], [ts[-1] + 1e-6]])
    return interp1d(ts_ext, pos, axis=0)(ts)


def recon_traj_3d(dataset, preds, seq_id=0):
    """Integrate predicted 3D displacements into a trajectory."""
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int64)
    if len(ind) < 2:
        return np.zeros((len(ts), 3))
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    vel = preds / (dataset.window_size / SAMPLE_RATE)
    pos = np.zeros([preds.shape[0] + 2, 3])
    pos[0] = dataset.gt_pos[seq_id][0]
    pos[1:-1] = np.cumsum(vel * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-6], ts[ind], [ts[-1] + 1e-6]])
    interpolated = np.zeros((len(ts), 3))
    for d in range(3):
        interpolated[:, d] = interp1d(
            ts_ext, pos[:, d], kind="linear", fill_value="extrapolate"
        )(ts)
    return interpolated


def run_ronin_pipeline(network, dataset, device, use_3d=False, rte_delta=None,
                       batch_size=1024):
    """Predict, integrate and score one sequence.

    Returns preds/targets, the estimated and ground-truth trajectories, and
    ATE plus both RTE horizons (60s and the short ``rte_delta`` window).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    targets, preds = run_inference(network, loader, device)

    if use_3d:
        pos_pred = recon_traj_3d(dataset, preds)
        pos_gt = dataset.gt_pos[0]
    else:
        pos_pred = recon_traj_2d(dataset, preds)[:, :2]
        pos_gt = dataset.gt_pos[0][:, :2]

    ate, rte = compute_ate_rte(pos_pred, pos_gt, SAMPLE_RATE * 60)
    rte_delta = SAMPLE_RATE * 10 if rte_delta is None else int(rte_delta)
    return {
        "preds": preds,
        "targets": targets,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "ate": ate,
        "rte": rte,
        "rte_short": compute_rte_at_delta(pos_pred, pos_gt, rte_delta),
    }


def compare_against_baseline(network, baseline_ds, other_ds, outdir, device, args,
                             other_key, other_label, title=None, plots=True):
    """Score a modified-IMU sequence against the original through RoNIN.

    Every model in this repo is evaluated the same way - run RoNIN on the
    original IMU and on the VAE reconstruction / LDM generation of it, then
    report the deltas - so the pipeline runs, prints, figures and metric block
    live here rather than being repeated per script.

    Returns the metrics block, plus the two raw pipeline results for callers
    that need the trajectories themselves.
    """
    from ldm.evaluation import metrics, plots as plot_mod, report

    rte_delta = int(round(args.rte_delta_sec * SAMPLE_RATE))
    short = metrics.format_rte_sec(args.rte_delta_sec)

    results = {}
    width = max(len("Original"), len(other_label))
    for key, dataset, label in (
        ("original", baseline_ds, "Original"),
        (other_key, other_ds, other_label),
    ):
        results[key] = run_ronin_pipeline(
            network, dataset, device, use_3d=args.ronin_3d, rte_delta=rte_delta,
        )
        print(f"  {label:<{width}} — ATE: {results[key]['ate']:.4f}, "
              f"RTE_60s: {results[key]['rte']:.4f}, "
              f"RTE_{short}: {results[key]['rte_short']:.4f}")

    if plots:
        plot_mod.plot_trajectories(
            results["original"], results[other_key], outdir, other_label=other_label,
            rte_delta_sec=args.rte_delta_sec, title=title,
        )
        plot_mod.plot_position_error(
            results["original"], results[other_key], outdir, other_label=other_label,
        )
    # Also retain these for checkpoint-free --plots_only.
    report.write_trajectories(
        outdir, results["original"]["pos_gt"], results["original"]["pos_pred"],
        results[other_key]["pos_pred"],
    )

    block = {
        "ronin_ckpt": osp.abspath(args.ronin_ckpt),
        "rte_delta_sec": args.rte_delta_sec,
        "original": metrics.trajectory_metrics(results["original"]),
        other_key: metrics.trajectory_metrics(results[other_key]),
        "delta": metrics.trajectory_delta(
            metrics.trajectory_metrics(results[other_key]),
            metrics.trajectory_metrics(results["original"]),
        ),
    }
    return block, results
