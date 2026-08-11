"""Evaluate the carrying-type classifier.

Modes:
  1) Labeled split (train/val):
       --data_dir data/real_data_by_carrying_type --split val
     Uses overlapping windows from the same seeded per-file split as training.
     Prints accuracy, confusion matrix, and per-window IMU plots with GT & Pred.

  2) Single trajectory (hdf5/parquet):
       --input data/some_recording.hdf5
     Uses consecutive (non-overlapping) windows for chronological readout.
     Produces per-window IMU plots with Pred label + a timeline summary.
     --imu_frame local|world selects device vs global IMU (HDF5 rotates via
     game_rv; parquet reads *_local_* or *_world_* columns). For LDM-generated
     parquet with stale local columns, pass --parquet_from_world with
     --imu_frame local to rotate world→local via phone_rot.

Example usage:
  python scripts/eval_carrying_classifier.py \
      --config configs/imu/vae_1d.yaml \
      --vae_ckpt logs/vae_1d/.../total_loss=0.0116.ckpt \
      --clf_ckpt logs/carrying_classifier/best_classifier.pt \
      --data_dir data/real_data_by_carrying_type --split val \
      --outdir outputs/carrying_eval

  python scripts/eval_carrying_classifier.py \
      --config configs/imu/vae_1d.yaml \
      --vae_ckpt logs/vae_1d/.../total_loss=0.0116.ckpt \
      --clf_ckpt logs/carrying_classifier/best_classifier.pt \
      --input data/real_data_by_carrying_type/pocket/john_left_pocket_ios_corrected.hdf5 \
      --outdir outputs/carrying_eval_single
"""

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from ldm.data.carrying_dataset import (
    CLASS_NAMES, CLASS_TO_IDX, CarryingTypeDataset,
    build_file_list, split_files_stratified,
)
from ldm.data.imu_dataset import resolve_stats
from ldm.models.carrying_classifier import CarryingTypeClassifier, LatentMLPClassifier
from ldm.util import instantiate_from_config

CHANNEL_NAMES = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
SAMPLE_RATE = 200
WINDOW_SAMPLES = 2000  # 10s at 200Hz

ACCEL_LOCAL_COLS = ["accel_local_x", "accel_local_y", "accel_local_z"]
GYRO_LOCAL_COLS = ["gyro_local_x", "gyro_local_y", "gyro_local_z"]
ACCEL_WORLD_COLS = ["accel_world_x", "accel_world_y", "accel_world_z"]
GYRO_WORLD_COLS = ["gyro_world_x", "gyro_world_y", "gyro_world_z"]
PHONE_ROT_COLS = ["phone_rot_x", "phone_rot_y", "phone_rot_z", "phone_rot_w"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_process_file():
    path = os.path.join(os.path.dirname(__file__), "preprocess_imu.py")
    spec = importlib.util.spec_from_file_location("preprocess_imu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _export_world_to_sim(v, heading_rad=0.0):
    """Invert juxta-simulator-rs map_world_frame (heading then Y-up→Z-up axes)."""
    ch, sh = np.cos(heading_rad), np.sin(heading_rad)
    ex, ey, ez = v[:, 0], v[:, 1], v[:, 2]
    bx = ch * ex + sh * ey
    by = -sh * ex + ch * ey
    bz = ez
    # inv map_world_axes: (bx,by,bz)=(sx,-sz,sy) → (sx,sy,sz)=(bx,bz,-by)
    return np.stack([bx, bz, -by], axis=1)


def world_imu_to_local_parquet(accel_w, gyro_w, phone_rot_xyzw, heading_rad=0.0):
    """Rotate exported parquet world-frame IMU to phone-local via phone_rot.

    Matches simulator: accel_local = phone_rot^{-1} * accel_world_sim, after
    undoing the export-frame axis remap / heading applied to world columns.
    """
    from scipy.spatial.transform import Rotation as R

    accel_sim = _export_world_to_sim(accel_w, heading_rad)
    gyro_sim = _export_world_to_sim(gyro_w, heading_rad)
    rot = R.from_quat(phone_rot_xyzw)
    return rot.inv().apply(accel_sim), rot.inv().apply(gyro_sim)


def load_parquet_imu(path, imu_frame="local", parquet_from_world=False,
                     world_heading=0.0):
    """Load [N, 6] IMU (accel|gyro) from a simulator parquet.

    imu_frame:
      - local: use accel_local_*/gyro_local_* (or rotate world→local if
               parquet_from_world / local columns missing)
      - world: use accel_world_*/gyro_world_* directly
    """
    import pandas as pd

    df = pd.read_parquet(path)
    has_local = all(c in df.columns for c in ACCEL_LOCAL_COLS + GYRO_LOCAL_COLS)
    has_world = all(c in df.columns for c in ACCEL_WORLD_COLS + GYRO_WORLD_COLS)
    has_rot = all(c in df.columns for c in PHONE_ROT_COLS)

    if imu_frame == "world":
        if not has_world:
            raise ValueError(f"{path}: missing world IMU columns for --imu_frame world")
        print("  parquet IMU: world columns")
        accel = df[ACCEL_WORLD_COLS].values.astype(np.float64)
        gyro = df[GYRO_WORLD_COLS].values.astype(np.float64)
        return np.concatenate([accel, gyro], axis=1)

    # imu_frame == "local"
    use_world_src = parquet_from_world or not has_local
    if not use_world_src:
        print("  parquet IMU: local columns")
        accel = df[ACCEL_LOCAL_COLS].values.astype(np.float64)
        gyro = df[GYRO_LOCAL_COLS].values.astype(np.float64)
        return np.concatenate([accel, gyro], axis=1)

    if not has_world:
        raise ValueError(
            f"{path}: need world IMU columns to rotate to local "
            f"(parquet_from_world={parquet_from_world}, has_local={has_local})"
        )
    if not has_rot:
        raise ValueError(
            f"{path}: missing phone_rot_*; cannot rotate world IMU to local"
        )

    accel_w = df[ACCEL_WORLD_COLS].values.astype(np.float64)
    gyro_w = df[GYRO_WORLD_COLS].values.astype(np.float64)
    phone_rot = df[PHONE_ROT_COLS].values.astype(np.float64)
    print(
        f"  parquet IMU: world→local via phone_rot "
        f"(heading={world_heading:.4f} rad)"
        + ("; local columns ignored (stale after LDM gen)" if has_local else "")
    )
    accel, gyro = world_imu_to_local_parquet(
        accel_w, gyro_w, phone_rot, heading_rad=world_heading
    )
    return np.concatenate([accel, gyro], axis=1)


def load_vae(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def load_classifier(clf_ckpt_path, device):
    ckpt = torch.load(clf_ckpt_path, map_location="cpu")
    mlp = LatentMLPClassifier(
        in_dim=ckpt['in_dim'],
        hidden=tuple(ckpt['hidden']),
        n_classes=ckpt['n_classes'],
        dropout=ckpt.get('dropout', 0.0),
    )
    mlp.load_state_dict(ckpt['mlp_state_dict'])
    class_names = ckpt.get('class_names', CLASS_NAMES)
    return mlp.to(device).eval(), class_names, ckpt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_imu_window(imu, outdir, idx, gt_label=None, pred_label=None,
                    sample_rate=200, time_offset=0.0):
    """Plot 6-channel IMU for one window with GT/Pred in title."""
    t = np.arange(imu.shape[-1]) / sample_rate + time_offset
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for c, ax in enumerate(axes):
        ax.plot(t, imu[c], linewidth=0.8, alpha=0.9)
        ax.set_ylabel(CHANNEL_NAMES[c], fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")

    parts = [f"Window {idx}"]
    if gt_label is not None:
        parts.append(f"GT: {gt_label}")
    if pred_label is not None:
        parts.append(f"Pred: {pred_label}")
    fig.suptitle("  |  ".join(parts), fontsize=12)

    path = os.path.join(outdir, f"window_{idx:03d}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_timeline(predictions, class_names, outdir, window_sec=10.0, logits=None):
    """Plot predicted class vs time, optionally with logits and softmax confidence."""
    n = len(predictions)
    t_start = np.arange(n) * window_sec
    t_mid = t_start + window_sec / 2
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))

    has_logits = logits is not None and len(logits) == n
    n_rows = 3 if has_logits else 1
    height = 9 if has_logits else 4
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(max(8, n * 0.6), height),
        sharex=True, constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    # --- Panel 0: predicted class bars ---
    ax = axes[0]
    for i, (ts, pred) in enumerate(zip(t_mid, predictions)):
        ax.barh(0, window_sec, left=t_start[i], height=0.6,
                color=colors[pred], edgecolor='white', linewidth=0.5)
        label = class_names[pred]
        if has_logits:
            probs = _softmax(logits[i])
            label = f"{class_names[pred]}\n{probs[pred]:.2f}"
        ax.text(ts, 0, label, ha='center', va='center', fontsize=7)
    ax.set_yticks([])
    ax.set_title("Predicted carrying type" + (" (label = class, conf)" if has_logits else ""))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(class_names))]
    ax.legend(handles, class_names, loc='upper right', fontsize=8)

    if has_logits:
        logits = np.asarray(logits, dtype=np.float64)

        # --- Panel 1: raw logits ---
        ax = axes[1]
        for c, name in enumerate(class_names):
            ax.plot(t_mid, logits[:, c], marker='o', markersize=4,
                    linewidth=1.2, color=colors[c], label=name)
        ax.set_ylabel("logit")
        ax.set_title("Class logits per window")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # --- Panel 2: softmax probabilities ---
        ax = axes[2]
        probs = np.stack([_softmax(logits[i]) for i in range(n)], axis=0)
        for c, name in enumerate(class_names):
            ax.plot(t_mid, probs[:, c], marker='o', markersize=4,
                    linewidth=1.2, color=colors[c], label=name)
        ax.set_ylabel("softmax prob")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Class confidence (softmax)")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (s)")

    path = os.path.join(outdir, "timeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved timeline to {path}")


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def plot_confusion_matrix(cm, class_names, outdir):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.colorbar(im, ax=ax)
    path = os.path.join(outdir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {path}")


def plot_training_curves(results_path, outdir):
    """Plot train/val loss and accuracy from training results.json history."""
    with open(results_path) as f:
        results = json.load(f)
    history = results.get("history")
    if not history:
        print(f"No history found in {results_path}; skipping training curves")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    best_val = results.get("best_val_acc")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(epochs, train_loss, label="train", alpha=0.85)
    ax.plot(epochs, val_loss, label="val", alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Cross-entropy loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, train_acc, label="train", alpha=0.85)
    ax.plot(epochs, val_acc, label="val", alpha=0.85)
    if best_val is not None:
        ax.axhline(best_val, color="C2", linestyle="--", linewidth=1.0,
                   label=f"best val={best_val:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def resolve_train_results(train_results, clf_ckpt):
    """Find training results.json: explicit path, or next to clf_ckpt."""
    if train_results is not None:
        if not os.path.isfile(train_results):
            raise FileNotFoundError(f"--train_results not found: {train_results}")
        return train_results
    guess = os.path.join(os.path.dirname(os.path.abspath(clf_ckpt)), "results.json")
    if os.path.isfile(guess):
        return guess
    return None


# ---------------------------------------------------------------------------
# Single trajectory mode
# ---------------------------------------------------------------------------

def eval_single_trajectory(args, vae, mlp, imu_mean, imu_std, device, class_names):
    """Classify consecutive non-overlapping windows from a single file."""
    preprocess = _load_process_file()
    ext = os.path.splitext(args.input)[1].lower()
    local_frame = args.imu_frame == "local"

    if ext in (".hdf5", ".h5"):
        acce, gyro, pos, time = preprocess.load_hdf5(
            args.input, local_frame=local_frame
        )
        print(f"  hdf5 IMU: {'local' if local_frame else 'world (via game_rv)'}")
        imu_raw = np.concatenate([acce, gyro], axis=1)  # [N, 6]
    elif ext == ".parquet":
        imu_raw = load_parquet_imu(
            args.input,
            imu_frame=args.imu_frame,
            parquet_from_world=args.parquet_from_world,
            world_heading=args.world_heading,
        )
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    N = imu_raw.shape[0]
    n_windows = N // WINDOW_SAMPLES
    if n_windows == 0:
        print(f"Recording too short for a full 10s window ({N} samples)")
        return

    print(f"  {N} samples ({N / SAMPLE_RATE:.1f}s), {n_windows} consecutive windows")

    plot_dir = os.path.join(args.outdir, "window_plots")
    os.makedirs(plot_dir, exist_ok=True)

    mean = imu_mean.view(6, 1)
    std = imu_std.view(6, 1)
    predictions = []
    all_logits = []

    clf = CarryingTypeClassifier(vae, mlp)
    clf.eval()

    for i in range(n_windows):
        s, e = i * WINDOW_SAMPLES, (i + 1) * WINDOW_SAMPLES
        w_imu = torch.tensor(imu_raw[s:e].T, dtype=torch.float32)  # [6, 2000]
        w_std = (w_imu - mean) / std

        with torch.no_grad():
            logits = clf(w_std.unsqueeze(0).to(device))
            logits_np = logits.squeeze(0).cpu().numpy()
            pred = int(logits_np.argmax())

        predictions.append(pred)
        all_logits.append(logits_np)
        time_offset = i * (WINDOW_SAMPLES / SAMPLE_RATE)

        conf = float(_softmax(logits_np)[pred])
        pred_label = f"{class_names[pred]} ({conf:.2f})"
        if args.n_plot < 0 or i < args.n_plot:
            plot_imu_window(w_imu.numpy(), plot_dir, i,
                            gt_label="N/A", pred_label=pred_label,
                            sample_rate=SAMPLE_RATE, time_offset=time_offset)

    all_logits = np.stack(all_logits, axis=0)
    plot_timeline(predictions, class_names, args.outdir,
                  window_sec=WINDOW_SAMPLES / SAMPLE_RATE, logits=all_logits)

    # Majority vote
    from collections import Counter
    counts = Counter(predictions)
    majority = counts.most_common(1)[0]
    probs = np.stack([_softmax(all_logits[i]) for i in range(n_windows)], axis=0)
    print(f"\nPer-window predictions: {[class_names[p] for p in predictions]}")
    print(f"Per-window confidence: {[f'{probs[i, predictions[i]]:.2f}' for i in range(n_windows)]}")
    print(f"Majority vote: {class_names[majority[0]]} ({majority[1]}/{len(predictions)} windows)")

    results = {
        "input": os.path.abspath(args.input),
        "n_windows": n_windows,
        "predictions": [class_names[p] for p in predictions],
        "logits": all_logits.tolist(),
        "softmax": probs.tolist(),
        "confidence": [float(probs[i, predictions[i]]) for i in range(n_windows)],
        "majority_vote": class_names[majority[0]],
        "majority_count": majority[1],
        "class_names": list(class_names),
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    n_plotted = n_windows if args.n_plot < 0 else min(args.n_plot, n_windows)
    print(f"Saved {n_plotted} window plots to {plot_dir}")


# ---------------------------------------------------------------------------
# Labeled split mode
# ---------------------------------------------------------------------------

def eval_labeled_split(args, vae, mlp, imu_mean, imu_std, device, class_names):
    """Evaluate on the labeled train/val split with overlapping windows."""
    file_list = build_file_list(args.data_dir)
    train_files, val_files = split_files_stratified(
        file_list, val_fraction=args.val_fraction, seed=args.seed
    )
    files = train_files if args.split == "train" else val_files
    print(f"  {len(files)} files in {args.split} split")

    ds = CarryingTypeDataset(
        files, imu_mean, imu_std,
        window_sec=args.window_sec, sample_rate=SAMPLE_RATE,
        latent_length=args.latent_length, stride_sec=args.stride_sec,
        imu_frame=args.imu_frame,
    )
    print(f"  {len(ds)} windows (imu_frame={args.imu_frame})")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    clf = CarryingTypeClassifier(vae, mlp)
    clf.eval()

    all_preds, all_labels, all_imu = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['imu'].float().to(device)
            logits = clf(x)
            preds = logits.argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(torch.tensor(batch['label']))
            all_imu.append(batch['imu'].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_imu = torch.cat(all_imu)

    acc = (all_preds == all_labels).mean()
    print(f"\n=== {args.split} Results ({len(all_preds)} windows) ===")
    print(f"Overall accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)

    plot_confusion_matrix(cm, class_names, args.outdir)

    # Per-window IMU plots
    plot_dir = os.path.join(args.outdir, "window_plots")
    os.makedirs(plot_dir, exist_ok=True)
    n_plot = len(all_preds) if args.n_plot < 0 else min(args.n_plot, len(all_preds))
    for i in range(n_plot):
        plot_imu_window(
            all_imu[i].numpy(), plot_dir, i,
            gt_label=class_names[all_labels[i]],
            pred_label=class_names[all_preds[i]],
            sample_rate=SAMPLE_RATE,
        )
    print(f"Saved {n_plot} window plots to {plot_dir}")

    results = {
        "split": args.split,
        "n_windows": len(all_preds),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate carrying-type classifier")
    parser.add_argument("--config", type=str, default="configs/imu/vae_1d.yaml")
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--clf_ckpt", type=str, required=True,
                        help="Classifier checkpoint (best_classifier.pt)")
    parser.add_argument("--input", type=str, default=None,
                        help="Single .hdf5 or .parquet trajectory")
    parser.add_argument(
        "--imu_frame",
        type=str,
        default=None,
        choices=["local", "world"],
        help="IMU frame fed to the classifier: 'local' (device) or 'world' "
             "(HDF5 via game_rv; parquet *_world_* columns). "
             "Defaults to imu_frame stored in --clf_ckpt, else 'local'.",
    )
    parser.add_argument(
        "--parquet_from_world",
        action="store_true",
        help="With --imu_frame local, rotate parquet *_world_* → local via "
             "phone_rot instead of reading *_local_* (for LDM-generated parquet).",
    )
    parser.add_argument(
        "--world_heading",
        type=float,
        default=0.0,
        help="Simulator world_heading_offset (rad) when rotating parquet "
             "world→local (--parquet_from_world)",
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Labeled data root with class subfolders")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="outputs/carrying_eval")
    parser.add_argument("--n_plot", type=int, default=8,
                        help="Number of window plots (-1 = all)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--stride_sec", type=float, default=2.0)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--latent_length", type=int, default=100)
    parser.add_argument("--train_results", type=str, default=None,
                        help="Path to training results.json (with history). "
                             "Defaults to results.json next to --clf_ckpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading VAE ...")
    vae = load_vae(args.config, args.vae_ckpt, device)

    print("Loading classifier ...")
    mlp, class_names, clf_ckpt = load_classifier(args.clf_ckpt, device)

    if args.imu_frame is None:
        args.imu_frame = clf_ckpt.get("imu_frame", "local")
    print(f"  imu_frame: {args.imu_frame}")

    # Training curves (from train results.json next to clf ckpt, or --train_results)
    train_results = resolve_train_results(args.train_results, args.clf_ckpt)
    if train_results is not None:
        print(f"Plotting training curves from {train_results}")
        plot_training_curves(train_results, args.outdir)
    else:
        print("No training results.json found; skipping training curves "
              "(pass --train_results)")

    # Resolve stats: prefer clf checkpoint (which embeds them), then VAE, then --stats
    if 'imu_mean' in clf_ckpt and 'imu_std' in clf_ckpt:
        imu_mean = clf_ckpt['imu_mean'].view(-1, 1)
        imu_std = clf_ckpt['imu_std'].view(-1, 1)
        print("  Using stats from classifier checkpoint")
    else:
        imu_mean, imu_std = resolve_stats(model=vae, stats_path=args.stats)
        print("  Using stats from VAE / --stats")

    if args.input is not None:
        print(f"\nEvaluating single trajectory: {args.input}")
        eval_single_trajectory(args, vae, mlp, imu_mean, imu_std, device, class_names)
    elif args.data_dir is not None:
        print(f"\nEvaluating labeled {args.split} split from {args.data_dir}")
        eval_labeled_split(args, vae, mlp, imu_mean, imu_std, device, class_names)
    else:
        raise ValueError("Provide --input (single trajectory) or --data_dir (labeled split)")

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
