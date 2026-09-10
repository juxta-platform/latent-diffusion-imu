"""Train a carrying-type classifier on frozen VAE latent features.

Example usage:
python scripts/train_carrying_classifier.py \
    --vae_ckpt logs/vae_1d/full_dataset_local/checkpoints/last.ckpt \
    --input_dir data/real_data_by_carrying_type \
    --vae_stats data/dataset_processed_overlapped/stats.pt \
    --imu_frame local \
    --outdir outputs/carrying_classifier
"""

import argparse
import json
import os


import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from ldm.data.carrying_dataset import (
    CLASS_NAMES, CarryingTypeDataset, build_file_list, split_files_stratified,
)
from ldm.data.imu_dataset import resolve_stats
from ldm.evaluation import models
from ldm.evaluation.constants import DEFAULT_VAE_CONFIG
from ldm.models.carrying_classifier import CarryingTypeClassifier, LatentMLPClassifier


@torch.no_grad()
def encode_dataset(clf, loader, device):
    """Pre-encode all windows into latent features + labels."""
    all_z, all_labels = [], []
    for batch in loader:
        x = batch['imu'].float().to(device)
        z = clf.encode(x)  # [B, 800]
        all_z.append(z.cpu())
        all_labels.append(torch.tensor(batch['label']))
    return torch.cat(all_z), torch.cat(all_labels)


def train_epoch(mlp, features, labels, optimizer, criterion, batch_size, device):
    mlp.train()
    n = len(features)
    perm = torch.randperm(n)
    total_loss, correct, total = 0.0, 0, 0

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        x = features[idx].to(device)
        y = labels[idx].to(device)

        logits = mlp(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(idx)

    return total_loss / total, correct / total


def compute_class_weights(labels, n_classes):
    """Inverse-frequency weights so each class contributes equally in expectation."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return len(labels) / (n_classes * counts)


@torch.no_grad()
def eval_epoch(mlp, features, labels, criterion, batch_size, device):
    mlp.eval()
    n = len(features)
    total_loss, correct, total = 0.0, 0, 0

    for i in range(0, n, batch_size):
        x = features[i:i + batch_size].to(device)
        y = labels[i:i + batch_size].to(device)

        logits = mlp(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(x)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train carrying-type classifier")
    parser.add_argument("--vae_config", "--config", dest="vae_config", type=str,
                        default=DEFAULT_VAE_CONFIG)
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--input_dir", "--data_dir", dest="input_dir", type=str,
                        required=True,
                        help="Root dir with one <class>/*.hdf5 folder per class in "
                             "CLASS_NAMES; other placement folders are ignored")
    parser.add_argument("--vae_stats", "--stats", dest="vae_stats", type=str,
                        default=None,
                        help="Path to stats.pt (falls back to ckpt-embedded stats)")
    parser.add_argument("--outdir", type=str, default="outputs/carrying_classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--stride_sec", type=float, default=2.0)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--sample_rate", type=int, default=200)
    parser.add_argument("--latent_length", type=int, default=100)
    parser.add_argument(
        "--activity_threshold", type=float, default=0.5,
        help="Stationary must cover more than this fraction of a "
             "window to override its folder carrying class (default: 0.5)",
    )
    parser.add_argument(
        "--imu_frame",
        type=str,
        default="world",
        choices=["local", "world"],
        help="IMU frame for training windows: 'local' (device) or 'world' "
             "(HDF5 rotated via game_rv). Recorded in the checkpoint so "
             "eval_classifier.py picks it up automatically.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 <= args.activity_threshold < 1.0:
        parser.error("--activity_threshold must be in [0, 1)")

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    print("Loading VAE ...")
    vae, _config = models.load_vae(args.vae_config, args.vae_ckpt, device)
    if vae.imu_frame is not None and vae.imu_frame != args.imu_frame:
        parser.error("The VAE checkpoint frame must match --imu_frame")
    imu_mean, imu_std = resolve_stats(model=vae, stats_path=args.vae_stats)
    print(f"  imu_mean: {imu_mean.view(-1).numpy()}")
    print(f"  imu_std:  {imu_std.view(-1).numpy()}")

    # Build datasets
    print("Building datasets ...")
    file_list = build_file_list(args.input_dir)
    train_files, val_files = split_files_stratified(
        file_list, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"  {len(train_files)} train files, {len(val_files)} val files")

    print(f"  imu_frame: {args.imu_frame}")
    print(f"  activity_threshold: {args.activity_threshold}")
    train_ds = CarryingTypeDataset(
        train_files, imu_mean, imu_std,
        window_sec=args.window_sec, sample_rate=args.sample_rate,
        latent_length=args.latent_length, stride_sec=args.stride_sec,
        imu_frame=args.imu_frame, activity_threshold=args.activity_threshold,
    )
    val_ds = CarryingTypeDataset(
        val_files, imu_mean, imu_std,
        window_sec=args.window_sec, sample_rate=args.sample_rate,
        latent_length=args.latent_length, stride_sec=args.stride_sec,
        imu_frame=args.imu_frame, activity_threshold=args.activity_threshold,
    )
    print(f"  {len(train_ds)} train windows, {len(val_ds)} val windows")

    # Print class distribution
    for name, ds in [("train", train_ds), ("val", val_ds)]:
        counts = [0] * len(CLASS_NAMES)
        for lbl in ds.labels:
            counts[lbl] += 1
        print(f"  {name}: {dict(zip(CLASS_NAMES, counts))}")

    # Pre-encode all windows
    print("Encoding windows through frozen VAE ...")
    in_dim = vae.embed_dim * args.latent_length  # 8 * 100 = 800
    mlp = LatentMLPClassifier(
        in_dim=in_dim, hidden=tuple(args.hidden),
        n_classes=len(CLASS_NAMES), dropout=args.dropout,
    ).to(device)
    clf = CarryingTypeClassifier(vae, mlp)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_z, train_y = encode_dataset(clf, train_loader, device)
    val_z, val_y = encode_dataset(clf, val_loader, device)
    print(f"  Encoded: train {train_z.shape}, val {val_z.shape}")

    # Train MLP
    class_weights = compute_class_weights(train_y.numpy(), len(CLASS_NAMES))
    print(f"  class_weights: {dict(zip(CLASS_NAMES, class_weights.round(3)))}")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = []
    print(f"\nTraining MLP for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(mlp, train_z, train_y, optimizer, criterion,
                                      args.batch_size, device)
        va_loss, va_acc = eval_epoch(mlp, val_z, val_y, criterion, args.batch_size, device)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
        })
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                'mlp_state_dict': mlp.state_dict(),
                'in_dim': in_dim,
                'hidden': list(args.hidden),
                'n_classes': len(CLASS_NAMES),
                'dropout': args.dropout,
                'class_names': CLASS_NAMES,
                'imu_frame': args.imu_frame,
                'activity_threshold': args.activity_threshold,
                'class_weights': class_weights.tolist(),
                'imu_mean': imu_mean.view(-1),
                'imu_std': imu_std.view(-1),
                'epoch': epoch,
                'val_acc': va_acc,
            }, os.path.join(args.outdir, "best_classifier.pt"))

    # Final evaluation
    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    mlp.eval()
    with torch.no_grad():
        val_logits = mlp(val_z.to(device))
        val_preds = val_logits.argmax(1).cpu().numpy()
        val_labels = val_y.numpy()

    print("\n=== Validation Report ===")
    class_ids = list(range(len(CLASS_NAMES)))
    print(classification_report(
        val_labels, val_preds, labels=class_ids, target_names=CLASS_NAMES,
        digits=3, zero_division=0,
    ))
    cm = confusion_matrix(val_labels, val_preds, labels=class_ids)
    print("Confusion matrix:")
    print(cm)

    # Save training history + final metrics
    results = {
        "args": vars(args),
        "best_val_acc": best_val_acc,
        "history": history,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "class_weights": class_weights.tolist(),
        "train_files": [os.path.basename(f) for f, _ in train_files],
        "val_files": [os.path.basename(f) for f, _ in val_files],
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save split for reproducibility
    with open(os.path.join(args.outdir, "split.txt"), "w") as f:
        f.write(f"# seed={args.seed} val_fraction={args.val_fraction}\n")
        f.write(f"\n[train]\n")
        for path, cls in train_files:
            f.write(f"{cls}/{os.path.basename(path)}\n")
        f.write(f"\n[val]\n")
        for path, cls in val_files:
            f.write(f"{cls}/{os.path.basename(path)}\n")

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
