"""Trajectory, reconstruction and classification metrics."""

import numpy as np
import torch

from ldm.evaluation.constants import CHANNEL_NAMES


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def compute_ate(est, gt):
    """Absolute trajectory error: RMS position difference over the sequence."""
    return float(np.sqrt(np.mean((est - gt) ** 2)))


def compute_rte(est, gt, delta):
    """Relative trajectory error over a fixed ``delta``-sample horizon."""
    if delta <= 0 or delta >= est.shape[0]:
        return 0.0
    err = est[delta:] + gt[:-delta] - est[:-delta] - gt[delta:]
    return float(np.sqrt(np.mean(err ** 2)))


def compute_rte_at_delta(est, gt, delta):
    """RTE at ``delta``, extrapolated when the sequence is shorter than that."""
    if delta <= 0:
        return 0.0
    if est.shape[0] < delta:
        return compute_rte(est, gt, est.shape[0] - 1) * (delta / est.shape[0])
    return compute_rte(est, gt, delta)


def compute_ate_rte(est, gt, pred_per_min=12000):
    return compute_ate(est, gt), compute_rte_at_delta(est, gt, pred_per_min)


def format_rte_sec(sec):
    return f"{int(sec)}s" if float(sec).is_integer() else f"{sec:g}s"


def rte_title_suffix(result, rte_delta_sec):
    """One-line ATE/RTE summary used in figure titles."""
    return (f"ATE={result['ate']:.3f}  RTE_60s={result['rte']:.3f}  "
            f"RTE_{format_rte_sec(rte_delta_sec)}={result['rte_short']:.3f}")


def trajectory_metrics(result):
    """The scalar subset of a RoNIN pipeline result, ready for JSON."""
    return {
        "ate": result["ate"],
        "rte": result["rte"],
        "rte_short": result["rte_short"],
    }


def trajectory_delta(after, before):
    """Signed change in each trajectory metric, ``after - before``."""
    return {key: after[key] - before[key] for key in ("ate", "rte", "rte_short")}


def mean_metric(rows, side, key):
    """Mean of ``rows[i][side][key]``, or None when nothing was collected."""
    values = [row[side][key] for row in rows if side in row and key in row[side]]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# Signal reconstruction
# ---------------------------------------------------------------------------

def channel_metrics(inputs, outputs):
    """Per-channel MSE / RMSE / MAE between two [B, 6, T] tensors."""
    diff = outputs - inputs
    mse = diff.pow(2).mean(dim=(0, 2))
    mae = diff.abs().mean(dim=(0, 2))
    return {
        "per_channel_mse": {name: mse[i].item() for i, name in enumerate(CHANNEL_NAMES)},
        "per_channel_rmse": {
            name: float(torch.sqrt(mse[i]).item()) for i, name in enumerate(CHANNEL_NAMES)
        },
        "per_channel_mae": {name: mae[i].item() for i, name in enumerate(CHANNEL_NAMES)},
    }


def physical_metrics(inputs_phys, outputs_phys):
    """MSE / RMSE / MAE in physical units (m/s^2 and rad/s)."""
    mse = (inputs_phys - outputs_phys).pow(2).mean().item()
    return {
        "physical_mse": mse,
        "physical_rmse": float(np.sqrt(mse)),
        "physical_mae": (inputs_phys - outputs_phys).abs().mean().item(),
    }


def window_rmse(inputs, outputs):
    """RMSE of one [T] channel pair, for per-panel figure annotations."""
    return float(np.sqrt(np.mean((np.asarray(inputs) - np.asarray(outputs)) ** 2)))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def softmax(x):
    """Softmax over the last axis; accepts one logit vector or a stack."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def prediction_indices(predictions, class_names):
    """Class indices for predictions given as names (or already as indices)."""
    return [class_names.index(p) if p in class_names else int(p) for p in predictions]


def confusion(labels, predictions, class_names):
    """Confusion matrix and accuracy over per-window label/prediction names.

    Windows whose ground-truth placement is outside ``class_names`` are
    dropped: a dataset may hold carrying types this classifier never predicts.
    """
    from sklearn.metrics import confusion_matrix

    index = {name: i for i, name in enumerate(class_names)}
    pairs = [(t, p) for t, p in zip(labels, predictions) if t in index]
    if not pairs:
        return None
    y_true = np.array([index[t] for t, _ in pairs])
    y_pred = np.array([index[p] for _, p in pairs])
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {
        "matrix": matrix,
        "n_windows": len(y_true),
        "accuracy": float(np.mean(y_true == y_pred)),
    }
