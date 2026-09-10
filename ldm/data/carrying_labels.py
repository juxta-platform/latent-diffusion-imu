"""Derive per-window stationary labels that can override folder carrying type."""

from __future__ import annotations

import numpy as np


# What the classifier predicts. This is NOT the set of carrying placements a
# dataset may contain - see ldm.evaluation.constants.CARRYING_PLACEMENTS, which
# the VAE/LDM paths use and which may legitimately be larger (e.g. head).
# Recordings whose placement is absent here are still loaded and generated;
# they are simply excluded from classifier accuracy and confusion matrices.
#
# Order defines CLASS_TO_IDX and is baked into trained checkpoints, so append
# rather than insert, and retrain after changing it.
CARRYING_CLASS_NAMES = ["chest", "demo_hand", "pocket", "swinging"]
ACTIVITY_CLASS_NAMES = ["stationary"]
CLASS_NAMES = CARRYING_CLASS_NAMES + ACTIVITY_CLASS_NAMES
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

STATIONARY_HORIZON_S = 2.0
STATIONARY_SPEED_MPS = 0.25
IMU_GYRO_QUIET = 0.5
IMU_ACCEL_QUIET = 0.5


def validate_activity_threshold(value):
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"activity threshold must be in [0, 1), got {value}")
    return value


def _rolling_mean(x, horizon):
    left = (horizon - 1) // 2
    right = horizon - 1 - left
    padded = np.pad(x, (left, right), mode="edge")
    return np.convolve(padded, np.ones(horizon) / horizon, mode="valid")


def _rolling_std(x, horizon):
    mean = _rolling_mean(x, horizon)
    mean_sq = _rolling_mean(x * x, horizon)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _rolling_percentile(x, horizon, q):
    """Centered rolling percentile via sliding window views."""
    left = (horizon - 1) // 2
    right = horizon - 1 - left
    padded = np.pad(x, (left, right), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, horizon)
    return np.percentile(windows, q, axis=1)


def _imu_quiet_masks(imu, sample_rate):
    horizon = max(1, int(round(STATIONARY_HORIZON_S * sample_rate)))
    gyro_mag = np.linalg.norm(imu[:, 3:6], axis=1)
    accel_mag = np.linalg.norm(imu[:, 0:3], axis=1)
    gyro_p95 = _rolling_percentile(gyro_mag, horizon, 95)
    accel_std = _rolling_std(accel_mag, horizon)
    quiet = (gyro_p95 < IMU_GYRO_QUIET) & (accel_std < IMU_ACCEL_QUIET)
    return quiet, gyro_p95, accel_std


def _speed_mask(pos, sample_rate):
    pos2d = np.asarray(pos, dtype=np.float64)[:, :2]
    n = len(pos2d)
    horizon = max(1, int(round(STATIONARY_HORIZON_S * sample_rate)))
    if n < 2:
        return np.zeros(n, dtype=bool)
    speed = np.linalg.norm(np.diff(pos2d, axis=0), axis=1) * sample_rate
    speed = np.concatenate([speed[:1], speed])
    return _rolling_mean(speed, horizon) < STATIONARY_SPEED_MPS


def derive_window_labels(
    imu_raw,
    pos,
    base_class,
    n_windows,
    window_samples,
    stride_samples,
    sample_rate=200,
    activity_threshold=0.5,
):
    """Return class names for windows, with stationary overriding carrying type.

    Stationary is decided on rolling 2s segments: a sample is stationary when
    mean speed and IMU quietness both pass. A 10s window is labeled stationary
    when that coverage strictly exceeds ``activity_threshold``.
    """
    threshold = validate_activity_threshold(activity_threshold)
    imu_raw = np.asarray(imu_raw)
    have_pos = pos is not None and len(np.asarray(pos)) >= len(imu_raw)

    quiet_mask, gyro_p95_series, accel_std_series = _imu_quiet_masks(
        imu_raw, sample_rate
    )
    if have_pos:
        pos2d = np.asarray(pos, dtype=np.float64)[:len(imu_raw), :2]
        speed_ok = _speed_mask(pos2d, sample_rate)
        stationary = speed_ok & quiet_mask
    else:
        stationary = quiet_mask

    labels = []
    details = []
    for i in range(n_windows):
        s = i * stride_samples
        e = min(s + window_samples, len(imu_raw))
        if e <= s:
            break

        stationary_fraction = float(np.mean(stationary[s:e]))
        quiet_fraction = float(np.mean(quiet_mask[s:e]))
        gyro_p95 = float(np.median(gyro_p95_series[s:e]))
        accel_std = float(np.median(accel_std_series[s:e]))

        if stationary_fraction > threshold:
            label = "stationary"
        else:
            label = base_class

        labels.append(label)
        details.append({
            "window": i,
            "stationary_fraction": stationary_fraction,
            "imu_quiet_fraction": quiet_fraction,
            "gyro_p95": gyro_p95,
            "accel_magnitude_std": accel_std,
            "label": label,
        })

    return labels, details
