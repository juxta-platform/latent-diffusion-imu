"""Tensor-level operations on IMU windows: channel order, scaling, conditioning.

Everything here works on arrays and tensors that are already in memory.
Reading recordings from disk and cutting them into windows lives in
:mod:`ldm.evaluation.sequences`, which this module builds on.
"""

import numpy as np
import torch

from ldm.evaluation import sequences
from ldm.evaluation.constants import LATENT_LENGTH, WINDOW_SAMPLES


def swap_imu_channels(imu):
    """Convert between ``[accel(3), gyro(3)]`` and ``[gyro(3), accel(3)]``.

    The permutation is its own inverse, so one function covers both directions.
    Accepts numpy arrays or torch tensors shaped [N, 6].
    """
    return imu[:, [3, 4, 5, 0, 1, 2]]


def standardize(x, mean, std):
    """Z-score IMU shaped [..., C, T] with [C, 1] statistics."""
    return (x - mean) / std


def inverse_standardize(x, mean, std):
    """Undo :func:`standardize`, returning physical units."""
    return x * std.to(x.device) + mean.to(x.device)


def window_starts(n_samples, window, stride):
    """Start indices of every full window of ``window`` samples."""
    if n_samples < window:
        return []
    return list(range(0, n_samples - window + 1, stride))


def build_conditioning(ts, gt_pos, starts, window, vel_mean, vel_std, device,
                       latent_length=LATENT_LENGTH):
    """Velocity + physical_time conditioning for a batch of window starts.

    Per-window signals come from :func:`sequences.window_conditioning`, the
    same construction the preprocessing window builders use, so generation-time
    conditioning matches what the LDM saw in training. Velocity is then
    standardized with the LDM's velocity statistics.

    Returns ``{"velocity": [B, 2, L], "physical_time": [B, 1, L]}`` on device.
    """
    per_window = [
        sequences.window_conditioning(
            ts, gt_pos, start, start + window, latent_length
        )
        for start in starts
    ]
    velocity = torch.stack([vel for vel, _ in per_window])
    physical_time = torch.stack([time for _, time in per_window])

    velocity = (velocity - vel_mean.cpu()) / vel_std.cpu()
    return {
        "velocity": velocity.to(device),
        "physical_time": physical_time.to(device),
    }


def latent_shape(model, latent_length=LATENT_LENGTH):
    """(z_channels, latent_length) sampling shape for a latent diffusion model."""
    return int(getattr(model.first_stage_model, "embed_dim", 8)), latent_length


def split_imu_windows(imu, window_samples=WINDOW_SAMPLES):
    """Stack consecutive non-overlapping windows of [N, 6] IMU into [W, 6, T]."""
    n_windows = imu.shape[0] // window_samples
    if n_windows == 0:
        raise ValueError(
            f"{imu.shape[0]} samples is short of one {window_samples}-sample window"
        )
    return np.stack([
        np.asarray(imu[i * window_samples:(i + 1) * window_samples]).T
        for i in range(n_windows)
    ]).astype(np.float32)
