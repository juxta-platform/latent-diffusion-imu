"""Preprocess HDF5 IMU recordings into windowed .pt files for training."""

import argparse
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def load_hdf5(path):
    """Load synced IMU data from HDF5 file."""
    with h5py.File(path, 'r') as f:
        acce = f['synced/acce'][:]       # [N, 3]
        gyro = f['synced/gyro'][:]       # [N, 3]
        pos = f['synced/tango_pos'][:]   # [N, 3]
        time = f['synced/time'][:]       # [N]
    return acce, gyro, pos, time


def process_file(path, window_samples, latent_length):
    """Process one HDF5 file into a list of window dicts."""
    acce, gyro, pos, time = load_hdf5(path)

    imu = np.concatenate([acce, gyro], axis=1)  # [N, 6]
    pos_2d = pos[:, [0, 2]]                     # [N, 2] (x, z)

    n_windows = len(imu) // window_samples
    windows = []

    for i in range(n_windows):
        s = i * window_samples
        e = s + window_samples

        w_imu = torch.tensor(imu[s:e].T, dtype=torch.float32)        # [6, 2000]
        w_pos = torch.tensor(pos_2d[s:e].T, dtype=torch.float32)     # [2, 2000]
        w_time = torch.tensor(time[s:e], dtype=torch.float32)         # [2000]

        # Velocity via forward difference
        dt = w_time[1:] - w_time[:-1]  # [1999]
        dp = w_pos[:, 1:] - w_pos[:, :-1]  # [2, 1999]
        vel = dp / dt.unsqueeze(0)  # [2, 1999]
        vel = torch.cat([vel[:, :1], vel], dim=1)  # [2, 2000] pad first

        # Resample velocity to latent_length
        vel_resampled = F.interpolate(
            vel.unsqueeze(0), size=latent_length, mode='linear', align_corners=True
        ).squeeze(0)  # [2, 100]

        # Physical time normalized [0, 1], resampled to latent_length
        t_norm = (w_time - w_time[0]) / (w_time[-1] - w_time[0] + 1e-8)
        physical_time = F.interpolate(
            t_norm.view(1, 1, -1), size=latent_length, mode='linear', align_corners=True
        ).view(1, latent_length)  # [1, 100]

        windows.append({
            'imu': w_imu,
            'velocity': vel_resampled,
            'physical_time': physical_time,
        })

    return windows


def compute_stats(windows):
    """Compute per-channel mean/std over a list of window dicts."""
    all_imu = torch.stack([w['imu'] for w in windows])       # [W, 6, 2000]
    all_vel = torch.stack([w['velocity'] for w in windows])  # [W, 2, 100]

    imu_mean = all_imu.mean(dim=(0, 2))  # [6]
    imu_std = all_imu.std(dim=(0, 2)) + 1e-8
    vel_mean = all_vel.mean(dim=(0, 2))  # [2]
    vel_std = all_vel.std(dim=(0, 2)) + 1e-8

    return {
        'imu_mean': imu_mean,
        'imu_std': imu_std,
        'vel_mean': vel_mean,
        'vel_std': vel_std,
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess HDF5 IMU data into .pt windows')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--window_sec', type=float, default=10)
    parser.add_argument('--sample_rate', type=int, default=200)
    parser.add_argument('--latent_length', type=int, default=100)
    parser.add_argument('--val_fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    window_samples = int(args.window_sec * args.sample_rate)
    random.seed(args.seed)

    # Find HDF5 files
    input_dir = Path(args.input_dir)
    hdf5_files = sorted(list(input_dir.glob('*.hdf5')) + list(input_dir.glob('*.h5')))
    print(f'Found {len(hdf5_files)} HDF5 files')

    # Process each file
    file_windows = {}
    for fpath in hdf5_files:
        print(f'  Processing {fpath.name}...')
        windows = process_file(str(fpath), window_samples, args.latent_length)
        if windows:
            file_windows[fpath.name] = windows
            print(f'    -> {len(windows)} windows')

    # Split by file into train/val
    file_names = sorted(file_windows.keys())
    random.shuffle(file_names)
    n_val = max(1, int(len(file_names) * args.val_fraction))
    val_files = set(file_names[:n_val])
    train_files = set(file_names[n_val:])

    train_windows = [w for f in train_files for w in file_windows[f]]
    val_windows = [w for f in val_files for w in file_windows[f]]
    print(f'Split: {len(train_files)} train files ({len(train_windows)} windows), '
          f'{len(val_files)} val files ({len(val_windows)} windows)')

    # Compute stats on training set
    stats = compute_stats(train_windows)
    print(f'IMU mean: {stats["imu_mean"].numpy()}')
    print(f'IMU std:  {stats["imu_std"].numpy()}')
    print(f'Vel mean: {stats["vel_mean"].numpy()}')
    print(f'Vel std:  {stats["vel_std"].numpy()}')

    # Save
    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for i, w in enumerate(train_windows):
        torch.save(w, train_dir / f'window_{i:04d}.pt')
    for i, w in enumerate(val_windows):
        torch.save(w, val_dir / f'window_{i:04d}.pt')
    torch.save(stats, output_dir / 'stats.pt')

    print(f'Saved to {output_dir}')


if __name__ == '__main__':
    main()
