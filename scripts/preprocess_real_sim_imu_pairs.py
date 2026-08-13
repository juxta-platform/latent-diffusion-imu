"""Preprocess aligned real/sim IMU pairs into windowed .pt files for training.

Reads manifest.csv from data/real_sim_imu_pairs/ and produces aligned windows
containing {imu, sim_imu, velocity, physical_time} for each pair.

Real IMU is rotated to world frame using game_rv (same as preprocess_imu.py).
Sim IMU uses the world-frame columns from the parquet directly.
Trajectory conditioning is derived from real HDF5 tango_pos (identical to existing pipeline).

Example usage:
python scripts/preprocess_real_sim_imu_pairs.py \
    --pairs_root data/real_sim_imu_pairs \
    --output_dir data/real_sim_pairs_processed \
    --window_sec 10 --stride_sec 2.0
"""

import argparse
import csv
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def load_real_hdf5(path, remove_gravity=False):
    """Load real IMU from HDF5 and rotate to world frame using game_rv."""
    with h5py.File(path, 'r') as f:
        acce = f['synced/acce'][:]
        gyro = f['synced/gyro'][:]
        pos = f['synced/tango_pos'][:]
        time = f['synced/time'][:]
        game_rv = f['synced/game_rv'][:]

        if remove_gravity:
            acce = f['synced/linacce'][:]

    # HDF5 stores (w, x, y, z); scipy expects (x, y, z, w)
    rot = Rotation.from_quat(game_rv[:, [1, 2, 3, 0]])
    acce = rot.apply(acce)
    gyro = rot.apply(gyro)

    return acce, gyro, pos, time


def load_sim_parquet(path):
    """Load sim IMU world-frame columns from parquet."""
    df = pd.read_parquet(path)
    accel = df[['accel_world_x', 'accel_world_y', 'accel_world_z']].values
    gyro = df[['gyro_world_x', 'gyro_world_y', 'gyro_world_z']].values
    return accel, gyro, len(df)


def process_pair(real_path, sim_path, window_samples, latent_length, stride,
                 remove_gravity=False):
    """Process one aligned pair into a list of window dicts."""
    real_acce, real_gyro, pos, time = load_real_hdf5(real_path, remove_gravity)
    sim_acce, sim_gyro, sim_len = load_sim_parquet(sim_path)

    # Use the minimum length to handle minor length mismatches
    n_samples = min(len(real_acce), sim_len)
    real_acce = real_acce[:n_samples]
    real_gyro = real_gyro[:n_samples]
    pos = pos[:n_samples]
    time = time[:n_samples]
    sim_acce = sim_acce[:n_samples]
    sim_gyro = sim_gyro[:n_samples]

    real_imu = np.concatenate([real_acce, real_gyro], axis=1)  # [N, 6]
    sim_imu = np.concatenate([sim_acce, sim_gyro], axis=1)      # [N, 6]
    pos_2d = pos[:, [0, 1]]  # [N, 2] (x, y)

    windows = []
    for s in range(0, n_samples - window_samples + 1, stride):
        e = s + window_samples

        w_real_imu = torch.tensor(real_imu[s:e].T, dtype=torch.float32)  # [6, 2000]
        w_sim_imu = torch.tensor(sim_imu[s:e].T, dtype=torch.float32)    # [6, 2000]
        w_pos = torch.tensor(pos_2d[s:e].T, dtype=torch.float32)          # [2, 2000]
        t_rel = time[s:e] - time[s]
        w_time = torch.tensor(t_rel, dtype=torch.float32)                  # [2000]

        # Velocity via forward difference
        dt = torch.tensor(np.diff(time[s:e]), dtype=torch.float32)  # [1999]
        dp = w_pos[:, 1:] - w_pos[:, :-1]  # [2, 1999]
        vel = dp / dt.unsqueeze(0).clamp_min(1e-8)  # [2, 1999]
        vel = torch.cat([vel[:, :1], vel], dim=1)  # [2, 2000] pad first

        # Resample velocity to latent_length
        vel_resampled = F.interpolate(
            vel.unsqueeze(0), size=latent_length, mode='linear', align_corners=True
        ).squeeze(0)  # [2, 100]

        # Physical time normalized [0, 1], resampled to latent_length
        t_norm = w_time / (w_time[-1] + 1e-8)
        physical_time = F.interpolate(
            t_norm.view(1, 1, -1), size=latent_length, mode='linear', align_corners=True
        ).view(1, latent_length)  # [1, 100]

        windows.append({
            'imu': w_real_imu,
            'sim_imu': w_sim_imu,
            'velocity': vel_resampled,
            'physical_time': physical_time,
        })

    return windows


def compute_stats(windows):
    """Compute per-channel mean/std from real IMU training windows."""
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
    parser = argparse.ArgumentParser(
        description='Preprocess aligned real/sim IMU pairs into .pt windows')
    parser.add_argument('--pairs_root', type=str, required=True,
                        help='Root directory containing manifest.csv and pair folders')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--window_sec', type=float, default=10)
    parser.add_argument('--sample_rate', type=int, default=200)
    parser.add_argument('--latent_length', type=int, default=100)
    parser.add_argument('--stride_sec', type=float, default=2.0)
    parser.add_argument('--remove_gravity', action='store_true',
                        help='Use linear acceleration (gravity removed) for real IMU')
    parser.add_argument('--val_fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    window_samples = int(args.window_sec * args.sample_rate)
    stride_samples = int(args.stride_sec * args.sample_rate)
    random.seed(args.seed)

    pairs_root = Path(args.pairs_root)
    manifest_path = pairs_root / 'manifest.csv'

    # Read manifest
    pairs = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] != 'generated':
                continue
            pairs.append(row)
    print(f'Found {len(pairs)} generated pairs in manifest')

    # Manifest paths are relative to the repo root (parent of data/)
    repo_root = pairs_root.parent.parent

    # Process each pair
    pair_windows = {}
    for row in pairs:
        pair_id = row['pair_id']
        real_path = repo_root / row['real_hdf5']
        sim_path = repo_root / row['synthetic_parquet']

        if not real_path.exists():
            print(f'  SKIP {pair_id}: real HDF5 not found at {real_path}')
            continue
        if not sim_path.exists():
            print(f'  SKIP {pair_id}: synthetic parquet not found at {sim_path}')
            continue

        print(f'  Processing {pair_id}...')
        windows = process_pair(str(real_path), str(sim_path),
                               window_samples, args.latent_length, stride_samples,
                               remove_gravity=args.remove_gravity)
        if windows:
            pair_windows[pair_id] = windows
            print(f'    -> {len(windows)} windows')
        else:
            print(f'    -> 0 windows (recording too short)')

    if not pair_windows:
        print('ERROR: No windows produced. Check paths and recording lengths.')
        return

    # Split by pair_id (file-level) into train/val
    pair_ids = sorted(pair_windows.keys())
    random.shuffle(pair_ids)
    if len(pair_ids) == 1:
        train_pairs = set(pair_ids)
        val_pairs = set(pair_ids)
    else:
        n_val = max(1, int(len(pair_ids) * args.val_fraction))
        n_val = min(n_val, len(pair_ids) - 1)
        val_pairs = set(pair_ids[:n_val])
        train_pairs = set(pair_ids[n_val:])

    train_windows = [w for pid in train_pairs for w in pair_windows[pid]]
    val_windows = [w for pid in val_pairs for w in pair_windows[pid]]
    print(f'Split: {len(train_pairs)} train pairs ({len(train_windows)} windows), '
          f'{len(val_pairs)} val pairs ({len(val_windows)} windows)')

    # Compute stats on training set (from real IMU only)
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

    # Write split manifest
    split_path = output_dir / 'split.txt'
    with open(split_path, 'w') as f:
        f.write(f'# seed={args.seed} val_fraction={args.val_fraction}\n')
        f.write(f'# {len(train_pairs)} train pairs, {len(val_pairs)} val pairs\n')
        f.write(f'\n[train]\n')
        for pid in sorted(train_pairs):
            f.write(f'{pid}\n')
        f.write(f'\n[val]\n')
        for pid in sorted(val_pairs):
            f.write(f'{pid}\n')
    print(f'Wrote split list to {split_path}')
    print(f'Saved to {output_dir}')


if __name__ == '__main__':
    main()
