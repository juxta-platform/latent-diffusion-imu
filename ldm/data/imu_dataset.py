"""PyTorch Dataset and Lightning DataModule for preprocessed IMU data."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class IMUDataset(Dataset):
    """Dataset for preprocessed IMU windows."""

    def __init__(self, data_dir, stats_path, split='train'):
        """
        Args:
            data_dir: path to output_dir from preprocessing
            stats_path: path to stats.pt (usually data_dir/stats.pt)
            split: 'train' or 'val'
        """
        self.split_dir = os.path.join(data_dir, split)
        self.stats = torch.load(stats_path, weights_only=True)

        self.files = sorted([
            os.path.join(self.split_dir, f)
            for f in os.listdir(self.split_dir)
            if f.endswith('.pt')
        ])

        self.imu_mean = self.stats['imu_mean'].view(6, 1)
        self.imu_std = self.stats['imu_std'].view(6, 1)
        self.vel_mean = self.stats['vel_mean'].view(2, 1)
        self.vel_std = self.stats['vel_std'].view(2, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)

        imu = (data['imu'] - self.imu_mean) / self.imu_std
        velocity = (data['velocity'] - self.vel_mean) / self.vel_std
        physical_time = data['physical_time']

        return {
            'imu': imu,                    # [6, 2000]
            'velocity': velocity,          # [2, 100]
            'physical_time': physical_time  # [1, 100]
        }

    def inverse_standardize_imu(self, imu):
        """Convert standardized IMU back to physical units."""
        return imu * self.imu_std.to(imu.device) + self.imu_mean.to(imu.device)


class IMUDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stats_path = os.path.join(data_dir, 'stats.pt')

    def setup(self, stage=None):
        self.train_dataset = IMUDataset(self.data_dir, self.stats_path, split='train')
        self.val_dataset = IMUDataset(self.data_dir, self.stats_path, split='val')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class SyntheticIMUDataset(Dataset):
    """Dataset for synthetic IMU from parquet files."""

    def __init__(self, parquet_path, stats_path, window_sec=10, sample_rate=200, latent_length=100):
        """Load synthetic parquet data, slice into windows."""
        import pandas as pd

        self.stats = torch.load(stats_path, weights_only=True)
        self.imu_mean = self.stats['imu_mean'].view(6, 1)
        self.imu_std = self.stats['imu_std'].view(6, 1)
        self.vel_mean = self.stats['vel_mean'].view(2, 1)
        self.vel_std = self.stats['vel_std'].view(2, 1)

        df = pd.read_parquet(parquet_path)
        window_samples = window_sec * sample_rate

        accel = df[['accel_local_x', 'accel_local_y', 'accel_local_z']].values
        gyro = df[['gyro_local_x', 'gyro_local_y', 'gyro_local_z']].values
        imu = np.concatenate([accel, gyro], axis=1)  # [N, 6]

        vel = df[['agent_vel_x', 'agent_vel_z']].values  # [N, 2]
        time = df['time'].values  # [N]

        self.windows = []
        n_windows = len(df) // window_samples
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples

            w_imu = torch.tensor(imu[start:end].T, dtype=torch.float32)   # [6, 2000]
            w_vel = torch.tensor(vel[start:end].T, dtype=torch.float32)   # [2, 2000]
            w_time = torch.tensor(time[start:end], dtype=torch.float32)   # [2000]

            # Resample velocity to latent_length
            w_vel_resampled = F.interpolate(
                w_vel.unsqueeze(0), size=latent_length, mode='linear', align_corners=True
            ).squeeze(0)  # [2, 100]

            # Physical time normalized [0, 1]
            t_norm = (w_time - w_time[0]) / (w_time[-1] - w_time[0] + 1e-8)
            physical_time = F.interpolate(
                t_norm.view(1, 1, -1), size=latent_length, mode='linear', align_corners=True
            ).squeeze(0)  # [1, 100]

            self.windows.append({
                'imu': w_imu,
                'velocity': w_vel_resampled,
                'physical_time': physical_time,
            })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data = self.windows[idx]
        imu = (data['imu'] - self.imu_mean) / self.imu_std
        velocity = (data['velocity'] - self.vel_mean) / self.vel_std
        return {
            'imu': imu,
            'velocity': velocity,
            'physical_time': data['physical_time'],
            'imu_raw': data['imu'],
        }
