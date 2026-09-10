"""Dataset for carrying/activity classification from HDF5 IMU recordings.

Walks ``data_dir/<class>/*.hdf5`` for each class the classifier predicts,
windows each recording into overlapping 10s windows (via
evaluation.sequences.process_file), and overrides folder labels with stationary
when that activity dominates.

Only CARRYING_CLASS_NAMES folders are read, on purpose: a dataset may hold
placements this classifier is not being trained on (see
ldm.evaluation.constants.CARRYING_PLACEMENTS), and those are skipped here
rather than silently folded into another class.
"""

import os
import random

from torch.utils.data import Dataset

from ldm.data.carrying_labels import (
    CARRYING_CLASS_NAMES, CLASS_NAMES, derive_window_labels,
)
from ldm.evaluation.sequences import check_imu_frame, load_activity_imu, load_hdf5, process_file


def build_file_list(data_dir):
    """Return list of (hdf5_path, class_name) sorted deterministically."""
    entries = []
    for cls_name in CARRYING_CLASS_NAMES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.endswith((".hdf5", ".h5")):
                entries.append((os.path.join(cls_dir, fname), cls_name))
    return entries


def split_files_stratified(file_list, val_fraction=0.2, seed=42):
    """Split file_list into train/val by file, stratified per class."""
    rng = random.Random(seed)
    by_class = {}
    for path, cls_name in file_list:
        by_class.setdefault(cls_name, []).append((path, cls_name))

    train_files, val_files = [], []
    for cls_name in CARRYING_CLASS_NAMES:
        files = by_class.get(cls_name, [])
        shuffled = list(files)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0
        val_files.extend(shuffled[:n_val])
        train_files.extend(shuffled[n_val:])

    return train_files, val_files


class CarryingTypeDataset(Dataset):
    """IMU windows labeled with carrying type or overriding activity."""

    def __init__(self, file_list, imu_mean, imu_std, window_sec=10,
                 sample_rate=200, latent_length=100, stride_sec=2.0,
                 imu_frame="world", activity_threshold=0.5,
                 class_names=CLASS_NAMES):
        """
        Args:
            file_list: list of (hdf5_path, class_name).
            imu_mean, imu_std: [C] or [C,1] tensors for standardization.
            stride_sec: stride between overlapping windows in seconds.
            imu_frame: 'local' keeps device-frame IMU; 'world' rotates to
                global frame via game_rv (same as preprocess_imu default).
            activity_threshold: fraction of a window that must be exceeded for
                stationary to override its carrying label.
        """
        check_imu_frame(imu_frame)
        self.imu_mean = imu_mean.view(-1, 1)
        self.imu_std = imu_std.view(-1, 1)
        self.imu_frame = imu_frame
        self.activity_threshold = activity_threshold
        self.class_names = list(class_names)
        class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        window_samples = int(window_sec * sample_rate)
        stride_samples = int(stride_sec * sample_rate)

        self.windows = []
        self.labels = []
        self.files = []

        for hdf5_path, cls_name in file_list:
            wins = process_file(
                hdf5_path, window_samples, latent_length, imu_frame,
                stride=stride_samples,
            )
            if "stationary" in self.class_names:
                # Stationary is decided on device-frame motion regardless of
                # the frame the classifier itself consumes.
                recording = load_hdf5(hdf5_path, imu_frame=imu_frame)
                activity_imu = load_activity_imu(hdf5_path, recording.imu, imu_frame)
                window_labels, _ = derive_window_labels(
                    activity_imu, recording.pos, cls_name, len(wins), window_samples,
                    stride_samples, sample_rate=sample_rate,
                    activity_threshold=activity_threshold,
                )
            else:
                window_labels = [cls_name] * len(wins)
            for w, window_label in zip(wins, window_labels):
                self.windows.append(w)
                self.labels.append(class_to_idx[window_label])
                self.files.append(os.path.basename(hdf5_path))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data = self.windows[idx]
        imu = (data['imu'] - self.imu_mean) / self.imu_std
        return {
            'imu': imu,           # [6, 2000]
            'label': self.labels[idx],
            'file': self.files[idx],
        }
