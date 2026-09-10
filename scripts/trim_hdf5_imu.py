"""Manually trim gyro-calibration periods from an HDF5 recording.

Plots synced accel/gyro, then trims the start/end of all synced arrays and
overwrites the file. Times are relative seconds from the recording start.

Example (interactive — plot then prompt for times):
  python scripts/trim_hdf5_imu.py \\
      --input data/real_data_by_carrying_type/pocket/john_left_pocket_ios_corrected.hdf5

Example (non-interactive):
  python scripts/trim_hdf5_imu.py \\
      --input data/real_data_by_carrying_type/pocket/john_left_pocket_ios_corrected.hdf5 \\
      --t_start 5.0 --t_end 130.0

Example (preview only, no write):
  python scripts/trim_hdf5_imu.py --input ... --t_start 5 --t_end 130 --dry_run
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ldm.evaluation.constants import CHANNEL_NAMES


def load_synced_imu(path):
    with h5py.File(path, "r") as f:
        if "synced" not in f:
            raise ValueError(f"No 'synced' group in {path}")
        g = f["synced"]
        for key in ("acce", "gyro", "time"):
            if key not in g:
                raise ValueError(f"Missing synced/{key} in {path}")
        acce = g["acce"][:]
        gyro = g["gyro"][:]
        time = g["time"][:].astype(np.float64)
        keys = list(g.keys())
    return acce, gyro, time, keys


def plot_imu(acce, gyro, time, out_path, t_start=None, t_end=None, title=None):
    t_rel = time - time[0]
    imu = np.concatenate([acce, gyro], axis=1)  # [N, 6]

    fig, axes = plt.subplots(6, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    for c, ax in enumerate(axes):
        ax.plot(t_rel, imu[:, c], linewidth=0.7, alpha=0.9)
        ax.set_ylabel(CHANNEL_NAMES[c], fontsize=9)
        ax.grid(True, alpha=0.25)
        if t_start is not None:
            ax.axvline(t_start, color="C2", linestyle="--", linewidth=1.2, label="trim start")
        if t_end is not None:
            ax.axvline(t_end, color="C3", linestyle="--", linewidth=1.2, label="trim end")
        if c == 0 and (t_start is not None or t_end is not None):
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s) from recording start")
    fig.suptitle(title or "Synced IMU (select trim region)", fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved IMU plot to {out_path}")


def prompt_times(duration):
    print(f"\nRecording duration: {duration:.2f} s")
    print("Enter trim times in seconds from start (keep the middle segment).")
    print("Leave blank for no trim on that side.")

    def parse(prompt, default):
        raw = input(prompt).strip()
        if raw == "":
            return default
        return float(raw)

    t_start = parse(f"  trim start [0]: ", 0.0)
    t_end = parse(f"  trim end [{duration:.2f}]: ", duration)
    return t_start, t_end


def find_trim_indices(time, t_start, t_end):
    t_rel = time - time[0]
    duration = float(t_rel[-1])
    if t_start < 0:
        raise ValueError(f"t_start={t_start} must be >= 0")
    if t_end > duration + 1e-6:
        raise ValueError(f"t_end={t_end} exceeds duration {duration:.3f}s")
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")

    i0 = int(np.searchsorted(t_rel, t_start, side="left"))
    i1 = int(np.searchsorted(t_rel, t_end, side="right"))
    if i1 - i0 < 2:
        raise ValueError(f"Trim window too short: indices [{i0}:{i1}]")
    return i0, i1


def trim_synced_group(path, i0, i1, backup=True, dry_run=False):
    """Overwrite synced/* arrays with the trimmed slice [i0:i1]."""
    with h5py.File(path, "r") as f:
        g = f["synced"]
        datasets = {}
        for key in g.keys():
            obj = g[key]
            if isinstance(obj, h5py.Dataset) and obj.shape[0] > 0:
                datasets[key] = obj[:]

    n_before = None
    for key, arr in datasets.items():
        if n_before is None:
            n_before = arr.shape[0]
        elif arr.shape[0] != n_before:
            print(f"  [warn] synced/{key} length {arr.shape[0]} != {n_before}; "
                  f"skipping (not time-aligned)")
            del datasets[key]
            continue

    n_after = i1 - i0
    print(f"  Trimming synced arrays: {n_before} -> {n_after} samples "
          f"(drop {i0} head, {n_before - i1} tail)")

    if dry_run:
        print("  [dry_run] No file written")
        return

    if backup:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print(f"  Backup written to {bak}")
        else:
            print(f"  Backup already exists: {bak}")

    # Rewrite synced datasets in place via temp file for safety
    tmp_path = path + ".tmp"
    with h5py.File(path, "r") as src, h5py.File(tmp_path, "w") as dst:
        # Copy everything except synced datasets we trim
        def copy_item(name, obj):
            if name.startswith("synced/") and name.split("/", 1)[1] in datasets:
                return  # handled below
            if isinstance(obj, h5py.Group):
                if name not in dst:
                    dst.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                parent = "/".join(name.split("/")[:-1])
                if parent and parent not in dst:
                    dst.require_group(parent)
                dst.create_dataset(name, data=obj[:],
                                   compression=obj.compression,
                                   compression_opts=obj.compression_opts)

        src.visititems(copy_item)

        if "synced" not in dst:
            dst.create_group("synced")
        for key, arr in datasets.items():
            trimmed = arr[i0:i1]
            dst.create_dataset(f"synced/{key}", data=trimmed)

        # Copy root attrs
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        if "synced" in src:
            for k, v in src["synced"].attrs.items():
                dst["synced"].attrs[k] = v

    os.replace(tmp_path, path)
    print(f"  Overwrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Trim gyro-calibration periods from an HDF5 IMU recording"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to .hdf5 / .h5 file")
    parser.add_argument("--t_start", type=float, default=None,
                        help="Keep data starting at this relative time (seconds)")
    parser.add_argument("--t_end", type=float, default=None,
                        help="Keep data ending at this relative time (seconds)")
    parser.add_argument("--plot", type=str, default=None,
                        help="Path to save IMU plot (default: <input>_trim_preview.png)")
    parser.add_argument("--no_backup", action="store_true",
                        help="Do not write a .bak copy before overwriting")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show trim plan and plot, but do not write")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt before overwrite")
    args = parser.parse_args()

    path = args.input
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    acce, gyro, time, keys = load_synced_imu(path)
    duration = float(time[-1] - time[0])
    print(f"Loaded {path}")
    print(f"  synced keys: {keys}")
    print(f"  n={len(time)}, duration={duration:.2f}s")

    plot_path = args.plot or str(Path(path).with_suffix("")) + "_trim_preview.png"
    plot_imu(acce, gyro, time, plot_path, title=f"IMU — {os.path.basename(path)}")

    t_start = args.t_start
    t_end = args.t_end
    if t_start is None or t_end is None:
        ts, te = prompt_times(duration)
        if t_start is None:
            t_start = ts
        if t_end is None:
            t_end = te

    i0, i1 = find_trim_indices(time, t_start, t_end)
    t_rel = time - time[0]
    print(f"\nTrim plan: keep [{t_start:.3f}, {t_end:.3f}] s "
          f"-> samples [{i0}:{i1}] "
          f"(actual [{t_rel[i0]:.3f}, {t_rel[i1-1]:.3f}] s)")

    # Re-plot with trim markers
    marked = str(Path(plot_path).with_suffix("")) + "_marked.png"
    plot_imu(acce, gyro, time, marked, t_start=t_start, t_end=t_end,
             title=f"IMU trim preview — {os.path.basename(path)}")

    if args.dry_run:
        trim_synced_group(path, i0, i1, backup=False, dry_run=True)
        return

    if not args.yes:
        ans = input(f"Overwrite {path}? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    trim_synced_group(path, i0, i1, backup=not args.no_backup, dry_run=False)

    # Verify + plot trimmed result
    acce2, gyro2, time2, _ = load_synced_imu(path)
    out_plot = str(Path(path).with_suffix("")) + "_trimmed.png"
    plot_imu(acce2, gyro2, time2, out_plot,
             title=f"Trimmed IMU — {os.path.basename(path)}")
    print(f"Done. New duration: {time2[-1] - time2[0]:.2f}s ({len(time2)} samples)")


if __name__ == "__main__":
    main()
