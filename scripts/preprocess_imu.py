"""Preprocess HDF5 IMU recordings into windowed .pt files for training.

Produces a dataset directory with train/, val/, stats.pt and split.txt, which
the training configs and every --dataset_dir / --vae_stats flag accept.

Example usage:
python scripts/preprocess_imu.py \
    --input_dir data/hdf5_data/dataset --outdir data/dataset_processed \
    --imu_frame world --window_sec 10 --sample_rate 200 --stride_sec 2.0
"""

import argparse
from pathlib import Path

from ldm.evaluation.sequences import process_file, write_window_dataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess HDF5 IMU recordings into .pt windows")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of .hdf5/.h5 recordings")
    parser.add_argument("--outdir", "--output_dir", dest="outdir", type=str,
                        required=True, help="Dataset directory to write")
    parser.add_argument("--imu_frame", type=str, default="world",
                        choices=["local", "world"],
                        help="Frame to store: world rotates via game_rv (what the "
                             "VAE and LDM are trained on), local keeps device frame")
    parser.add_argument("--window_sec", type=float, default=10)
    parser.add_argument("--sample_rate", type=int, default=200)
    parser.add_argument("--latent_length", type=int, default=100)
    parser.add_argument("--stride_sec", type=float, default=2.0,
                        help="Stride between windows in seconds "
                             "(default: 2.0, i.e. 80%% overlap with a 10s window)")
    parser.add_argument("--remove_gravity", action="store_true",
                        help="Use linear acceleration instead of raw accelerometer")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def window_recordings(input_dir, args):
    """Window every recording under ``input_dir``, keyed by filename."""
    window_samples = int(args.window_sec * args.sample_rate)
    stride_samples = int(args.stride_sec * args.sample_rate)

    paths = sorted(list(input_dir.glob("*.hdf5")) + list(input_dir.glob("*.h5")))
    print(f"Found {len(paths)} HDF5 files")

    windows_by_file = {}
    for path in paths:
        print(f"  Processing {path.name} ...")
        windows = process_file(
            str(path), window_samples, args.latent_length, args.imu_frame,
            stride=stride_samples, remove_gravity=args.remove_gravity,
        )
        if windows:
            windows_by_file[path.name] = windows
            print(f"    -> {len(windows)} windows")
    return windows_by_file


def main():
    args = build_parser().parse_args()
    windows_by_file = window_recordings(Path(args.input_dir), args)
    write_window_dataset(
        args.outdir, windows_by_file, args.val_fraction, args.seed,
        source_label="files",
    )


if __name__ == "__main__":
    main()
