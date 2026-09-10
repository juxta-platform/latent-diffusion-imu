"""Preprocess aligned real/sim IMU pairs into windowed .pt files for training.

Reads manifest.csv from a pairs root and produces aligned windows containing
{imu, sim_imu, velocity, physical_time} for each pair, the input the sim_cond
LDM trains on.

Both IMUs are world frame: the real one is rotated via game_rv, the sim one
comes from the parquet's world columns. Trajectory conditioning is derived from
the real HDF5's tango_pos, exactly as in preprocess_imu.py.

Example usage:
python scripts/preprocess_real_sim_imu_pairs.py \
    --pairs_root data/real_sim_paired_data_ldm \
    --outdir data/real_sim_pairs_processed \
    --window_sec 10 --stride_sec 2.0
"""

import argparse
import csv
from pathlib import Path

from ldm.evaluation.constants import MANIFEST_NAME
from ldm.evaluation.sequences import process_pair, write_window_dataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess aligned real/sim IMU pairs into .pt windows")
    parser.add_argument("--pairs_root", type=str, required=True,
                        help=f"Directory containing {MANIFEST_NAME} and pair folders")
    parser.add_argument("--outdir", "--output_dir", dest="outdir", type=str,
                        required=True, help="Dataset directory to write")
    parser.add_argument("--window_sec", type=float, default=10)
    parser.add_argument("--sample_rate", type=int, default=200)
    parser.add_argument("--latent_length", type=int, default=100)
    parser.add_argument("--stride_sec", type=float, default=2.0)
    parser.add_argument("--remove_gravity", action="store_true",
                        help="Use linear acceleration for the real IMU")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def read_manifest(pairs_root):
    """Generated pairs from manifest.csv as (pair_id, real_path, sim_path).

    Manifest paths are relative to the repo root, two levels above the pairs
    root (``<repo>/data/<pairs_root>``).
    """
    manifest_path = pairs_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {pairs_root}")
    repo_root = pairs_root.parent.parent

    with open(manifest_path) as f:
        rows = [row for row in csv.DictReader(f) if row["status"] == "generated"]
    print(f"Found {len(rows)} generated pairs in manifest")
    return [
        (row["pair_id"],
         repo_root / row["real_hdf5"],
         repo_root / row["synthetic_parquet"])
        for row in rows
    ]


def window_pairs(pairs, args):
    """Window every pair, keyed by pair_id, skipping any with missing files."""
    window_samples = int(args.window_sec * args.sample_rate)
    stride_samples = int(args.stride_sec * args.sample_rate)

    windows_by_pair = {}
    for pair_id, real_path, sim_path in pairs:
        for label, path in (("real HDF5", real_path), ("synthetic parquet", sim_path)):
            if not path.exists():
                print(f"  SKIP {pair_id}: {label} not found at {path}")
                break
        else:
            print(f"  Processing {pair_id} ...")
            windows = process_pair(
                str(real_path), str(sim_path), window_samples, args.latent_length,
                stride=stride_samples, remove_gravity=args.remove_gravity,
            )
            print(f"    -> {len(windows)} windows"
                  + ("" if windows else " (recording too short)"))
            if windows:
                windows_by_pair[pair_id] = windows
    return windows_by_pair


def main():
    args = build_parser().parse_args()
    pairs = read_manifest(Path(args.pairs_root))
    windows_by_pair = window_pairs(pairs, args)
    write_window_dataset(
        args.outdir, windows_by_pair, args.val_fraction, args.seed,
        source_label="pairs",
    )


if __name__ == "__main__":
    main()
