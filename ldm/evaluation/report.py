"""Writers for metrics, trajectories, generated recordings and split lists."""

import csv
import json
import os
import os.path as osp
import random
import shutil
from pathlib import Path

import numpy as np

from ldm.evaluation.constants import ACCEL_WORLD_COLS, GYRO_WORLD_COLS, HDF5_EXTS


# ---------------------------------------------------------------------------
# Metrics and trajectories
# ---------------------------------------------------------------------------

def write_json(payload, outdir, filename, what=None, echo=False):
    """Write a JSON document under ``outdir`` and return its path."""
    os.makedirs(outdir, exist_ok=True)
    path = osp.join(outdir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    if echo:
        print(json.dumps(payload, indent=2))
    if what:
        print(f"{what} saved to {path}")
    return path


def read_json(path):
    with open(path) as f:
        return json.load(f)


def save_plot_data(outdir, **arrays):
    """Numeric plot inputs, so --plots_only needs neither models nor raw data."""
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(osp.join(outdir, "plot_data.npz"),
                        **{k: np.asarray(v) for k, v in arrays.items() if v is not None})


def save_signal_plot_data(outdir, inputs, outputs, inputs_phys, outputs_phys, n_plot):
    limit = len(inputs) if n_plot < 0 else min(len(inputs), max(8, n_plot))
    save_plot_data(outdir, inputs=inputs[:limit], outputs=outputs[:limit],
                   inputs_phys=inputs_phys[:limit], outputs_phys=outputs_phys[:limit])


def load_plot_data(outdir):
    path = osp.join(outdir, "plot_data.npz")
    if not osp.isfile(path):
        raise FileNotFoundError(f"{path} is missing; re-run evaluation once to save plot inputs")
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def replot_signals(outdir, result, n_plot):
    from ldm.evaluation import plots

    data = load_plot_data(outdir)
    sample_rate = result.get("sample_rate", 200)
    for suffix, dest in (("", outdir), ("_phys", osp.join(outdir, "physical"))):
        plots.plot_signal_overlays(
            data["inputs" + suffix], data["outputs" + suffix], dest, n_plot,
            sample_rate=sample_rate, title_prefix=result.get("source", "IMU evaluation"),
        )
    if "original" in result:
        path = osp.join(outdir, "trajectories.npz")
        with np.load(path, allow_pickle=False) as trajectories:
            other_key = "vae_recon" if "vae_recon" in result else "ldm_gen"
            original = dict(result["original"], pos_gt=trajectories["pos_gt"],
                            pos_pred=trajectories["pos_pred_orig"])
            other = dict(result[other_key], pos_gt=trajectories["pos_gt"],
                         pos_pred=trajectories["pos_pred_gen"])
            plots.plot_trajectories(original, other, outdir, other_label=other_key,
                                    rte_delta_sec=result.get("rte_delta_sec", 10))
            plots.plot_position_error(original, other, outdir, other_label=other_key)


def write_trajectories(outdir, pos_gt, pos_pred_orig, pos_pred_gen,
                       filename="trajectories.npz"):
    """Save trajectories so a downstream aggregation can re-plot without re-running."""
    os.makedirs(outdir, exist_ok=True)
    path = osp.join(outdir, filename)
    np.savez_compressed(
        path, pos_gt=pos_gt, pos_pred_orig=pos_pred_orig, pos_pred_gen=pos_pred_gen,
    )
    print(f"Trajectories saved to {path}")
    return path


def write_csv(rows, header, outdir, filename):
    os.makedirs(outdir, exist_ok=True)
    path = osp.join(outdir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Generated IMU exports
# ---------------------------------------------------------------------------

def write_generated_hdf5(src_path, out_path, features_ronin, already_world=False):
    """Copy an HDF5 recording, replacing ``synced/acce`` and ``synced/gyro``.

    ``features_ronin`` is world-frame [N, 6] in RoNIN order ``[gyro, accel]``
    and is stored as-is (no local-frame rotation). Every other dataset,
    ``linacce`` included, is left alone, and samples past N keep the original
    IMU rotated to world frame.
    """
    import h5py
    from ldm.evaluation.sequences import load_hdf5

    src_path = src_path if src_path.endswith(HDF5_EXTS) else src_path + ".hdf5"
    n_gen = features_ronin.shape[0]
    source_world = load_hdf5(src_path, "world", already_world=already_world).imu

    os.makedirs(osp.dirname(osp.abspath(out_path)), exist_ok=True)
    shutil.copy2(src_path, out_path)
    with h5py.File(out_path, "a") as f:
        if "synced" not in f or "acce" not in f["synced"] or "gyro" not in f["synced"]:
            raise ValueError(f"{src_path} missing synced/acce or synced/gyro")
        n_file = f["synced/acce"].shape[0]
        n_write = min(n_gen, n_file)
        if n_write < n_gen:
            print(f"  [warn] Truncating generated IMU {n_gen} -> {n_write} to fit HDF5 length")
        # A shortened generated span must not leave a local-frame tail in a
        # file marked world-frame.
        f["synced/gyro"][:] = source_world[:, 3:]
        f["synced/acce"][:] = source_world[:, :3]
        f["synced/gyro"][:n_write] = features_ronin[:n_write, :3]
        f["synced/acce"][:n_write] = features_ronin[:n_write, 3:6]
        # Mark the frame so readers do not rotate this IMU by game_rv again.
        f.attrs["imu_frame"] = "world"
        f.attrs["imu_channel_order"] = "acce_xyz,gyro_xyz"

    print(f"Wrote generated world-frame IMU HDF5 to {out_path} "
          f"({n_write}/{n_file} samples replaced)")
    return out_path


def write_generated_parquet(src_path, out_path, features_ronin, gen_ts):
    """Copy a parquet, replacing its world-frame IMU columns with generated IMU.

    ``features_ronin`` is [N, 6] in RoNIN order ``[gyro, accel]`` sampled on
    ``gen_ts``; values are interpolated onto the parquet's (deduplicated) time
    grid when the grids differ, e.g. after a 200 Hz resample. Local IMU columns
    and every other field are left unchanged.
    """
    import pandas as pd

    src_path = src_path if src_path.endswith(".parquet") else src_path + ".parquet"
    df = pd.read_parquet(src_path)
    df = df.drop_duplicates(subset="time", keep="first").reset_index(drop=True)
    t_df = df["time"].values.astype(np.float64)

    n_gen = features_ronin.shape[0]
    gen_ts = np.asarray(gen_ts, dtype=np.float64)[:n_gen]
    gyro_gen = features_ronin[:n_gen, :3]
    accel_gen = features_ronin[:n_gen, 3:6]

    same_grid = len(gen_ts) == len(t_df) and np.allclose(gen_ts, t_df, atol=1e-4, rtol=0.0)
    if same_grid:
        gyro_out, accel_out = gyro_gen, accel_gen
    else:
        print(f"  [info] Interpolating generated IMU ({n_gen} @ gen_ts) onto "
              f"parquet time grid ({len(t_df)} samples)")
        gyro_out = np.column_stack([
            np.interp(t_df, gen_ts, gyro_gen[:, d], left=np.nan, right=np.nan)
            for d in range(3)
        ])
        accel_out = np.column_stack([
            np.interp(t_df, gen_ts, accel_gen[:, d], left=np.nan, right=np.nan)
            for d in range(3)
        ])
        # Keep the original IMU outside the generated span
        mask = np.isfinite(gyro_out[:, 0])
        for out, cols in ((gyro_out, GYRO_WORLD_COLS), (accel_out, ACCEL_WORLD_COLS)):
            for d, col in enumerate(cols):
                values = df[col].values.astype(np.float64)
                values[mask] = out[mask, d]
                out[:, d] = values

    df = df.copy()
    for d, col in enumerate(GYRO_WORLD_COLS):
        df[col] = gyro_out[:, d]
    for d, col in enumerate(ACCEL_WORLD_COLS):
        df[col] = accel_out[:, d]
    os.makedirs(osp.dirname(osp.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote generated world-frame IMU parquet to {out_path} "
          f"({n_gen} gen samples -> {len(df)} rows)")
    return out_path


def write_gen_world_hdf5(path, time, position, orientation, generated, metadata):
    """Write a fresh RoNIN ``gen_world`` HDF5 from generated world-frame IMU.

    ``generated`` is [N, 6] in ``[accel, gyro]`` order. Written to a temporary
    file and renamed, so an interrupted run never leaves a half-written dataset.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp_path, "w") as f:
        synced = f.create_group("synced")
        synced.create_dataset("time", data=time, compression="gzip")
        synced.create_dataset("tango_pos", data=position, compression="gzip")
        synced.create_dataset("game_rv", data=orientation, compression="gzip")
        synced.create_dataset("acce", data=generated[:, :3], compression="gzip")
        synced.create_dataset("gyro", data=generated[:, 3:], compression="gzip")
        f.attrs["imu_frame"] = "world"
        f.attrs["imu_channel_order"] = "acce_xyz,gyro_xyz"
        f.attrs["generator"] = metadata.get("generator", "LDM")
        f.attrs["generation_metadata"] = json.dumps(metadata)
    os.replace(tmp_path, path)
    return path


def gen_world_is_valid(path):
    """Whether a gen_world HDF5 has equal-length, non-empty synced datasets."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            lengths = [
                len(f[name]) for name in (
                    "synced/time", "synced/tango_pos", "synced/game_rv",
                    "synced/acce", "synced/gyro",
                )
            ]
        return min(lengths) > 0 and len(set(lengths)) == 1
    except (OSError, KeyError):
        return False


def write_lists(outdir, stems, val_fraction, seed):
    """Write ``all.txt`` / ``train.txt`` / ``val.txt`` for a generated dataset.

    The split is a deterministic shuffle by seed, and always leaves at least one
    sequence on each side whenever ``val_fraction`` is non-zero.
    """
    outdir = Path(outdir)
    stems = sorted(stems)
    shuffled = stems.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    if val_fraction > 0 and len(shuffled) > 1:
        n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_set = set(shuffled[:n_val])
    train = [stem for stem in stems if stem not in val_set]
    val = [stem for stem in stems if stem in val_set]

    for name, values in (("all.txt", stems), ("train.txt", train), ("val.txt", val)):
        with (outdir / name).open("w") as f:
            f.write("\n".join(values))
            if values:
                f.write("\n")
    return len(train), len(val)
