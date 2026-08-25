"""Evaluate the carrying-type classifier.

Pick exactly one input:
  --data_dir   labeled train/val split, overlapping windows as in training.
               Prints accuracy, a classification report and a confusion matrix.
  --input      one .hdf5 / .parquet, or a pair dir holding real.hdf5. Uses
               consecutive (non-overlapping) windows for a chronological
               readout: per-window IMU plots plus a timeline summary.
  --input_dir  a folder of trajectories, or of pair dirs. Runs the single
               trajectory eval on each, into per-file subdirs plus summary.json.

Options that layer on top of --input / --input_dir:
  --ldm_ckpt         generate IMU with the LDM before classifying, either
                     trajectory- or sim-conditioned (--ldm_mode).
  --n_noise N        sample N independent LDM noise vectors per trajectory and
                     classify each, adding a per-noise-init comparison plot.
  --compare_sources  classify the real recording, its synthetic.parquet and one
                     LDM sample of the same trajectory side by side, with
                     overlay window plots and a source_comparison.png. Over
                     --input_dir it walks every pair dir and aggregates
                     agreement / accuracy into summary.json.

IMU frames: HDF5 rotates to world via game_rv unless --hdf5_already_world (for
gen_world exports whose synced/acce|gyro are already world). Parquet reads
*_local_* or *_world_* columns; --parquet_from_world rotates world->local via
phone_rot, for LDM output whose local columns are stale. LDM and simulator IMU
are world-frame, so those runs want --imu_frame world.

--input_dir and --compare_sources write a per-window confusion matrix against
the carrying type named by the path (the <carrying_type>/<sequence>/ folder of a
pair dataset, or the <carrying_type>_<sequence>.hdf5 prefix of a gen_world
export); --compare_sources writes one per source. Recordings open and close with
a gyro calibration (phone held vertical, swayed side to side on the spot) and
contain incidental standing pauses, none of which show how the phone is carried,
so those windows are left out of the matrices. Everything else -- timelines,
window plots, majority votes, agreement metrics -- still covers the whole
recording.

Example usage:
  python scripts/eval_carrying_classifier.py \
      --config configs/imu/vae_1d.yaml \
      --vae_ckpt logs/vae_1d/.../total_loss=0.0116.ckpt \
      --clf_ckpt logs/carrying_classifier/best_classifier.pt \
      --data_dir data/real_data_by_carrying_type --split val \
      --outdir outputs/carrying_eval

  python scripts/eval_carrying_classifier.py \
      --config configs/imu/vae_1d.yaml \
      --vae_ckpt logs/vae_1d/.../total_loss=0.0116.ckpt \
      --clf_ckpt logs/carrying_classifier/best_classifier.pt \
      --input data/real_data_by_carrying_type/pocket/john_left_pocket_ios_corrected.hdf5 \
      --outdir outputs/carrying_eval_single

  python scripts/eval_carrying_classifier.py \
      --config configs/imu/vae_1d.yaml \
      --vae_ckpt logs/vae_1d/.../total_loss=0.0116.ckpt \
      --clf_ckpt logs/carrying_classifier/best_classifier.pt \
      --input_dir data/ronin_ldm_gen_world \
      --imu_frame world --hdf5_already_world \
      --outdir outputs/carrying_eval_gen_world

  python scripts/eval_carrying_classifier.py \
      --config configs/imu/vae_1d.yaml \
      --vae_ckpt logs/vae_1d/.../best.ckpt \
      --clf_ckpt logs/carrying_classifier/best_classifier.pt \
      --ldm_ckpt logs/ldm_1d_sim_cond/.../last.ckpt \
      --ldm_config configs/imu/ldm_1d_sim_cond.yaml \
      --ldm_stats data/real_sim_pairs_processed/stats.pt \
      --ldm_mode sim_cond --n_noise 8 \
      --input data/real_sim_imu_pairs/.../a000_4 \
      --imu_frame world \
      --outdir outputs/carrying_eval_noise_sweep

  python scripts/eval_carrying_classifier.py \
      --config configs/imu/vae_1d.yaml \
      --vae_ckpt logs/vae_1d/.../best.ckpt \
      --clf_ckpt logs/carrying_classifier/best_classifier.pt \
      --ldm_ckpt logs/ldm_1d_sim_cond/.../last.ckpt \
      --ldm_config configs/imu/ldm_1d_sim_cond.yaml \
      --ldm_stats data/real_sim_pairs_processed/stats.pt \
      --ldm_mode sim_cond --compare_sources \
      --input_dir data/real_sim_imu_pairs_our_dataset_full \
      --imu_frame world \
      --outdir outputs/carrying_eval_real_sim_gen
"""

import argparse
import csv
import functools
import importlib.util
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from ldm.models.diffusion.ddim_1d import DDIMSampler1D

from ldm.data.carrying_dataset import (
    CLASS_NAMES, CarryingTypeDataset, build_file_list, split_files_stratified,
)
from ldm.data.imu_dataset import resolve_stats
from ldm.models.carrying_classifier import CarryingTypeClassifier, LatentMLPClassifier
from ldm.util import instantiate_from_config

CHANNEL_NAMES = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
SAMPLE_RATE = 200
WINDOW_SAMPLES = 2000  # 10s at 200Hz
WINDOW_SEC = WINDOW_SAMPLES / SAMPLE_RATE

# Window filtering for the confusion matrices only. Recordings start and end
# with a gyro calibration (phone held vertical, swayed side to side while
# standing) and contain incidental standing pauses; neither carries evidence of
# how the phone is being carried. Timelines, window plots, majority votes and
# the agreement metrics all stay unfiltered.
EXCL_ONSET_DISP_M = 1.0        # travel from a recording edge that marks walking
EXCL_ONSET_BACKOFF_S = 2.0     # lead-in kept around the detected onset
EXCL_STATIONARY_HORIZON_S = 2.0  # window over which net travel is measured
EXCL_STATIONARY_DISP_M = 0.5   # net travel below this counts as standing
EXCL_STATIONARY_RUN_S = 3.0    # contiguous standing that voids a window
EXCL_SWAY_DISP_M = 1.0         # calibration stays within this of its start
EXCL_SWAY_GYRO_Y_RMS = 0.5     # rad/s, side-to-side sway amplitude
EXCL_SWAY_RATE_HZ = (0.2, 1.2)  # sway cycles/s; walking cadence sits above
EXCL_IMU_GYRO_QUIET = 0.15     # rad/s, p95 gyro magnitude of a still phone
EXCL_IMU_ACCEL_QUIET = 0.30    # m/s^2, accel magnitude spread of a still phone

SOURCE_NAMES = ("real", "sim", "gen")
SOURCE_STYLES = {
    "real": {"color": "k", "linewidth": 0.9, "alpha": 0.85},
    "sim": {"color": "tab:orange", "linewidth": 0.9, "alpha": 0.8},
    "gen": {"color": "tab:blue", "linewidth": 0.9, "alpha": 0.8},
}

ACCEL_LOCAL_COLS = ["accel_local_x", "accel_local_y", "accel_local_z"]
GYRO_LOCAL_COLS = ["gyro_local_x", "gyro_local_y", "gyro_local_z"]
ACCEL_WORLD_COLS = ["accel_world_x", "accel_world_y", "accel_world_z"]
GYRO_WORLD_COLS = ["gyro_world_x", "gyro_world_y", "gyro_world_z"]
PHONE_ROT_COLS = ["phone_rot_x", "phone_rot_y", "phone_rot_z", "phone_rot_w"]


# ---------------------------------------------------------------------------
# Sibling scripts
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def load_script_module(filename):
    """Import a sibling script by path; cached so it executes once per run."""
    path = os.path.join(os.path.dirname(__file__), filename)
    name = f"carrying_eval_{os.path.splitext(filename)[0]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preprocess_module():
    return load_script_module("preprocess_imu.py")


# ---------------------------------------------------------------------------
# IMU loading
# ---------------------------------------------------------------------------

def _export_world_to_sim(v, heading_rad=0.0):
    """Invert juxta-simulator-rs map_world_frame (heading then Y-up→Z-up axes)."""
    ch, sh = np.cos(heading_rad), np.sin(heading_rad)
    ex, ey, ez = v[:, 0], v[:, 1], v[:, 2]
    bx = ch * ex + sh * ey
    by = -sh * ex + ch * ey
    bz = ez
    # inv map_world_axes: (bx,by,bz)=(sx,-sz,sy) → (sx,sy,sz)=(bx,bz,-by)
    return np.stack([bx, bz, -by], axis=1)


def world_imu_to_local_parquet(accel_w, gyro_w, phone_rot_xyzw, heading_rad=0.0):
    """Rotate exported parquet world-frame IMU to phone-local via phone_rot.

    Matches simulator: accel_local = phone_rot^{-1} * accel_world_sim, after
    undoing the export-frame axis remap / heading applied to world columns.
    """
    from scipy.spatial.transform import Rotation as R

    accel_sim = _export_world_to_sim(accel_w, heading_rad)
    gyro_sim = _export_world_to_sim(gyro_w, heading_rad)
    rot = R.from_quat(phone_rot_xyzw)
    return rot.inv().apply(accel_sim), rot.inv().apply(gyro_sim)


def load_parquet_imu(path, imu_frame="local", parquet_from_world=False,
                     world_heading=0.0):
    """Load [N, 6] IMU (accel|gyro) from a simulator parquet.

    imu_frame:
      - local: use accel_local_*/gyro_local_* (or rotate world→local if
               parquet_from_world / local columns missing)
      - world: use accel_world_*/gyro_world_* directly
    """
    import pandas as pd

    df = pd.read_parquet(path)
    has_local = all(c in df.columns for c in ACCEL_LOCAL_COLS + GYRO_LOCAL_COLS)
    has_world = all(c in df.columns for c in ACCEL_WORLD_COLS + GYRO_WORLD_COLS)
    has_rot = all(c in df.columns for c in PHONE_ROT_COLS)

    if imu_frame == "world":
        if not has_world:
            raise ValueError(f"{path}: missing world IMU columns for --imu_frame world")
        print("  parquet IMU: world columns")
        accel = df[ACCEL_WORLD_COLS].values.astype(np.float64)
        gyro = df[GYRO_WORLD_COLS].values.astype(np.float64)
        return np.concatenate([accel, gyro], axis=1)

    # imu_frame == "local"
    use_world_src = parquet_from_world or not has_local
    if not use_world_src:
        print("  parquet IMU: local columns")
        accel = df[ACCEL_LOCAL_COLS].values.astype(np.float64)
        gyro = df[GYRO_LOCAL_COLS].values.astype(np.float64)
        return np.concatenate([accel, gyro], axis=1)

    if not has_world:
        raise ValueError(
            f"{path}: need world IMU columns to rotate to local "
            f"(parquet_from_world={parquet_from_world}, has_local={has_local})"
        )
    if not has_rot:
        raise ValueError(
            f"{path}: missing phone_rot_*; cannot rotate world IMU to local"
        )

    accel_w = df[ACCEL_WORLD_COLS].values.astype(np.float64)
    gyro_w = df[GYRO_WORLD_COLS].values.astype(np.float64)
    phone_rot = df[PHONE_ROT_COLS].values.astype(np.float64)
    print(
        f"  parquet IMU: world→local via phone_rot "
        f"(heading={world_heading:.4f} rad)"
        + ("; local columns ignored (stale after LDM gen)" if has_local else "")
    )
    accel, gyro = world_imu_to_local_parquet(
        accel_w, gyro_w, phone_rot, heading_rad=world_heading
    )
    return np.concatenate([accel, gyro], axis=1)


def load_parquet_position(path):
    """Load horizontal trajectory and timestamps from a simulator parquet.

    Prefers agent_pos_x / agent_pos_z (z-up horizontal plane). Returns
    (pos [N, 2], time [N] or None). pos is None if no position columns.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    time = df["time"].values.astype(np.float64) if "time" in df.columns else None
    if "agent_pos_x" in df.columns and "agent_pos_z" in df.columns:
        pos = df[["agent_pos_x", "agent_pos_z"]].values.astype(np.float64)
    elif "agent_pos_x" in df.columns and "agent_pos_y" in df.columns:
        pos = df[["agent_pos_x", "agent_pos_y"]].values.astype(np.float64)
    else:
        return None, time
    return pos, time


def load_trajectory_imu(args, input_path):
    """Load classifier-order IMU [N, 6] accel|gyro plus pos/time."""
    ext = os.path.splitext(input_path)[1].lower()
    want_local = args.imu_frame == "local"

    if ext in (".hdf5", ".h5"):
        if args.hdf5_already_world and want_local:
            raise ValueError(
                "--hdf5_already_world is for world-frame synced IMU; "
                "use --imu_frame world (not local)"
            )
        # An already-world file needs the same "no rotation" path as a local one.
        raw = args.hdf5_already_world or want_local
        acce, gyro, pos, time = preprocess_module().load_hdf5(
            input_path, local_frame=raw
        )
        if args.hdf5_already_world:
            print("  hdf5 IMU: world (synced as-is, no game_rv)")
        else:
            print(f"  hdf5 IMU: {'local' if want_local else 'world (via game_rv)'}")
        return np.concatenate([acce, gyro], axis=1), pos, time

    if ext == ".parquet":
        imu_raw = load_parquet_imu(
            input_path,
            imu_frame=args.imu_frame,
            parquet_from_world=args.parquet_from_world,
            world_heading=args.world_heading,
        )
        pos, time = load_parquet_position(input_path)
        return imu_raw, pos, time

    raise ValueError(f"Unsupported file type: {ext}")


def resolve_real_path(input_path):
    """The recording to classify: input_path itself, or its real.hdf5."""
    if not os.path.isdir(input_path):
        return input_path
    real = os.path.join(input_path, "real.hdf5")
    if not os.path.isfile(real):
        raise ValueError(f"{input_path} is a directory but has no real.hdf5")
    return real


def resolve_sim_path(input_path, args):
    """Find synthetic parquet for sim-cond LDM next to the trajectory file."""
    if args.sim_input:
        if not os.path.isfile(args.sim_input):
            raise FileNotFoundError(f"--sim_input not found: {args.sim_input}")
        return args.sim_input
    d = os.path.dirname(os.path.abspath(input_path))
    for cand in (
        os.path.join(d, "synthetic.parquet"),
        os.path.join(d, ".run", "synthetic_0000.parquet"),
        os.path.join(d, ".run", "synthetic.parquet"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def list_trajectory_files(input_dir, recursive_pairs=False):
    """Sorted .hdf5 / .parquet files under input_dir.

    If none are found at the top level and recursive_pairs is True, also
    pick real.hdf5 from pair subfolders (real.hdf5 + synthetic.parquet).
    """
    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in (
            ".hdf5", ".h5", ".parquet"
        ):
            files.append(path)
    if files or not recursive_pairs:
        return files
    return sorted(
        os.path.join(root, "real.hdf5")
        for root, _dirs, fnames in os.walk(input_dir)
        if "real.hdf5" in fnames
    )


def list_pair_dirs(input_dir):
    """Sorted directories under input_dir that hold a real.hdf5."""
    return sorted(
        root for root, _dirs, fnames in os.walk(input_dir) if "real.hdf5" in fnames
    )


def output_name(path, input_dir):
    """Flattened, collision-free subdirectory name for one input under a root."""
    rel = os.path.splitext(os.path.relpath(path, input_dir))[0]
    # A pair dir names its recording real.hdf5, so the dir carries the identity.
    if os.path.basename(rel) == "real":
        rel = os.path.dirname(rel)
    return rel.replace(os.sep, "_") or os.path.basename(os.path.abspath(input_dir))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_vae(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def load_classifier(clf_ckpt_path, device):
    ckpt = torch.load(clf_ckpt_path, map_location="cpu")
    mlp = LatentMLPClassifier(
        in_dim=ckpt['in_dim'],
        hidden=tuple(ckpt['hidden']),
        n_classes=ckpt['n_classes'],
        dropout=ckpt.get('dropout', 0.0),
    )
    mlp.load_state_dict(ckpt['mlp_state_dict'])
    class_names = ckpt.get('class_names', CLASS_NAMES)
    return mlp.to(device).eval(), class_names, ckpt


def load_ldm_bundle(args, device):
    """Load LDM + DDIM sampler + stats used for IMU generation."""
    if args.ldm_mode == "sim_cond":
        ldm_mod = load_script_module("eval_ronin_ldm_sim_cond.py")
        config = args.ldm_config or "configs/imu/ldm_1d_sim_cond.yaml"
    else:
        ldm_mod = load_script_module("eval_ronin_ldm.py")
        config = args.ldm_config or "configs/imu/ldm_1d.yaml"

    stats_path = args.ldm_stats or args.stats
    if not stats_path:
        raise ValueError("LDM generation requires --ldm_stats or --stats (needs vel_mean/vel_std)")
    stats_keys = torch.load(stats_path, weights_only=True).keys()
    missing = [k for k in ("imu_mean", "imu_std", "vel_mean", "vel_std")
               if k not in stats_keys]
    if missing:
        raise ValueError(
            f"LDM stats {stats_path} is missing {missing}. Pass the stats.pt the "
            f"LDM was trained with via --ldm_stats (classifier --stats usually "
            f"has no vel_mean/vel_std)."
        )

    first_stage_ckpt = args.ldm_vae_ckpt or args.vae_ckpt
    print(f"Loading LDM ({args.ldm_mode}) from {args.ldm_ckpt} ...")
    print(f"  first stage: {first_stage_ckpt}"
          + ("" if args.ldm_vae_ckpt else " (from --vae_ckpt; pass --ldm_vae_ckpt "
                                          "if the LDM used a different VAE)"))
    ldm = ldm_mod.load_ldm(
        config, args.ldm_ckpt, device,
        first_stage_ckpt=first_stage_ckpt,
        scale_factor=args.ldm_scale_factor,
    )
    print(f"  LDM scale_factor={ldm.scale_factor}, config={config}")
    return {
        "mod": ldm_mod,
        "ldm": ldm,
        "sampler": DDIMSampler1D(ldm),
        "stats": ldm_mod.load_vae_stats(stats_path),
        "mode": args.ldm_mode,
    }


def swap_imu_channels(imu):
    """Convert between [accel(3), gyro(3)] and [gyro(3), accel(3)] ordering.

    The swap is its own inverse, so the same call takes classifier/LDM order to
    RoNIN order and back.
    """
    return imu[:, [3, 4, 5, 0, 1, 2]]


def generate_ldm_imu(args, bundle, imu_clf, pos, time, sim_imu, device, seed):
    """Generate world-frame IMU [N, 6] accel|gyro from a fresh noise seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if pos is None:
        raise ValueError("LDM generation needs trajectory position (tango_pos / agent_pos)")
    n = imu_clf.shape[0]
    pos = np.asarray(pos)[:n]
    if time is None:
        ts = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    else:
        ts = np.asarray(time, dtype=np.float64)[:n]

    imu_mean, imu_std, vel_mean, vel_std = bundle["stats"]
    kwargs = dict(
        features=swap_imu_channels(np.asarray(imu_clf, dtype=np.float32)),
        ts=ts,
        gt_pos=pos,
        model=bundle["ldm"],
        sampler=bundle["sampler"],
        imu_mean=imu_mean,
        imu_std=imu_std,
        vel_mean=vel_mean,
        vel_std=vel_std,
        device=device,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        use_ema=not args.no_ema,
        strength=args.strength,
    )
    if bundle["mode"] == "sim_cond":
        if sim_imu is None:
            raise ValueError(
                "sim-cond LDM needs synthetic IMU (--sim_input or synthetic.parquet "
                "next to the trajectory)"
            )
        kwargs["sim_features_ldm"] = np.asarray(sim_imu, dtype=np.float32)[:n]
    return swap_imu_channels(bundle["mod"].generate_features_ldm(**kwargs))


# ---------------------------------------------------------------------------
# Ground truth from paths
# ---------------------------------------------------------------------------

# Carrying placements as they appear in dataset paths, flat gen_world stems and
# the pair manifest (which writes swing_left where paths use left_swing),
# collapsed onto the four classes the classifier predicts.
CARRYING_ALIASES = {
    "chest": "chest",
    "demohand": "demo_hand",
    "demo_hand": "demo_hand",
    "hand": "demo_hand",
    "left_pocket": "pocket",
    "right_pocket": "pocket",
    "pocket": "pocket",
    "left_swing": "swinging",
    "right_swing": "swinging",
    "swing_left": "swinging",
    "swing_right": "swinging",
    "swing": "swinging",
    "swinging": "swinging",
}

# Longest alias first, so "left_swing_..." resolves as a swing instead of
# falling through to a shorter alias.
CARRYING_ALIASES_BY_LENGTH = sorted(CARRYING_ALIASES, key=len, reverse=True)


def infer_gt_class_from_name(path, class_names):
    """Best-effort GT class from a filename / directory name.

    Handles the flat gen_world stems (``left_pocket_richa_1_slam.hdf5``) and the
    carrying-type directories of the pair datasets. Matching prefers a placement
    that ends on a token boundary, then falls back to a bare prefix for older
    exports that ran placements together (e.g. "pocketrun3").
    """
    stem = os.path.splitext(os.path.basename(str(path).rstrip("/\\")))[0].lower()

    for boundary in (True, False):
        for prefix in CARRYING_ALIASES_BY_LENGTH:
            hit = (stem == prefix or stem.startswith(prefix + "_")) if boundary \
                else stem.startswith(prefix)
            if hit:
                cls = CARRYING_ALIASES[prefix]
                return cls if cls in class_names else None
    return None


def infer_carrying_in_name(name, class_names):
    """Carrying class named anywhere inside a name, not just at its start.

    Sequence names bury the placement in the middle (``john_left_pocket_corrected``),
    so prefix matching alone would miss it. Longest placement wins, and matches
    have to cover whole words so ``richa_hip_corrected`` stays unlabelled.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", str(name).lower()) if w]
    for token in CARRYING_ALIASES_BY_LENGTH:
        parts = token.split("_")
        span = len(parts)
        if any(words[i:i + span] == parts for i in range(len(words) - span + 1)):
            cls = CARRYING_ALIASES[token]
            return cls if cls in class_names else None
    return None


def resolve_gt_class(path, class_names):
    """Carrying class for a recording, from its filename or enclosing folders.

    Covers the flat gen_world layout (``chest_richa_2_slam.hdf5``) and the pair
    layout (``.../chest/richa_2_slam/real.hdf5``), whose filename carries no
    label so the carrying-type directory has to supply it.
    """
    for part in [path] + os.path.abspath(str(path)).split(os.sep)[-2::-1]:
        cls = infer_gt_class_from_name(part, class_names)
        if cls is not None:
            return cls
    return None


@functools.lru_cache(maxsize=None)
def _native_holding_map(manifest):
    """native_holding per pair_id, read from one manifest.csv."""
    table = {}
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            native = (row.get("native_holding") or "").strip()
            if native and row.get("pair_id"):
                table[row["pair_id"]] = native
    return table


def _find_manifest(real_path):
    """Nearest manifest.csv above a recording, or None."""
    directory = os.path.dirname(os.path.abspath(real_path))
    for _ in range(4):
        directory = os.path.dirname(directory)
        candidate = os.path.join(directory, "manifest.csv")
        if os.path.isfile(candidate):
            return candidate
    return None


def infer_native_gt_class(real_path, class_names):
    """Carrying type the real recording actually captured, or None if unknown.

    A pair folder is named for the placement the simulator re-rendered, which
    usually is not how the phone was really being carried, so real IMU has to be
    scored against the manifest's native_holding (or a placement spelled out in
    the sequence name) instead.
    """
    abspath = os.path.abspath(real_path)
    parts = abspath.split(os.sep)
    manifest = _find_manifest(abspath) if len(parts) >= 3 else None
    if manifest is not None:
        native = _native_holding_map(manifest).get("/".join(parts[-3:-1]))
        if native:
            cls = CARRYING_ALIASES.get(native.lower())
            if cls in class_names:
                return cls
    return infer_carrying_in_name(os.path.basename(os.path.dirname(abspath)),
                                  class_names)


# ---------------------------------------------------------------------------
# Confusion-matrix window filtering (calibration / stationary periods)
# ---------------------------------------------------------------------------

def _longest_true_run(mask):
    """Length of the longest contiguous True run in a boolean array."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    # Run lengths from the gaps between transitions, padded so runs touching
    # either end are counted too.
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(edges[1::2] - edges[0::2]))


def _standing_sample_mask(pos, sample_rate=SAMPLE_RATE):
    """Per-sample mask: True where the walker barely travels.

    Uses net travel over a short horizon rather than instantaneous speed, which
    keeps SLAM jitter from reading as motion while standing.
    """
    pos2d = np.asarray(pos, dtype=np.float64)[:, :2]
    n = len(pos2d)
    horizon = max(1, int(round(EXCL_STATIONARY_HORIZON_S * sample_rate)))
    if n <= horizon:
        return np.zeros(n, dtype=bool)
    travel = np.linalg.norm(pos2d[horizon:] - pos2d[:-horizon], axis=1)
    travel = np.concatenate([travel, np.full(horizon, travel[-1])])
    return travel < EXCL_STATIONARY_DISP_M


def _walking_span(pos, sample_rate=SAMPLE_RATE):
    """(onset, offset) sample indices bounding the walked part of a recording.

    The calibration sway keeps the walker on the spot, so the first and last
    samples that get clear of the respective endpoints bracket the real walk.
    """
    pos2d = np.asarray(pos, dtype=np.float64)[:, :2]
    n = len(pos2d)
    backoff = int(round(EXCL_ONSET_BACKOFF_S * sample_rate))

    from_start = np.linalg.norm(pos2d - pos2d[0], axis=1)
    moved = np.flatnonzero(from_start > EXCL_ONSET_DISP_M)
    onset = max(0, int(moved[0]) - backoff) if len(moved) else 0

    from_end = np.linalg.norm(pos2d - pos2d[-1], axis=1)
    moved = np.flatnonzero(from_end > EXCL_ONSET_DISP_M)
    offset = min(n, int(moved[-1]) + backoff) if len(moved) else n
    return onset, offset


def _sway_signature(gyro, sample_rate=SAMPLE_RATE):
    """(rms, cycles/s) of the busiest gyro axis.

    Standing and swaying the phone side to side shows up most clearly on
    device-frame gyro_y; taking the highest-variance axis picks that up without
    assuming the IMU has been left in the device frame.
    """
    gyro = np.asarray(gyro, dtype=np.float64)
    if len(gyro) < 2:
        return 0.0, 0.0
    g = gyro[:, int(np.argmax(gyro.var(axis=0)))]
    g = g - g.mean()
    rms = float(np.sqrt(np.mean(g * g)))
    if rms <= 0:
        return 0.0, 0.0
    # Only count sign changes of samples with real amplitude, so noise hovering
    # around zero does not inflate the rate.
    strong = np.sign(g[np.abs(g) > 0.2 * rms])
    crossings = int(np.count_nonzero(np.diff(strong))) if len(strong) > 1 else 0
    return rms, crossings / (len(g) / sample_rate) / 2.0


class WindowExclusions:
    """Windows holding gyro calibration or standing rather than walking.

    Consumed by the confusion matrices only; predictions, timelines, window
    plots and majority votes all keep every window.
    """

    def __init__(self, windows, used_position):
        self.windows = windows
        self.used_position = used_position

    def __len__(self):
        return len(self.windows)

    @property
    def mask(self):
        return [w["excluded"] for w in self.windows]

    def head(self, n):
        return WindowExclusions(self.windows[:n], self.used_position)

    def summary(self):
        """Counts and per-window detail for the metadata in results.json."""
        excluded = [w for w in self.windows if w["excluded"]]
        return {
            "n_windows": len(self.windows),
            "n_included": len(self.windows) - len(excluded),
            "n_excluded": len(excluded),
            "detector": "position" if self.used_position else "imu_only",
            "by_reason": dict(Counter(r for w in excluded for r in w["reasons"])),
            "excluded_windows": [
                {"window": w["window"], "t_start": w["t_start"],
                 "t_end": w["t_end"], "reasons": w["reasons"]}
                for w in excluded
            ],
        }

    @classmethod
    def all_excluded(cls, n_windows, reason, sample_rate=SAMPLE_RATE):
        return cls(
            [{"window": i, "t_start": i * WINDOW_SEC, "t_end": (i + 1) * WINDOW_SEC,
              "excluded": True, "reasons": [reason]}
             for i in range(n_windows)],
            used_position=False,
        )


def compute_window_exclusions(imu_raw, pos, time, n_windows,
                              window_samples=WINDOW_SAMPLES,
                              stride_samples=None,
                              sample_rate=SAMPLE_RATE):
    """Flag windows that hold gyro calibration or standing instead of walking."""
    if stride_samples is None:
        stride_samples = window_samples
    imu_raw = np.asarray(imu_raw)
    time = None if time is None else np.asarray(time, dtype=np.float64)

    have_pos = pos is not None and len(np.asarray(pos)) >= len(imu_raw)
    if have_pos:
        pos2d = np.asarray(pos, dtype=np.float64)[:len(imu_raw), :2]
        standing = _standing_sample_mask(pos2d, sample_rate)
        onset, offset = _walking_span(pos2d, sample_rate)
    min_run = int(round(EXCL_STATIONARY_RUN_S * sample_rate))

    windows = []
    for i in range(n_windows):
        s = i * stride_samples
        e = s + window_samples
        reasons = []

        if have_pos:
            if e <= onset:
                reasons.append("head_calibration")
            elif s >= offset:
                reasons.append("tail_calibration")

            if _longest_true_run(standing[s:e]) >= min_run:
                reasons.append("standing")

            w_pos = pos2d[s:e]
            travelled = float(np.linalg.norm(w_pos - w_pos[0], axis=1).max())
            if travelled < EXCL_SWAY_DISP_M:
                rms, rate = _sway_signature(imu_raw[s:e, 3:6], sample_rate)
                if rms > EXCL_SWAY_GYRO_Y_RMS and EXCL_SWAY_RATE_HZ[0] <= rate <= EXCL_SWAY_RATE_HZ[1]:
                    reasons.append("calibration_sway")
        else:
            # No trajectory to lean on, so only drop windows where the phone
            # itself is unmistakably at rest.
            gyro_mag = np.linalg.norm(imu_raw[s:e, 3:6], axis=1)
            accel_mag = np.linalg.norm(imu_raw[s:e, 0:3], axis=1)
            if (np.percentile(gyro_mag, 95) < EXCL_IMU_GYRO_QUIET
                    and float(np.std(accel_mag)) < EXCL_IMU_ACCEL_QUIET):
                reasons.append("still_imu")

        if time is not None and e <= len(time):
            t_start = float(time[s] - time[0])
            t_end = float(time[e - 1] - time[0])
        else:
            t_start = s / sample_rate
            t_end = e / sample_rate

        windows.append({
            "window": i,
            "t_start": round(t_start, 2),
            "t_end": round(t_end, 2),
            "excluded": bool(reasons),
            "reasons": reasons,
        })
    return WindowExclusions(windows, have_pos)


def compute_hdf5_window_exclusions(path, window_samples, stride_samples,
                                   sample_rate=SAMPLE_RATE):
    """Exclusion mask for one HDF5, windowed the way CarryingTypeDataset does.

    Reads the device-frame IMU whatever frame the classifier runs in, since the
    filter looks at the phone's own motion rather than the classifier input.
    """
    acce, gyro, pos, time = preprocess_module().load_hdf5(path, local_frame=True)
    imu = np.concatenate([acce, gyro], axis=1)
    if len(imu) < window_samples:
        return WindowExclusions([], used_position=False)
    n_windows = (len(imu) - window_samples) // stride_samples + 1
    return compute_window_exclusions(
        imu, pos, time, n_windows, window_samples=window_samples,
        stride_samples=stride_samples, sample_rate=sample_rate,
    )


def kept_window_predictions(preds, excluded_mask):
    """Predictions of the windows that survive the calibration/standing filter."""
    if not excluded_mask:
        return list(preds)
    return [p for p, drop in zip(preds, excluded_mask) if not drop]


class ExclusionTally:
    """Running totals of dropped windows, kept apart from the reason counts.

    A window can trip several rules at once, so the reasons cannot simply be
    summed to get the number of windows that were dropped.
    """

    def __init__(self):
        self.n_excluded = 0
        self.by_reason = Counter()

    def add(self, result):
        summary = result.get("cm_exclusion") or {}
        self.n_excluded += summary.get("n_excluded", 0)
        self.by_reason.update(summary.get("by_reason", {}))

    def as_dict(self):
        return {"n_excluded": self.n_excluded, "by_reason": dict(self.by_reason)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _softmax(x):
    """Softmax over the last axis; accepts one logit vector or a stack of them."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _class_colors(class_names):
    return plt.cm.Set2(np.linspace(0, 1, len(class_names)))


def _pred_indices(preds, class_names):
    """Class indices for predictions given as names (or already as indices)."""
    return [class_names.index(p) if p in class_names else int(p) for p in preds]


def _traj_velocity(pos, time, sample_rate=SAMPLE_RATE):
    """Horizontal velocity from differentiating trajectory position.

    Uses the first two position columns (HDF5 tango x/y, parquet agent x/z).
    Returns (t_sec, vel_xy [N,2], speed [N]) aligned with pos samples.
    """
    pos2d = np.asarray(pos, dtype=np.float64)[:, :2]
    n = len(pos2d)
    if n < 2:
        return None, None, None
    if time is None:
        t = np.arange(n, dtype=np.float64) / sample_rate
        dt = np.full(n - 1, 1.0 / sample_rate)
    else:
        time = np.asarray(time, dtype=np.float64)[:n]
        t = time - time[0]
        dt = np.maximum(np.diff(time), 1e-8)
    vel = np.diff(pos2d, axis=0) / dt[:, None]
    vel = np.vstack([vel[:1], vel])
    return t, vel, np.linalg.norm(vel, axis=1)


def _has_velocity(pos):
    """Whether a velocity panel can be drawn (matches _traj_velocity)."""
    return pos is not None and len(np.asarray(pos)) >= 2


def _plot_velocity_panel(ax, pos, time, n_win, window_sec, sample_rate,
                         shade_by=None, colors=None, window_mean=False):
    """Trajectory velocity with window boundaries, optionally shaded by class."""
    t_start = np.arange(n_win) * window_sec
    if shade_by is not None:
        for i, pred in enumerate(shade_by):
            ax.axvspan(t_start[i], t_start[i] + window_sec,
                       color=colors[pred], alpha=0.12, linewidth=0)
    for edge in np.append(t_start, n_win * window_sec):
        ax.axvline(edge, color="0.6", linewidth=0.6, linestyle="--")

    n_keep = int(round(n_win * window_sec * sample_rate))
    t_vel, vel, speed = _traj_velocity(
        np.asarray(pos)[:n_keep],
        None if time is None else np.asarray(time)[:n_keep],
        sample_rate,
    )
    ax.set_ylabel("m/s")
    ax.grid(True, alpha=0.3)
    if speed is None:
        return
    ax.plot(t_vel, vel[:, 0], linewidth=0.8, alpha=0.85, label="vx", color="C0")
    ax.plot(t_vel, vel[:, 1], linewidth=0.8, alpha=0.85, label="vy", color="C1")
    ax.plot(t_vel, speed, linewidth=1.3, alpha=0.95, label="|v|", color="k")
    ncol = 3
    if window_mean:
        win = int(round(window_sec * sample_rate))
        bounds = [(i * win, min((i + 1) * win, len(speed))) for i in range(n_win)]
        means = [float(np.mean(speed[s:e])) if e > s else 0.0 for s, e in bounds]
        ax.step(np.append(t_start, n_win * window_sec), means + means[-1:],
                where="post", color="0.25", linewidth=1.4, linestyle=":",
                label="window mean |v|")
        ncol = 4
    ax.legend(fontsize=8, loc="upper right", ncol=ncol)


def _plot_pred_heatmap(fig, ax, pred_mat, row_labels, class_names, colors,
                       window_sec, title):
    """Rows of per-window predicted classes as a categorical heatmap."""
    n_rows, n_win = pred_mat.shape
    im = ax.imshow(
        pred_mat, aspect="auto", interpolation="nearest",
        cmap=plt.cm.colors.ListedColormap(colors),
        vmin=-0.5, vmax=len(class_names) - 0.5,
        extent=[0, n_win * window_sec, n_rows - 0.5, -0.5],
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                        ticks=range(len(class_names)))
    cbar.ax.set_yticklabels(class_names, fontsize=8)


def _save(fig, outdir, filename, what, dpi=150):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {what} to {path}")
    return path


def plot_imu_window(imu, outdir, idx, gt_label=None, pred_label=None,
                    sample_rate=SAMPLE_RATE, time_offset=0.0):
    """Plot 6-channel IMU for one window with GT/Pred in title."""
    t = np.arange(imu.shape[-1]) / sample_rate + time_offset
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for c, ax in enumerate(axes):
        ax.plot(t, imu[c], linewidth=0.8, alpha=0.9)
        ax.set_ylabel(CHANNEL_NAMES[c], fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")

    parts = [f"Window {idx}"]
    if gt_label is not None:
        parts.append(f"GT: {gt_label}")
    if pred_label is not None:
        parts.append(f"Pred: {pred_label}")
    fig.suptitle("  |  ".join(parts), fontsize=12)

    fig.savefig(os.path.join(outdir, f"window_{idx:03d}.png"), dpi=140)
    plt.close(fig)


def plot_source_overlay(imu_by_source, outdir, idx, label_by_source,
                        sample_rate=SAMPLE_RATE, time_offset=0.0):
    """Overlay real / sim / gen IMU for one window, one subplot per channel."""
    any_imu = next(iter(imu_by_source.values()))
    t = np.arange(any_imu.shape[-1]) / sample_rate + time_offset
    fig, axes = plt.subplots(6, 1, figsize=(13, 11), sharex=True,
                             constrained_layout=True)
    for c, ax in enumerate(axes):
        for name, imu in imu_by_source.items():
            ax.plot(t, imu[c], label=name, **SOURCE_STYLES[name])
        ax.set_ylabel(CHANNEL_NAMES[c], fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right", ncol=len(imu_by_source))
    axes[-1].set_xlabel("time (s)")

    preds = "  |  ".join(f"{n}: {label_by_source[n]}" for n in imu_by_source)
    fig.suptitle(f"Window {idx}  ({t[0]:.0f}-{t[-1]:.0f}s)\n{preds}", fontsize=11)

    fig.savefig(os.path.join(outdir, f"window_{idx:03d}.png"), dpi=140)
    plt.close(fig)


def plot_timeline(predictions, class_names, outdir, window_sec=WINDOW_SEC,
                  logits=None, pos=None, time=None, sample_rate=SAMPLE_RATE):
    """Plot predicted class vs time, optionally with velocity, logits, and confidence."""
    n = len(predictions)
    t_start = np.arange(n) * window_sec
    t_mid = t_start + window_sec / 2
    colors = _class_colors(class_names)

    has_logits = logits is not None and len(logits) == n
    has_vel = _has_velocity(pos)
    probs = _softmax(logits) if has_logits else None

    n_rows = 1 + int(has_vel) + (2 if has_logits else 0)
    height = 4 + (3 if has_vel else 0) + (5 if has_logits else 0)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(max(8, n * 0.6), height),
        sharex=True, constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    # --- Panel 0: predicted class bars ---
    ax = axes[0]
    for i, pred in enumerate(predictions):
        ax.barh(0, window_sec, left=t_start[i], height=0.6,
                color=colors[pred], edgecolor='white', linewidth=0.5)
        label = class_names[pred]
        if has_logits:
            label = f"{label}\n{probs[i, pred]:.2f}"
        ax.text(t_mid[i], 0, label, ha='center', va='center', fontsize=7)
    ax.set_yticks([])
    ax.set_title("Predicted carrying type" + (" (label = class, conf)" if has_logits else ""))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(class_names))]
    ax.legend(handles, class_names, loc='upper right', fontsize=8)

    row = 1
    if has_vel:
        _plot_velocity_panel(axes[row], pos, time, n, window_sec, sample_rate,
                             shade_by=predictions, colors=colors, window_mean=True)
        axes[row].set_title("Trajectory velocity (d/dt position, window-aligned)")
        row += 1

    if has_logits:
        logits = np.asarray(logits, dtype=np.float64)
        for values, ylabel, title, ylim in (
            (logits, "logit", "Class logits per window", None),
            (probs, "softmax prob", "Class confidence (softmax)", (-0.05, 1.05)),
        ):
            ax = axes[row]
            for c, name in enumerate(class_names):
                ax.plot(t_mid, values[:, c], marker='o', markersize=4,
                        linewidth=1.2, color=colors[c], label=name)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            row += 1

    axes[-1].set_xlabel("time (s)")
    _save(fig, outdir, "timeline.png", "timeline")


def plot_noise_comparison(runs, class_names, outdir, window_sec=WINDOW_SEC,
                          pos=None, time=None, sample_rate=SAMPLE_RATE):
    """Compare classifier labels across LDM noise inits for one trajectory."""
    n_runs = len(runs)
    pred_mat = np.array(
        [_pred_indices(r["predictions"], class_names) for r in runs], dtype=np.int64
    )
    n_win = pred_mat.shape[1]
    colors = _class_colors(class_names)
    t_start = np.arange(n_win) * window_sec

    has_vel = _has_velocity(pos)
    heights = [0.9 + 0.12 * n_runs] + ([2.2] if has_vel else []) + [2.4]
    height = (2.2 + 0.28 * n_runs) + (3.0 if has_vel else 0) + 2.4
    fig, axes = plt.subplots(
        len(heights), 1, figsize=(max(10, n_win * 0.55), height),
        sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": heights},
    )

    _plot_pred_heatmap(
        fig, axes[0], pred_mat, [f"seed {r['ldm_seed']}" for r in runs],
        class_names, colors, window_sec, "Predicted class vs LDM noise init",
    )

    row = 1
    if has_vel:
        _plot_velocity_panel(axes[row], pos, time, n_win, window_sec, sample_rate)
        axes[row].set_title("Trajectory velocity")
        row += 1

    ax = axes[row]
    # Bars share the x axis with the panels above, so use seconds (not indices)
    x = t_start + window_sec / 2
    bottom = np.zeros(n_win)
    for c, name in enumerate(class_names):
        frac = (pred_mat == c).mean(axis=0)
        ax.bar(x, frac, bottom=bottom, color=colors[c],
               width=window_sec * 0.9, label=name)
        bottom += frac
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of noise inits")
    ax.set_title("Per-window class agreement across noise inits")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}" for t in t_start], fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xlabel("window start time (s)")

    _save(fig, outdir, "noise_comparison.png", "noise comparison")


def plot_source_comparison(per_source, class_names, outdir, window_sec=WINDOW_SEC,
                           pos=None, time=None, sample_rate=SAMPLE_RATE,
                           gt_class=None, title=None):
    """Compare classifier output on real vs sim vs gen IMU for one trajectory."""
    names = [n for n in SOURCE_NAMES if n in per_source]
    pred_mat = np.array(
        [_pred_indices(per_source[n]["predictions"], class_names) for n in names],
        dtype=np.int64,
    )
    n_win = pred_mat.shape[1]
    colors = _class_colors(class_names)
    t_start = np.arange(n_win) * window_sec
    t_mid = t_start + window_sec / 2

    has_vel = _has_velocity(pos)
    heights = [0.5 + 0.3 * len(names), 2.4] + ([2.2] if has_vel else []) + [2.4]
    fig, axes = plt.subplots(
        len(heights), 1, figsize=(min(max(11, n_win * 0.6), 40), sum(heights) + 1.5),
        sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": heights},
    )

    _plot_pred_heatmap(
        fig, axes[0], pred_mat, names, class_names, colors, window_sec,
        "Predicted class per window"
        + ("" if gt_class is None else f"   (GT: {gt_class})"),
    )

    # Confidence of each source in its own predicted class
    ax = axes[1]
    for n in names:
        ax.plot(t_mid, per_source[n]["confidence"], marker="o", markersize=3.5,
                label=n, **SOURCE_STYLES[n])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("confidence")
    ax.set_title("Softmax confidence in the predicted class")
    ax.legend(fontsize=8, loc="upper right", ncol=len(names))
    ax.grid(True, alpha=0.3)

    row = 2
    if has_vel:
        _plot_velocity_panel(axes[row], pos, time, n_win, window_sec, sample_rate)
        axes[row].set_title("Trajectory velocity")
        row += 1

    # Probability the classifier assigns to the real IMU's class, per source
    ax = axes[row]
    ref = np.array(_pred_indices(per_source["real"]["predictions"], class_names))
    width = window_sec * 0.9 / len(names)
    for k, n in enumerate(names):
        probs = np.asarray(per_source[n]["softmax"])
        offset = (k - (len(names) - 1) / 2) * width
        ax.bar(t_mid + offset, probs[np.arange(n_win), ref], width=width, label=n,
               color=SOURCE_STYLES[n]["color"], alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(real's class)")
    ax.set_title("Probability assigned to the class predicted from real IMU")
    step = max(1, int(np.ceil(n_win / 40)))
    ax.set_xticks(t_mid[::step])
    ax.set_xticklabels([f"{t:.0f}" for t in t_start[::step]], fontsize=8)
    ax.legend(fontsize=8, loc="upper right", ncol=len(names))
    ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xlabel("window start time (s)")

    if title:
        fig.suptitle(title, fontsize=12)
    _save(fig, outdir, "source_comparison.png", "source comparison")


def plot_confusion_matrix(cm, class_names, outdir, filename="confusion_matrix.png",
                          title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.colorbar(im, ax=ax)
    _save(fig, outdir, filename, "confusion matrix")


def plot_training_curves(results_path, outdir):
    """Plot train/val loss and accuracy from training results.json history."""
    with open(results_path) as f:
        results = json.load(f)
    history = results.get("history")
    if not history:
        print(f"No history found in {results_path}; skipping training curves")
        return

    epochs = [h["epoch"] for h in history]
    best_val = results.get("best_val_acc")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], label="train", alpha=0.85)
    ax.plot(epochs, [h["val_loss"] for h in history], label="val", alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Cross-entropy loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [h["train_acc"] for h in history], label="train", alpha=0.85)
    ax.plot(epochs, [h["val_acc"] for h in history], label="val", alpha=0.85)
    if best_val is not None:
        ax.axhline(best_val, color="C2", linestyle="--", linewidth=1.0,
                   label=f"best val={best_val:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    _save(fig, outdir, "training_curves.png", "training curves")


def resolve_train_results(train_results, clf_ckpt):
    """Find training results.json: explicit path, or next to clf_ckpt."""
    if train_results is not None:
        if not os.path.isfile(train_results):
            raise FileNotFoundError(f"--train_results not found: {train_results}")
        return train_results
    guess = os.path.join(os.path.dirname(os.path.abspath(clf_ckpt)), "results.json")
    return guess if os.path.isfile(guess) else None


def build_window_confusion(labels, preds, class_names, outdir, filename,
                           title="Confusion Matrix"):
    """Confusion matrix over already-filtered windows; writes a PNG and a dict."""
    if not labels:
        return None
    idx = {name: i for i, name in enumerate(class_names)}
    y_true = np.array([idx[c] for c in labels])
    y_pred = np.array([idx[c] for c in preds])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion_matrix(cm, class_names, outdir, filename=filename, title=title)
    return {
        "matrix": cm.tolist(),
        "class_names": list(class_names),
        "n_windows": len(y_true),
        "accuracy": float(np.mean(y_true == y_pred)),
        "plot": filename,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_imu_windows(imu_raw, clf, imu_mean, imu_std, device, batch_size=32):
    """Classify consecutive non-overlapping 10s windows of [N, 6] accel|gyro IMU.

    Returns (predictions [W], logits [W, C], windows [W, 6, WINDOW_SAMPLES]) with
    the windows kept unstandardized for plotting.
    """
    n_windows = imu_raw.shape[0] // WINDOW_SAMPLES
    if n_windows == 0:
        raise ValueError(
            f"{imu_raw.shape[0]} samples is short of one {WINDOW_SAMPLES}-sample window"
        )
    windows = np.stack([
        np.asarray(imu_raw[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES]).T
        for i in range(n_windows)
    ]).astype(np.float32)

    mean = imu_mean.view(1, 6, 1).to(device)
    std = imu_std.view(1, 6, 1).to(device)
    batches = []
    with torch.no_grad():
        for s in range(0, n_windows, batch_size):
            x = torch.from_numpy(windows[s:s + batch_size]).to(device)
            batches.append(clf((x - mean) / std).cpu().numpy())
    logits = np.concatenate(batches, axis=0)
    return [int(p) for p in logits.argmax(1)], logits, windows


def window_result(predictions, logits, class_names):
    """Per-window predictions, confidences and majority vote as JSON-ready dict."""
    probs = _softmax(logits)
    majority_idx, majority_count = Counter(predictions).most_common(1)[0]
    return {
        "n_windows": len(predictions),
        "predictions": [class_names[p] for p in predictions],
        "logits": np.asarray(logits).tolist(),
        "softmax": probs.tolist(),
        "confidence": [float(probs[i, p]) for i, p in enumerate(predictions)],
        "majority_vote": class_names[majority_idx],
        "majority_count": majority_count,
    }


class EvalContext:
    """Models, stats and options shared by every evaluation mode."""

    def __init__(self, args, clf, imu_mean, imu_std, device, class_names,
                 ldm_bundle=None):
        self.args = args
        self.clf = clf
        self.imu_mean = imu_mean
        self.imu_std = imu_std
        self.device = device
        self.class_names = class_names
        self.ldm_bundle = ldm_bundle

    def classify(self, imu_raw):
        return classify_imu_windows(imu_raw, self.clf, self.imu_mean, self.imu_std,
                                    self.device, batch_size=self.args.batch_size)

    def n_window_plots(self, n_windows):
        return n_windows if self.args.n_plot < 0 else min(self.args.n_plot, n_windows)


# ---------------------------------------------------------------------------
# Single trajectory mode
# ---------------------------------------------------------------------------

def write_window_plots(window_imus, outdir, labels, n_plot):
    """Per-window IMU plots with their predicted label, into outdir/window_plots."""
    if n_plot <= 0:
        return
    plot_dir = os.path.join(outdir, "window_plots")
    os.makedirs(plot_dir, exist_ok=True)
    for i in range(n_plot):
        plot_imu_window(window_imus[i], plot_dir, i, gt_label="N/A",
                        pred_label=labels[i], time_offset=i * WINDOW_SEC)
    print(f"Saved {n_plot} window plots to {plot_dir}")


def write_trajectory_eval(ctx, input_path, outdir, predictions, logits, window_imus,
                          pos, time, plots=True, extra=None, exclusions=None):
    """Write timeline, window plots, and results.json for one classified IMU.

    ``exclusions`` only annotates the results so the aggregate confusion matrix
    can skip calibration and standing windows; everything written here (timeline,
    window plots, majority vote) still covers the whole recording.
    """
    class_names = ctx.class_names
    n_windows = len(predictions)
    os.makedirs(outdir, exist_ok=True)
    plot_timeline(predictions, class_names, outdir, logits=logits, pos=pos, time=time)

    results = window_result(predictions, logits, class_names)
    if plots:
        write_window_plots(
            window_imus, outdir,
            [f"{results['predictions'][i]} ({results['confidence'][i]:.2f})"
             for i in range(n_windows)],
            ctx.n_window_plots(n_windows),
        )

    gt_class = resolve_gt_class(input_path, class_names)
    print(f"\nPer-window predictions: {results['predictions']}")
    print(f"Per-window confidence: {[f'{c:.2f}' for c in results['confidence']]}")
    print(f"Majority vote: {results['majority_vote']} "
          f"({results['majority_count']}/{n_windows} windows)"
          + (f"  |  GT(from name): {gt_class}" if gt_class else ""))

    results = {
        "input": os.path.abspath(input_path),
        **results,
        "gt_from_name": gt_class,
        "class_names": list(class_names),
    }
    if exclusions is not None:
        if len(exclusions) != n_windows:
            # Without a window-for-window match there is no safe way to line the
            # mask up, so drop the recording from the matrix rather than
            # mislabel windows.
            print(f"  [warn] exclusion mask covers {len(exclusions)} of {n_windows} "
                  f"windows; leaving this recording out of the confusion matrix")
            exclusions = WindowExclusions.all_excluded(n_windows, "mask_misaligned")
        results["cm_window_excluded"] = exclusions.mask
        results["cm_exclusion"] = exclusions.summary()
        n_excl = results["cm_exclusion"]["n_excluded"]
        print(f"Confusion-matrix filter: {n_windows - n_excl}/{n_windows} windows kept"
              + (f" (dropped {results['cm_exclusion']['by_reason']})" if n_excl else ""))
    if extra:
        results.update(extra)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def eval_single_trajectory(ctx, input_path=None, outdir=None, plots=True):
    """Classify consecutive non-overlapping windows from a single file.

    With ``ctx.ldm_bundle`` set, generate IMU from the LDM (optionally over
    ``--n_noise`` independent noise inits) before classifying.

    Returns a results dict, or None if the recording is too short.
    """
    args = ctx.args
    bundle = ctx.ldm_bundle
    input_path = resolve_real_path(input_path or args.input)
    outdir = outdir or args.outdir
    os.makedirs(outdir, exist_ok=True)

    imu_raw, pos, time = load_trajectory_imu(args, input_path)
    N = imu_raw.shape[0]

    sim_imu = None
    if bundle is not None and bundle["mode"] == "sim_cond":
        sim_path = resolve_sim_path(input_path, args)
        if sim_path is None:
            raise FileNotFoundError(
                f"sim-cond LDM: no synthetic.parquet for {input_path}; pass --sim_input"
            )
        print(f"  sim IMU: {sim_path}")
        sim_imu = bundle["mod"].load_sim_imu_ldm(sim_path, n_samples=N)
        # The sim recording conditions every window, so it bounds the usable span
        if sim_imu.shape[0] < N:
            print(f"  [trim] sim IMU is shorter ({sim_imu.shape[0]} samples); "
                  f"truncating real IMU {N} -> {sim_imu.shape[0]}")
            N = sim_imu.shape[0]
            imu_raw = imu_raw[:N]
            pos = None if pos is None else np.asarray(pos)[:N]
            time = None if time is None else np.asarray(time)[:N]

    n_windows = N // WINDOW_SAMPLES
    if n_windows == 0:
        print(f"Recording too short for a full 10s window ({N} samples)")
        return None
    print(f"  {N} samples ({N / SAMPLE_RATE:.1f}s), {n_windows} consecutive windows")

    # Derived from the recorded trajectory, so LDM samples of the same walk all
    # share it.
    exclusions = compute_window_exclusions(imu_raw, pos, time, n_windows)

    def classify_and_write(imu, dest, extra):
        preds, logits, wins = ctx.classify(imu)
        return write_trajectory_eval(
            ctx, input_path, dest, preds, logits, wins, pos, time,
            plots=plots, extra=extra, exclusions=exclusions.head(len(preds)),
        )

    if bundle is None:
        return classify_and_write(imu_raw, outdir, extra=None)

    if args.imu_frame != "world":
        print("  [warn] LDM generates world-frame IMU; classifier --imu_frame "
              f"is '{args.imu_frame}'. Prefer --imu_frame world.")

    n_noise = args.n_noise
    if n_noise == 1:
        seed = args.seed
        print(f"  LDM generate (seed={seed}) ...")
        gen = generate_ldm_imu(args, bundle, imu_raw, pos, time, sim_imu,
                               ctx.device, seed)
        return classify_and_write(gen, outdir, {"ldm_seed": seed, "n_noise": 1})

    runs = []
    for k in range(n_noise):
        seed = args.seed + k
        print(f"\n  LDM noise {k + 1}/{n_noise} (seed={seed}) ...")
        gen = generate_ldm_imu(args, bundle, imu_raw, pos, time, sim_imu,
                               ctx.device, seed)
        runs.append(classify_and_write(
            gen, os.path.join(outdir, f"noise_{k:03d}_seed{seed}"),
            {"ldm_seed": seed, "noise_index": k},
        ))

    return write_noise_sweep_summary(ctx, input_path, outdir, runs, pos, time)


def write_noise_sweep_summary(ctx, input_path, outdir, runs, pos, time):
    """Aggregate the per-noise-init runs of one trajectory into one summary."""
    class_names = ctx.class_names
    n_noise = len(runs)
    plot_noise_comparison(runs, class_names, outdir, pos=pos, time=time)

    pred_mat = np.array([_pred_indices(r["predictions"], class_names) for r in runs])
    per_window = [Counter(pred_mat[:, w]).most_common(1)[0]
                  for w in range(pred_mat.shape[1])]
    # One verdict per physical window so the aggregate confusion matrix counts
    # each window once rather than once per noise init.
    maj_counts = Counter(r["majority_vote"] for r in runs)
    majority_vote, majority_count = maj_counts.most_common(1)[0]
    summary = {
        "input": os.path.abspath(input_path),
        "n_noise": n_noise,
        "seeds": [r["ldm_seed"] for r in runs],
        "n_windows": runs[0]["n_windows"],
        "mean_window_agreement": float(np.mean([c / n_noise for _, c in per_window])),
        "per_window_agreement": [c / n_noise for _, c in per_window],
        "predictions": [class_names[int(cls)] for cls, _ in per_window],
        "cm_window_excluded": runs[0].get("cm_window_excluded"),
        "cm_exclusion": runs[0].get("cm_exclusion"),
        "majority_votes": [r["majority_vote"] for r in runs],
        "majority_vote": majority_vote,
        "majority_count": majority_count,
        "gt_from_name": runs[0].get("gt_from_name"),
        "runs": [
            {"seed": r["ldm_seed"], "predictions": r["predictions"],
             "majority_vote": r["majority_vote"], "confidence": r["confidence"]}
            for r in runs
        ],
        "class_names": list(class_names),
    }
    for name in ("noise_summary.json", "results.json"):
        with open(os.path.join(outdir, name), "w") as f:
            json.dump(summary, f, indent=2)
    print(f"\n  Mean per-window agreement across {n_noise} noise inits: "
          f"{summary['mean_window_agreement']:.3f}")
    print(f"  Majority-of-runs: {majority_vote} ({majority_count}/{n_noise})")
    return summary


# ---------------------------------------------------------------------------
# Real / sim / gen comparison mode
# ---------------------------------------------------------------------------

def eval_source_comparison(ctx, input_path=None, outdir=None, plots=True):
    """Classify real, sim, and LDM-generated IMU for one trajectory and compare.

    Returns a summary dict, or None if the recording is too short.
    """
    args = ctx.args
    bundle = ctx.ldm_bundle
    class_names = ctx.class_names
    real_path = resolve_real_path(input_path or args.input)
    sim_path = resolve_sim_path(real_path, args)
    if sim_path is None:
        raise FileNotFoundError(
            f"no synthetic.parquet next to {real_path}; pass --sim_input"
        )
    outdir = outdir or args.outdir
    os.makedirs(outdir, exist_ok=True)

    real_imu, pos, time = load_trajectory_imu(args, real_path)
    print(f"  sim IMU: {sim_path}")
    sim_imu = load_parquet_imu(sim_path, imu_frame="world")

    N = min(len(real_imu), len(sim_imu))
    if len(real_imu) != len(sim_imu):
        print(f"  [trim] real {len(real_imu)} / sim {len(sim_imu)} samples -> {N}")
    real_imu, sim_imu = real_imu[:N], sim_imu[:N]
    pos = None if pos is None else np.asarray(pos)[:N]
    time = None if time is None else np.asarray(time)[:N]

    n_windows = N // WINDOW_SAMPLES
    if n_windows == 0:
        print(f"Recording too short for a full 10s window ({N} samples)")
        return None
    print(f"  {N} samples ({N / SAMPLE_RATE:.1f}s), {n_windows} consecutive windows")

    seed = args.seed
    print(f"  LDM generate ({bundle['mode']}, seed={seed}) ...")
    gen_imu = generate_ldm_imu(
        args, bundle, real_imu, pos, time,
        sim_imu if bundle["mode"] == "sim_cond" else None, ctx.device, seed,
    )

    # sim and gen render the placement the pair folder names, but the real IMU
    # was captured however the phone happened to be carried that session.
    gt_class = resolve_gt_class(real_path, class_names)
    native_gt_class = infer_native_gt_class(real_path, class_names)
    if native_gt_class != gt_class:
        print(f"  real recording's actual carrying type: {native_gt_class or 'unknown'}"
              f" (folder names {gt_class} for sim/gen)")
    source_gt = {"real": native_gt_class, "sim": gt_class, "gen": gt_class}

    # One mask from the recorded trajectory, shared by all three sources so their
    # confusion matrices are built over exactly the same windows.
    exclusions = compute_window_exclusions(real_imu, pos, time, n_windows)
    sources = {"real": real_imu, "sim": sim_imu, "gen": gen_imu[:N]}
    per_source, window_imus = {}, {}
    for name, imu in sources.items():
        preds, logits, wins = ctx.classify(imu)
        per_source[name] = window_result(preds, logits, class_names)
        window_imus[name] = wins
        print(f"    {name:4s}: majority {per_source[name]['majority_vote']} "
              f"({per_source[name]['majority_count']}/{len(preds)})")

    plot_source_comparison(
        per_source, class_names, outdir, pos=pos, time=time, gt_class=gt_class,
        title=os.path.relpath(real_path, start=os.path.dirname(os.path.dirname(real_path))),
    )

    n_plot = ctx.n_window_plots(n_windows) if plots else 0
    if n_plot > 0:
        plot_dir = os.path.join(outdir, "window_plots")
        os.makedirs(plot_dir, exist_ok=True)
        for i in range(n_plot):
            labels = {
                n: f"{per_source[n]['predictions'][i]} "
                   f"({per_source[n]['confidence'][i]:.2f})"
                for n in sources
            }
            plot_source_overlay({n: window_imus[n][i] for n in sources},
                                plot_dir, i, labels, time_offset=i * WINDOW_SEC)
        print(f"Saved {n_plot} overlay window plots to {plot_dir}")

    def agree(a, b):
        return float(np.mean([x == y for x, y in
                              zip(per_source[a]["predictions"],
                                  per_source[b]["predictions"])]))

    summary = {
        "input": os.path.abspath(real_path),
        "sim_input": os.path.abspath(sim_path),
        "n_windows": n_windows,
        "gt_from_name": gt_class,
        "gt_native_real": native_gt_class,
        "gt_per_source": source_gt,
        "ldm_mode": bundle["mode"],
        "ldm_seed": seed,
        "sources": per_source,
        "window_agreement": {
            "gen_vs_real": agree("gen", "real"),
            "sim_vs_real": agree("sim", "real"),
            "gen_vs_sim": agree("gen", "sim"),
        },
        "majority_votes": {n: per_source[n]["majority_vote"] for n in sources},
        # Each source is scored against the placement it actually represents.
        "window_accuracy_vs_gt": {
            n: float(np.mean([p == source_gt[n] for p in per_source[n]["predictions"]]))
            for n in SOURCE_NAMES if source_gt[n] is not None
        },
        "cm_window_excluded": exclusions.mask,
        "cm_exclusion": exclusions.summary(),
        # Keep the real IMU's verdict as the trajectory-level vote for dir summaries
        "majority_vote": per_source["real"]["majority_vote"],
        "majority_count": per_source["real"]["majority_count"],
        "class_names": list(class_names),
    }
    n_excl = summary["cm_exclusion"]["n_excluded"]
    print(f"  Confusion-matrix filter: {n_windows - n_excl}/{n_windows} windows kept"
          + (f" (dropped {summary['cm_exclusion']['by_reason']})" if n_excl else ""))

    for name in ("source_comparison.json", "results.json"):
        with open(os.path.join(outdir, name), "w") as f:
            json.dump(summary, f, indent=2)

    ag = summary["window_agreement"]
    print(f"  Per-window agreement: gen~real {ag['gen_vs_real']:.3f}, "
          f"sim~real {ag['sim_vs_real']:.3f}, gen~sim {ag['gen_vs_sim']:.3f}")
    return summary


# ---------------------------------------------------------------------------
# Directory modes
# ---------------------------------------------------------------------------

def run_over_inputs(items, outdir, run_one, strict, kind="input"):
    """Evaluate ``items`` of (name, path) into per-item subdirs of ``outdir``.

    Returns (results, failures), where results holds (name, result) for the
    inputs that produced one.
    """
    results, failures = [], []
    for i, (name, path) in enumerate(items):
        print(f"\n[{i + 1}/{len(items)}] {name}")
        try:
            res = run_one(path, os.path.join(outdir, name))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append({kind: name, "error": repr(exc)})
            if strict:
                raise
            continue
        if res is None:
            failures.append({kind: name, "error": "too_short"})
            continue
        results.append((name, res))
    return results, failures


def eval_input_dir(ctx):
    """Run the single-trajectory eval on every hdf5/parquet under --input_dir."""
    args = ctx.args
    # Pair datasets keep their recordings one level down, so fall back to the
    # real.hdf5 walk whenever the top level holds no trajectories.
    files = list_trajectory_files(args.input_dir, recursive_pairs=True)
    if not files:
        raise ValueError(f"No .hdf5 / .parquet files found in {args.input_dir}")
    print(f"  {len(files)} trajectories in {args.input_dir}")

    plots = args.n_plot != 0
    results, failures = run_over_inputs(
        [(output_name(p, args.input_dir), p) for p in files], args.outdir,
        lambda path, subdir: eval_single_trajectory(ctx, path, subdir, plots=plots),
        args.strict,
    )

    per_file, cm_labels, cm_preds = [], [], []
    n_correct = n_labeled = 0
    cm_excluded = ExclusionTally()
    for name, res in results:
        gt = res.get("gt_from_name")
        correct = gt is not None and res["majority_vote"] == gt
        per_file.append({
            "input": res["input"],
            "stem": name,
            "majority_vote": res["majority_vote"],
            "majority_count": res["majority_count"],
            "n_windows": res["n_windows"],
            "gt_from_name": gt,
            "n_noise": res.get("n_noise"),
            "mean_window_agreement": res.get("mean_window_agreement"),
            "correct": correct,
        })
        if gt is None:
            continue
        n_labeled += 1
        n_correct += int(correct)
        keep = kept_window_predictions(res["predictions"],
                                       res.get("cm_window_excluded"))
        cm_labels.extend([gt] * len(keep))
        cm_preds.extend(keep)
        cm_excluded.add(res)

    summary = {
        "input_dir": os.path.abspath(args.input_dir),
        "n_files": len(files),
        "n_evaluated": len(per_file),
        "imu_frame": args.imu_frame,
        "hdf5_already_world": args.hdf5_already_world,
        "n_noise": args.n_noise if ctx.ldm_bundle else 1,
        "ldm_mode": None if ctx.ldm_bundle is None else ctx.ldm_bundle["mode"],
        "failures": failures,
        "files": per_file,
    }
    if n_labeled:
        summary["majority_accuracy_vs_name"] = n_correct / n_labeled
        summary["n_labeled_from_name"] = n_labeled
        print(f"\nMajority accuracy vs filename class: "
              f"{n_correct}/{n_labeled} = {n_correct / n_labeled:.3f}")

    cm = build_window_confusion(
        cm_labels, cm_preds, ctx.class_names, args.outdir, "confusion_matrix.png",
        title="Per-window confusion (calibration / standing removed)",
    )
    if cm is not None:
        cm.update(cm_excluded.as_dict())
        summary["window_confusion_matrix"] = cm
        print(f"Per-window accuracy over {cm['n_windows']} kept windows: "
              f"{cm['accuracy']:.3f} ({cm['n_excluded']} windows dropped, "
              f"{cm['by_reason']})")

    return _write_dir_summary(summary, args.outdir, "directory summary")


def compare_input_dir(ctx):
    """Run the real/sim/gen comparison on every pair under --input_dir."""
    args = ctx.args
    pairs = list_pair_dirs(args.input_dir)
    if not pairs:
        raise ValueError(f"No pair directories with real.hdf5 under {args.input_dir}")
    print(f"  {len(pairs)} pairs in {args.input_dir}")

    plots = args.n_plot != 0
    results, failures = run_over_inputs(
        [(output_name(p, args.input_dir), p) for p in pairs], args.outdir,
        lambda path, subdir: eval_source_comparison(ctx, path, subdir, plots=plots),
        args.strict, kind="pair",
    )

    per_pair = []
    cm_labels = {n: [] for n in SOURCE_NAMES}
    cm_preds = {n: [] for n in SOURCE_NAMES}
    n_unlabeled = Counter()
    cm_excluded = ExclusionTally()
    for name, res in results:
        per_pair.append({
            "pair": name,
            "gt_from_name": res["gt_from_name"],
            "gt_native_real": res["gt_native_real"],
            "gt_per_source": res["gt_per_source"],
            "n_windows": res["n_windows"],
            "majority_votes": res["majority_votes"],
            "window_agreement": res["window_agreement"],
            "window_accuracy_vs_gt": res["window_accuracy_vs_gt"],
            "cm_exclusion": res["cm_exclusion"],
        })
        # sim and gen are scored against the placement the simulator rendered;
        # real against what the phone was actually doing, which is unknown for
        # some sequences and those are left out of the real matrix.
        for source in SOURCE_NAMES:
            gt = res["gt_per_source"][source]
            if gt is None:
                n_unlabeled[source] += 1
                continue
            keep = kept_window_predictions(res["sources"][source]["predictions"],
                                           res["cm_window_excluded"])
            cm_labels[source].extend([gt] * len(keep))
            cm_preds[source].extend(keep)
        if any(g is not None for g in res["gt_per_source"].values()):
            cm_excluded.add(res)

    summary = {
        "input_dir": os.path.abspath(args.input_dir),
        "n_pairs": len(pairs),
        "n_evaluated": len(per_pair),
        "ldm_mode": ctx.ldm_bundle["mode"],
        "ldm_seed": args.seed,
        "imu_frame": args.imu_frame,
        "failures": failures,
        "pairs": per_pair,
    }
    if per_pair:
        summary["mean_window_agreement"] = {
            k: float(np.mean([p["window_agreement"][k] for p in per_pair]))
            for k in ("gen_vs_real", "sim_vs_real", "gen_vs_sim")
        }
        labeled = {n: [p for p in per_pair if p["gt_per_source"][n] is not None]
                   for n in SOURCE_NAMES}
        summary["mean_window_accuracy_vs_gt"] = {
            n: float(np.mean([p["window_accuracy_vs_gt"][n] for p in ps]))
            for n, ps in labeled.items() if ps
        }
        summary["majority_accuracy_vs_gt"] = {
            n: float(np.mean([p["majority_votes"][n] == p["gt_per_source"][n]
                              for p in ps]))
            for n, ps in labeled.items() if ps
        }
        summary["n_labeled"] = {n: len(ps) for n, ps in labeled.items()}

    matrices = {}
    for name in SOURCE_NAMES:
        label = "actual carrying type" if name == "real" else "rendered carrying type"
        cm = build_window_confusion(
            cm_labels[name], cm_preds[name], ctx.class_names, args.outdir,
            f"confusion_matrix_{name}.png",
            title=f"Per-window confusion — {name} vs {label}",
        )
        if cm is not None:
            cm["n_pairs_unlabeled"] = n_unlabeled[name]
            matrices[name] = cm
    if matrices:
        summary["window_confusion_matrix"] = matrices
        summary["window_confusion_excluded"] = cm_excluded.as_dict()
        print(f"\nPer-window accuracy ({cm_excluded.n_excluded} calibration / "
              f"standing windows dropped):")
        for name, cm in matrices.items():
            skipped = (f", {cm['n_pairs_unlabeled']} pairs skipped for unknown "
                       f"carrying type" if cm["n_pairs_unlabeled"] else "")
            print(f"  {name:4s} {cm['accuracy']:.3f} over {cm['n_windows']} "
                  f"windows{skipped}")

    if per_pair:
        ag = summary["mean_window_agreement"]
        print(f"\nMean per-window agreement over {len(per_pair)} pairs: "
              f"gen~real {ag['gen_vs_real']:.3f}, sim~real {ag['sim_vs_real']:.3f}, "
              f"gen~sim {ag['gen_vs_sim']:.3f}")
        acc = summary["majority_accuracy_vs_gt"]
        if acc:
            print("Majority accuracy vs GT: "
                  + ", ".join(f"{n} {acc[n]:.3f}" for n in SOURCE_NAMES if n in acc))
    return _write_dir_summary(summary, args.outdir, "comparison summary")


def _write_dir_summary(summary, outdir, what):
    path = os.path.join(outdir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {what} to {path}")
    return summary


# ---------------------------------------------------------------------------
# Labeled split mode
# ---------------------------------------------------------------------------

def _labeled_split_confusion(args, files, all_labels, all_preds, class_names):
    """Confusion matrix over the split's windows, minus calibration/standing.

    CarryingTypeDataset emits each file's overlapping windows in order, so
    repeating that windowing per file lines the mask up with the predictions.
    """
    window_samples = int(args.window_sec * SAMPLE_RATE)
    stride_samples = int(args.stride_sec * SAMPLE_RATE)

    mask, reasons = [], Counter()
    for path, _cls_name in files:
        exclusions = compute_hdf5_window_exclusions(
            path, window_samples, stride_samples, SAMPLE_RATE
        )
        mask.extend(exclusions.mask)
        reasons.update(r for w in exclusions.windows if w["excluded"]
                       for r in w["reasons"])

    if len(mask) != len(all_preds):
        print(f"  [warn] exclusion mask covers {len(mask)} windows but the split "
              f"has {len(all_preds)}; skipping the filtered confusion matrix")
        return None

    keep = ~np.asarray(mask, dtype=bool)
    if not keep.any():
        return None

    cm = build_window_confusion(
        [class_names[i] for i in all_labels[keep]],
        [class_names[i] for i in all_preds[keep]],
        class_names, args.outdir, "confusion_matrix_filtered.png",
        title="Per-window confusion (calibration / standing removed)",
    )
    cm["n_excluded"] = int((~keep).sum())
    cm["excluded_by_reason"] = dict(reasons)
    print(f"Filtered accuracy over {cm['n_windows']} kept windows: "
          f"{cm['accuracy']:.4f} ({cm['n_excluded']} windows dropped)")
    return cm


def eval_labeled_split(ctx):
    """Evaluate on the labeled train/val split with overlapping windows."""
    args = ctx.args
    class_names = ctx.class_names
    train_files, val_files = split_files_stratified(
        build_file_list(args.data_dir), val_fraction=args.val_fraction, seed=args.seed
    )
    files = train_files if args.split == "train" else val_files
    print(f"  {len(files)} files in {args.split} split")

    ds = CarryingTypeDataset(
        files, ctx.imu_mean, ctx.imu_std,
        window_sec=args.window_sec, sample_rate=SAMPLE_RATE,
        latent_length=args.latent_length, stride_sec=args.stride_sec,
        imu_frame=args.imu_frame,
    )
    print(f"  {len(ds)} windows (imu_frame={args.imu_frame})")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Windows are only kept around for the plots, so stop hoarding IMU once the
    # --n_plot budget is covered (a full split does not fit comfortably in RAM).
    plot_budget = None if args.n_plot < 0 else max(args.n_plot, 0)
    all_preds, all_labels, plot_imu = [], [], []
    n_kept = 0
    with torch.no_grad():
        for batch in loader:
            logits = ctx.clf(batch['imu'].float().to(ctx.device))
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(batch['label'])
            if plot_budget is None or n_kept < plot_budget:
                take = batch['imu'] if plot_budget is None \
                    else batch['imu'][:plot_budget - n_kept]
                plot_imu.append(take)
                n_kept += len(take)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    plot_imu = torch.cat(plot_imu) if plot_imu else torch.empty(0)

    acc = float((all_preds == all_labels).mean())
    print(f"\n=== {args.split} Results ({len(all_preds)} windows) ===")
    print(f"Overall accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names, args.outdir)

    filtered = _labeled_split_confusion(args, files, all_labels, all_preds, class_names)

    n_plot = len(plot_imu)
    if n_plot:
        plot_dir = os.path.join(args.outdir, "window_plots")
        os.makedirs(plot_dir, exist_ok=True)
        for i in range(n_plot):
            plot_imu_window(plot_imu[i].numpy(), plot_dir, i,
                            gt_label=class_names[all_labels[i]],
                            pred_label=class_names[all_preds[i]])
        print(f"Saved {n_plot} window plots to {plot_dir}")

    results = {
        "split": args.split,
        "n_windows": len(all_preds),
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
    if filtered is not None:
        results["window_confusion_matrix"] = filtered
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate carrying-type classifier")
    parser.add_argument("--config", type=str, default="configs/imu/vae_1d.yaml")
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--clf_ckpt", type=str, required=True,
                        help="Classifier checkpoint (best_classifier.pt)")
    parser.add_argument("--input", type=str, default=None,
                        help="Single .hdf5/.parquet, or a pair directory "
                             "containing real.hdf5 (+ synthetic.parquet)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of .hdf5 / .parquet trajectories "
                             "(runs single-traj eval on each)")
    parser.add_argument(
        "--imu_frame", type=str, default=None, choices=["local", "world"],
        help="IMU frame fed to the classifier: 'local' (device) or 'world' "
             "(HDF5 via game_rv; parquet *_world_* columns). "
             "Defaults to imu_frame stored in --clf_ckpt, else 'local'.",
    )
    parser.add_argument(
        "--hdf5_already_world", action="store_true",
        help="HDF5 synced/acce|gyro are already world-frame (e.g. gen_world "
             "LDM exports). Skip game_rv. Use with --imu_frame world.",
    )
    parser.add_argument(
        "--parquet_from_world", action="store_true",
        help="With --imu_frame local, rotate parquet *_world_* → local via "
             "phone_rot instead of reading *_local_* (for LDM-generated parquet).",
    )
    parser.add_argument(
        "--world_heading", type=float, default=0.0,
        help="Simulator world_heading_offset (rad) when rotating parquet "
             "world→local (--parquet_from_world)",
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Labeled data root with class subfolders")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="outputs/carrying_eval")
    parser.add_argument("--n_plot", type=int, default=8,
                        help="Window plots per trajectory (-1 = all, 0 = none)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--stride_sec", type=float, default=2.0)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--latent_length", type=int, default=100)
    parser.add_argument("--train_results", type=str, default=None,
                        help="Path to training results.json (with history). "
                             "Defaults to results.json next to --clf_ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ldm_ckpt", type=str, default=None,
        help="LDM checkpoint; if set, generate IMU before classifying",
    )
    parser.add_argument(
        "--ldm_config", type=str, default=None,
        help="LDM yaml (default: ldm_1d.yaml or ldm_1d_sim_cond.yaml)",
    )
    parser.add_argument(
        "--ldm_vae_ckpt", type=str, default=None,
        help="VAE checkpoint used as the LDM first stage "
             "(defaults to --vae_ckpt, i.e. the classifier's VAE)",
    )
    parser.add_argument(
        "--ldm_stats", type=str, default=None,
        help="LDM stats.pt with imu_mean/std and vel_mean/std "
             "(falls back to --stats)",
    )
    parser.add_argument(
        "--ldm_mode", type=str, default="traj_cond",
        choices=["traj_cond", "sim_cond"],
        help="traj_cond: GT velocity only; sim_cond: also condition on "
             "synthetic IMU (needs --sim_input or synthetic.parquet)",
    )
    parser.add_argument(
        "--sim_input", type=str, default=None,
        help="Synthetic parquet for sim_cond (default: synthetic.parquet "
             "next to the trajectory)",
    )
    parser.add_argument(
        "--n_noise", type=int, default=1,
        help="Independent LDM noise inits (seeds = --seed, --seed+1, ...)",
    )
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="0=VAE recon, (0,1)=img2img from encoded IMU, >=1=from noise",
    )
    parser.add_argument(
        "--ldm_scale_factor", type=float, default=None,
        help="Override LDM scale_factor (otherwise from checkpoint)",
    )
    parser.add_argument(
        "--no_ema", action="store_true",
        help="Disable EMA weights for LDM sampling",
    )
    parser.add_argument(
        "--compare_sources", action="store_true",
        help="Classify real, synthetic, and LDM-generated IMU for the same "
             "trajectory and plot them overlayed with per-window predictions. "
             "Needs --ldm_ckpt and a pair dir/file with a synthetic.parquet.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="With --input_dir, abort on the first trajectory that fails "
             "instead of recording it and continuing",
    )
    return parser


def validate_args(args):
    if args.input is None and args.input_dir is None and args.data_dir is None:
        raise ValueError("Provide --input, --input_dir, or --data_dir (labeled split)")

    if args.compare_sources:
        if not args.ldm_ckpt:
            raise ValueError("--compare_sources needs --ldm_ckpt to generate IMU")
        if args.input is None and args.input_dir is None:
            raise ValueError("--compare_sources needs --input or --input_dir")
        if args.n_noise > 1:
            raise ValueError(
                "--compare_sources uses one LDM sample per trajectory; "
                "drop --n_noise (run the noise sweep separately)"
            )

    if not args.ldm_ckpt:
        if args.n_noise > 1:
            raise ValueError("--n_noise > 1 only applies to LDM generation; pass --ldm_ckpt")
        return

    if args.input is None and args.input_dir is None:
        raise ValueError("--ldm_ckpt is only used with --input / --input_dir")
    if args.n_noise < 1:
        raise ValueError("--n_noise must be >= 1")
    if args.n_noise > 1 and args.strength == 0.0:
        raise ValueError(
            "--strength 0 is a deterministic VAE reconstruction, so every "
            "noise init would be identical. Use --strength > 0 with --n_noise > 1."
        )
    if args.sim_input and args.input_dir is not None:
        raise ValueError(
            "--sim_input is a single file and would condition every trajectory "
            "in --input_dir on it. Drop it so each trajectory uses its own "
            "synthetic.parquet."
        )


def main():
    args = build_parser().parse_args()
    validate_args(args)

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading VAE ...")
    vae = load_vae(args.config, args.vae_ckpt, device)

    print("Loading classifier ...")
    mlp, class_names, clf_ckpt = load_classifier(args.clf_ckpt, device)

    if args.imu_frame is None:
        args.imu_frame = clf_ckpt.get("imu_frame", "local")
    print(f"  imu_frame: {args.imu_frame}")

    train_results = resolve_train_results(args.train_results, args.clf_ckpt)
    if train_results is not None:
        print(f"Plotting training curves from {train_results}")
        plot_training_curves(train_results, args.outdir)
    else:
        print("No training results.json found; skipping training curves "
              "(pass --train_results)")

    # Prefer the stats embedded in the classifier checkpoint, then the VAE / --stats
    if 'imu_mean' in clf_ckpt and 'imu_std' in clf_ckpt:
        imu_mean = clf_ckpt['imu_mean'].view(-1, 1)
        imu_std = clf_ckpt['imu_std'].view(-1, 1)
        print("  Using stats from classifier checkpoint")
    else:
        imu_mean, imu_std = resolve_stats(model=vae, stats_path=args.stats)
        print("  Using stats from VAE / --stats")

    clf = CarryingTypeClassifier(vae, mlp)
    clf.eval()
    ctx = EvalContext(
        args, clf, imu_mean, imu_std, device, class_names,
        ldm_bundle=load_ldm_bundle(args, device) if args.ldm_ckpt else None,
    )

    if args.compare_sources:
        if args.imu_frame != "world":
            print("  [warn] sim/LDM IMU is world-frame; --imu_frame is "
                  f"'{args.imu_frame}'. Prefer --imu_frame world.")
        if args.input is not None:
            print(f"\nComparing real/sim/gen for: {args.input}")
            eval_source_comparison(ctx)
        else:
            print(f"\nComparing real/sim/gen over: {args.input_dir}")
            compare_input_dir(ctx)
    elif args.input is not None:
        print(f"\nEvaluating single trajectory: {args.input}")
        eval_single_trajectory(ctx)
    elif args.input_dir is not None:
        print(f"\nEvaluating directory: {args.input_dir}")
        eval_input_dir(ctx)
    else:
        print(f"\nEvaluating labeled {args.split} split from {args.data_dir}")
        eval_labeled_split(ctx)

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
