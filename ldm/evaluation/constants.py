"""Sampling rates, window sizes, channel orders and filesystem conventions.

Two IMU channel orders coexist in this codebase:

* ``CHANNEL_NAMES`` -- ``[accel(3), gyro(3)]``. Used by the VAE, the LDM, the
  carrying classifier and every preprocessed ``.pt`` window.
* ``RONIN_CHANNEL_NAMES`` -- ``[gyro(3), accel(3)]``. Used by RoNIN and by the
  sequence loaders that feed it.

:func:`ldm.evaluation.windows.swap_imu_channels` converts between them.
"""

import os.path as osp

SAMPLE_RATE = 200
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
LATENT_LENGTH = 100

CHANNEL_NAMES = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
RONIN_CHANNEL_NAMES = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]

HDF5_EXTS = (".hdf5", ".h5")
PARQUET_EXTS = (".parquet",)
TRAJECTORY_EXTS = HDF5_EXTS + PARQUET_EXTS

REAL_NAME = "real.hdf5"
SYNTHETIC_NAME = "synthetic.parquet"
TRAJECTORY_NAME = "trajectory.txt"
MANIFEST_NAME = "manifest.csv"
STATS_NAME = "stats.pt"

ACCEL_LOCAL_COLS = ["accel_local_x", "accel_local_y", "accel_local_z"]
GYRO_LOCAL_COLS = ["gyro_local_x", "gyro_local_y", "gyro_local_z"]
ACCEL_WORLD_COLS = ["accel_world_x", "accel_world_y", "accel_world_z"]
GYRO_WORLD_COLS = ["gyro_world_x", "gyro_world_y", "gyro_world_z"]
PHONE_ROT_COLS = ["phone_rot_x", "phone_rot_y", "phone_rot_z", "phone_rot_w"]

REPO_ROOT = osp.normpath(osp.join(osp.dirname(__file__), "..", ".."))

# Default location of the juxta-ronin checkout, which supplies the RoNIN
# network definitions (source/model_resnet1d*.py). This assumes juxta-ronin is
# a SIBLING of this repo; override with --ronin_root or $JUXTA_RONIN_ROOT.
# Only needed when a script is given --ronin_ckpt.
DEFAULT_RONIN_ROOT = osp.normpath(osp.join(REPO_ROOT, "..", "juxta-ronin"))

# Anchored to the repo so scripts resolve their defaults from any working
# directory once the package is installed with `pip install -e .`.
DEFAULT_VAE_CONFIG = osp.join(REPO_ROOT, "configs/imu/vae_1d.yaml")
DEFAULT_LDM_CONFIGS = {
    "traj": osp.join(REPO_ROOT, "configs/imu/ldm_1d.yaml"),
    "sim_cond": osp.join(REPO_ROOT, "configs/imu/ldm_1d_sim_cond.yaml"),
}

# ---------------------------------------------------------------------------
# Carrying placements
#
# This is DATASET vocabulary: how a recording was carried, as spelled in
# dataset paths. It is deliberately decoupled from what any given classifier
# predicts (ldm.data.carrying_labels.CLASS_NAMES, or a checkpoint's embedded
# class_names), so new placements can be recorded, preprocessed, evaluated and
# generated with the VAE/LDM long before a classifier is trained on them.
#
# Adding a placement is a one-line edit here. Directories under a labeled
# dataset root do not even need to be listed: an unrecognised folder name is
# normalized and used as-is (see ldm.evaluation.paths.normalize_placement).
# ---------------------------------------------------------------------------

CARRYING_PLACEMENTS = ("chest", "demo_hand", "head", "pocket", "swinging")

# Spelling variants seen in dataset paths, flat gen_world stems and the pair
# manifest (which writes swing_left where paths write left_swing). Canonical
# names map to themselves so this table cannot drift from CARRYING_PLACEMENTS.
CARRYING_ALIASES = {
    **{name: name for name in CARRYING_PLACEMENTS},
    "demohand": "demo_hand",
    "hand": "demo_hand",
    "left_pocket": "pocket",
    "right_pocket": "pocket",
    "left_swing": "swinging",
    "right_swing": "swinging",
    "swing_left": "swinging",
    "swing_right": "swinging",
    "swing": "swinging",
}

# Longest alias first, so "left_swing_..." resolves as a swing rather than
# falling through to a shorter alias.
CARRYING_ALIASES_BY_LENGTH = sorted(CARRYING_ALIASES, key=len, reverse=True)
