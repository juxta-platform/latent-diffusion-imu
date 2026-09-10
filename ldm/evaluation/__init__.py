"""Shared building blocks for the IMU evaluation and generation scripts.

Submodules are imported explicitly (``from ldm.evaluation import models``) so
that lightweight consumers such as :mod:`ldm.data.carrying_dataset` can pull in
:mod:`ldm.evaluation.sequences` without dragging in torch models or matplotlib.

Layout:
    constants   sampling / windowing / channel-order / carrying-placement definitions
    sequences   path -> windows: recording loaders, frame conversion, window datasets
    windows     tensor ops on windows already in memory: channel order,
                standardization, conditioning
    stats       stats.pt resolution for --vae_stats / --ldm_stats
    paths       input discovery and classification, output naming
    args        shared argparse groups and cross-argument validation
    models      VAE / LDM / classifier / RoNIN checkpoint loading
    generate    VAE reconstruction and LDM generation
    ronin       strided windowing, RoNIN inference, trajectory reconstruction
    metrics     ATE/RTE, reconstruction and classification metrics
    plots       every figure produced by the scripts
    report      JSON/CSV/NPZ/HDF5/parquet writers and split lists
    batch       shared directory runner with resume and failure capture
"""

from ldm.evaluation.constants import (
    CHANNEL_NAMES,
    HDF5_EXTS,
    LATENT_LENGTH,
    RONIN_CHANNEL_NAMES,
    SAMPLE_RATE,
    TRAJECTORY_EXTS,
    WINDOW_SAMPLES,
    WINDOW_SEC,
)

__all__ = [
    "CHANNEL_NAMES",
    "HDF5_EXTS",
    "LATENT_LENGTH",
    "RONIN_CHANNEL_NAMES",
    "SAMPLE_RATE",
    "TRAJECTORY_EXTS",
    "WINDOW_SAMPLES",
    "WINDOW_SEC",
]
