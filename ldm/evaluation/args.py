"""Shared argparse groups so every script speaks the same vocabulary.

Names mean the same thing everywhere: ``--vae_*`` always refers to the
autoencoder, ``--ldm_*`` to the diffusion model, ``--ronin_*`` to the inertial
navigation network, ``--input``/``--input_dir`` to raw recordings, and
``--dataset_dir`` to a preprocessed ``.pt`` window dataset.
"""

import torch

from ldm.evaluation.constants import (
    DEFAULT_LDM_CONFIGS,
    DEFAULT_RONIN_ROOT,
    DEFAULT_VAE_CONFIG,
    LATENT_LENGTH,
    SAMPLE_RATE,
)


def add_input_args(parser, dataset_dir=True, pair=False, split=None):
    """``--input`` / ``--input_dir`` and, optionally, ``--dataset_dir``+``--split``.

    ``--split`` follows ``dataset_dir`` unless a script selects a split some
    other way, as ``eval_classifier.py`` does for class-labeled directories.
    """
    group = parser.add_argument_group("input (provide exactly one)")
    group.add_argument("--input", type=str, default=None,
                       help="One .hdf5/.parquet recording, or a pair directory "
                            "containing real.hdf5")
    group.add_argument("--input_dir", type=str, default=None,
                       help="Directory of recordings, pair folders or class "
                            "subfolders; the layout is auto-detected")
    if dataset_dir:
        group.add_argument("--dataset_dir", type=str, default=None,
                           help="Preprocessed dataset dir with train/ val/ and stats.pt")
    if dataset_dir if split is None else split:
        group.add_argument("--split", type=str, default="val", choices=["train", "val"],
                           help="Split to evaluate for --dataset_dir and labeled dirs")
    if pair:
        group.add_argument("--sim_input", type=str, default=None,
                           help="Synthetic parquet override (default: the "
                                "synthetic.parquet beside the recording)")
    return group


def add_vae_args(parser, required=True, with_stats=True):
    group = parser.add_argument_group("VAE")
    group.add_argument("--vae_config", type=str, default=DEFAULT_VAE_CONFIG,
                       help="VAE config; must match the checkpoint architecture")
    group.add_argument("--vae_ckpt", type=str, required=required,
                       help="VAE checkpoint, also used as the LDM first stage")
    if with_stats:
        group.add_argument("--vae_stats", type=str, default=None,
                           help="stats.pt the VAE was trained with "
                                "(default: <dataset_dir>/stats.pt, else ckpt-embedded)")
    return group


def add_ldm_args(parser, required=True, with_mode=True):
    group = parser.add_argument_group("LDM")
    if with_mode:
        group.add_argument("--mode", type=str, default="sim_cond",
                           choices=sorted(DEFAULT_LDM_CONFIGS),
                           help="traj: condition on trajectory velocity only. "
                                "sim_cond: also condition on the synthetic IMU latent")
    group.add_argument("--ldm_config", type=str, default=None,
                       help="LDM config; defaults to the stock config for --mode")
    group.add_argument("--ldm_ckpt", type=str, required=required,
                       help="LDM checkpoint")
    group.add_argument("--ldm_stats", type=str, default=None,
                       help="stats.pt the LDM was trained with; must carry "
                            "vel_mean/vel_std")
    return group


def add_ronin_args(parser, required=False):
    group = parser.add_argument_group("RoNIN")
    group.add_argument("--ronin_ckpt", type=str, required=required, default=None,
                       help="RoNIN checkpoint; enables trajectory ATE/RTE metrics")
    group.add_argument("--ronin_root", type=str, default=None,
                       help="juxta-ronin checkout supplying the network definitions "
                            f"(default: $JUXTA_RONIN_ROOT, else the sibling directory "
                            f"{DEFAULT_RONIN_ROOT})")
    group.add_argument("--ronin_arch", type=str, default="resnet18",
                       choices=["resnet18", "resnet50", "resnet101"])
    group.add_argument("--ronin_window", type=int, default=200,
                       help="RoNIN input window in samples")
    group.add_argument("--ronin_step", type=int, default=10,
                       help="RoNIN window stride in samples")
    group.add_argument("--ronin_3d", action="store_true",
                       help="Use the 3D RoNIN model instead of the 2D one")
    group.add_argument("--rte_delta_sec", type=float, default=10.0,
                       help="Short RTE horizon in seconds, alongside the 60s RTE")
    return group


def add_sampling_args(parser, strength=True, n_noise=False):
    group = parser.add_argument_group("sampling")
    group.add_argument("--ddim_steps", type=int, default=50)
    group.add_argument("--ddim_eta", type=float, default=0.0)
    if strength:
        group.add_argument("--strength", type=float, default=None,
                           help="Trajectory mode only: 0 reconstructs synthetic IMU, "
                                "(0,1) uses synthetic img2img; default 1 generates "
                                "from noise. Not accepted in sim_cond mode")
    if n_noise:
        group.add_argument("--n_noise", type=int, default=1,
                           help="Independent noise inits; seeds are --seed, --seed+1, ...")
    group.add_argument("--no_ema", action="store_true",
                       help="Sample with the raw weights instead of the EMA weights")
    group.add_argument("--scale_factor", type=float, default=None,
                       help="Override the LDM scale_factor stored in the checkpoint")
    return group


def add_windowing_args(parser, stride=False):
    group = parser.add_argument_group("windowing")
    group.add_argument("--window_sec", type=float, default=10.0)
    group.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    group.add_argument("--latent_length", type=int, default=LATENT_LENGTH)
    if stride:
        group.add_argument("--stride_sec", type=float, default=None,
                           help="Window stride in seconds (default: --window_sec, "
                                "i.e. non-overlapping)")
    return group


def add_frame_args(parser, required=False):
    """``--imu_frame`` and the parquet/HDF5 frame overrides.

    An explicit frame overrides checkpoint metadata. Legacy checkpoints
    without frame metadata default to world.
    """
    group = parser.add_argument_group("IMU frame")
    group.add_argument("--imu_frame", type=str, required=required,
                       default=None, choices=["local", "world"],
                       help="Frame fed to the model: local (device) or world "
                            "(HDF5 via game_rv, parquet *_world_* columns). "
                            "Default: checkpoint metadata, then world. LDM is always world")
    group.add_argument("--already_world", "--hdf5_already_world", action="store_true",
                       dest="already_world",
                       help="Force 'HDF5 synced/acce|gyro are already world frame'. "
                            "Only needed for gen_world exports written before the "
                            "imu_frame attribute, which is detected automatically")
    group.add_argument("--parquet_from_world", action="store_true",
                       help="With --imu_frame local, rotate parquet *_world_* to "
                            "local via phone_rot instead of reading *_local_*")
    group.add_argument("--world_heading", type=float, default=0.0,
                       help="Simulator world_heading_offset in radians, used when "
                            "rotating parquet world to local")
    return group


def resolve_imu_frame(parser, args, ckpt, what):
    """Explicit frame, checkpoint metadata, then the world-frame default."""
    from ldm.evaluation.models import checkpoint_imu_frame

    if args.imu_frame is not None:
        return args.imu_frame
    frame = checkpoint_imu_frame(ckpt)
    origin = what if frame is not None else "default; checkpoint has no frame metadata"
    frame = frame or "world"
    print(f"  imu_frame: {frame} (from {origin})")
    return frame


def validate_sampling(parser, args):
    """Resolve traj strength and reject it for pure-noise sim conditioning."""
    strength = getattr(args, "strength", None)
    if args.mode == "sim_cond" and strength is not None:
        parser.error("--strength only applies to --mode traj; sim_cond always generates from noise")
    if strength is not None and not 0 <= strength <= 1:
        parser.error("--strength must be in [0, 1]")
    args.strength = 1.0 if strength is None else strength


def validate_windowing(parser, args):
    if args.window_sec <= 0 or args.sample_rate <= 0 or args.latent_length <= 0:
        parser.error("window_sec, sample_rate and latent_length must be positive")
    if int(args.window_sec * args.sample_rate) < 2:
        parser.error("a window must contain at least two samples")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if getattr(args, "stride_sec", None) is not None and args.stride_sec <= 0:
        parser.error("--stride_sec must be positive")


def add_output_args(parser, default_outdir, plots=True):
    group = parser.add_argument_group("output")
    group.add_argument("--outdir", type=str, default=default_outdir)
    if plots:
        group.add_argument("--n_plot", type=int, default=8,
                           help="Plots per item (-1 = all, 0 = none)")
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--cpu", action="store_true", help="Force CPU inference")
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--num_workers", type=int, default=4)
    return group


def add_export_args(parser, imu=True):
    group = parser.add_argument_group("exports")
    if imu:
        group.add_argument("--save_hdf5", nargs="?", const="", default=None,
                           help="Write generated world-frame IMU into a copy of the "
                                "source HDF5. Bare flag uses <outdir>/<stem>_gen.hdf5")
        group.add_argument("--save_parquet", nargs="?", const="", default=None,
                           help="Write generated world-frame IMU into a copy of the "
                                "source parquet. Bare flag uses "
                                "<outdir>/<stem>_gen.parquet")
    group.add_argument("--save_trajectories", action="store_true",
                       help="Save pos_gt / pos_pred arrays as trajectories.npz")
    return group


def add_batch_args(parser, plots_only=True):
    group = parser.add_argument_group("batch")
    group.add_argument("--force", action="store_true",
                       help="Re-run items whose outputs already exist")
    group.add_argument("--dry_run", action="store_true",
                       help="List the work that would run and exit")
    if plots_only:
        group.add_argument("--plots_only", action="store_true",
                           help="Rebuild figures from existing results without "
                                "re-running")
    group.add_argument("--strict", action="store_true",
                       help="Abort on the first failing item instead of recording it")
    return group


def validate_inputs(parser, args, allow_dataset_dir=True):
    """Require exactly one of the input flags and return which one was given."""
    chosen = [
        name for name in ("input", "input_dir", "dataset_dir")
        if getattr(args, name, None) is not None
    ]
    options = "--input, --input_dir" + (" or --dataset_dir" if allow_dataset_dir else "")
    if len(chosen) != 1:
        parser.error(f"Provide exactly one of {options}")
    return chosen[0]


def resolve_device(args):
    return torch.device(
        "cuda" if torch.cuda.is_available() and not getattr(args, "cpu", False) else "cpu"
    )


def resolve_ldm_config(args):
    """``--ldm_config`` if given, else the stock config for ``--mode``."""
    if getattr(args, "ldm_config", None):
        return args.ldm_config
    return DEFAULT_LDM_CONFIGS[getattr(args, "mode", "traj")]
