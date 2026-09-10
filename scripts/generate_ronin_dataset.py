#!/usr/bin/env python3
"""Generate a RoNIN ``gen_world`` HDF5 dataset with the LDM.

Each pair folder becomes one flat ``<pair_id>.hdf5`` holding the original
trajectory and orientation next to freshly generated world-frame IMU, plus
``all.txt`` / ``train.txt`` / ``val.txt`` for ``ronin_resnet.py --dataset
gen_world``.

    --mode sim_cond   condition on trajectory velocity and the synthetic IMU
                      latent; needs trajectory.txt and synthetic.parquet, and
                      real.hdf5 only to keep a sub-window tail of real IMU
    --mode traj       condition on trajectory velocity alone, taken from
                      real.hdf5's tango_pos

Examples:
    python scripts/generate_ronin_dataset.py --mode sim_cond \\
        --input_dir data/real_sim_paired_data_ldm \\
        --ldm_ckpt logs/ldm_1d_sim_cond/ldm_world_sim_cond/checkpoints/last.ckpt \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --ldm_stats data/real_sim_paired_data_processed_ldm_training/stats.pt \\
        --outdir data/ronin_ldm_gen_world

    python scripts/generate_ronin_dataset.py --mode traj \\
        --input_dir data/real_sim_paired_data_ldm \\
        --ldm_ckpt logs/ldm_1d/ldm_world_full_dataset_velbugfix/checkpoints/last.ckpt \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --ldm_stats data/dataset_full_processed_velbugfix/stats.pt \\
        --outdir data/ronin_ldm_gen_world_traj
"""

import argparse
import csv
import os
import os.path as osp
from dataclasses import dataclass


from ldm.evaluation import args as eval_args
from ldm.evaluation import generate, models, paths, report, sequences, stats
from ldm.evaluation.constants import REAL_NAME, SYNTHETIC_NAME, TRAJECTORY_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    """One pair folder's inputs, resolved to absolute paths or None."""

    name: str
    pair_dir: str
    trajectory: str = None
    synthetic: str = None
    real: str = None

    @property
    def stem(self):
        """Flat basename for the generated HDF5 and the split lists."""
        return self.name.replace("/", "_").replace("\\", "_")

    def ready(self, mode):
        if mode == "traj":
            return self.real is not None
        return self.trajectory is not None and self.synthetic is not None

    def why_not_ready(self, mode):
        needed = ("real.hdf5",) if mode == "traj" else (TRAJECTORY_NAME, SYNTHETIC_NAME)
        have = {TRAJECTORY_NAME: self.trajectory, SYNTHETIC_NAME: self.synthetic,
                "real.hdf5": self.real}
        return [name for name in needed if have[name] is None]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate world-frame HDF5s for ronin_resnet.py --dataset gen_world"
    )
    eval_args.add_input_args(parser, dataset_dir=False)
    eval_args.add_ldm_args(parser)
    eval_args.add_vae_args(parser, with_stats=False)
    eval_args.add_sampling_args(parser, strength=False)
    eval_args.add_windowing_args(parser, stride=True)
    eval_args.add_output_args(parser, default_outdir="data/ronin_ldm_gen_world",
                              plots=False)
    eval_args.add_batch_args(parser, plots_only=False)
    parser.add_argument("--sim_dir", type=str, default=None,
                        help="Mirrored root holding <pair_id>/synthetic.parquet, when "
                             "the synthetic IMU lives outside --input_dir")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Fraction of generated sequences listed in val.txt")
    parser.add_argument("--trim_tail", action="store_true",
                        help="Drop a remainder shorter than one window instead of "
                             "keeping real IMU there; always applied without real.hdf5")
    # Whole windows are decoded at once here, unlike the per-window eval scripts
    parser.set_defaults(batch_size=8)
    return parser


def validate(parser, args):
    """Cross-argument checks; returns (source, window, stride) in samples."""
    source = eval_args.validate_inputs(parser, args, allow_dataset_dir=False)
    if not 0 <= args.val_fraction < 1:
        parser.error("--val_fraction must be in [0, 1)")
    if args.sim_dir is not None and args.mode != "sim_cond":
        parser.error("--sim_dir only applies to --mode sim_cond")
    window = int(round(args.window_sec * args.sample_rate))
    stride = window if args.stride_sec is None \
        else int(round(args.stride_sec * args.sample_rate))
    if window <= 0 or stride <= 0 or stride > window:
        parser.error("require 0 < --stride_sec <= --window_sec")
    return source, window, stride


def _first_file(*candidates):
    return next((c for c in candidates if c is not None and osp.isfile(c)), None)


def build_pair(pair_dir, name, sim_dir=None, row=None, repo_root=None):
    """Resolve one pair folder's trajectory, synthetic and real inputs."""
    row = row or {}

    def from_manifest(key):
        value = row.get(key)
        if not value:
            return None
        if osp.isabs(value):
            return _first_file(value)
        return _first_file(osp.join(repo_root or pair_dir, value),
                           osp.join(pair_dir, osp.basename(value)))

    mirrored = None if sim_dir is None else osp.join(sim_dir, name)
    return Pair(
        name=name,
        pair_dir=pair_dir,
        trajectory=_first_file(from_manifest("trajectory_txt"),
                               osp.join(pair_dir, TRAJECTORY_NAME)),
        synthetic=_first_file(
            None if mirrored is None else osp.join(mirrored, SYNTHETIC_NAME),
            None if mirrored is None
            else osp.join(mirrored, ".run", "synthetic_0000.parquet"),
            from_manifest("synthetic_parquet"),
            osp.join(pair_dir, SYNTHETIC_NAME),
            osp.join(pair_dir, ".run", "synthetic_0000.parquet"),
        ),
        real=_first_file(from_manifest("real_hdf5"), osp.join(pair_dir, REAL_NAME)),
    )


def discover_pairs(args, source):
    """Every candidate pair, from one ``--input`` folder or a whole root.

    Pairs are listed even when inputs are missing, because ``--mode sim_cond``
    tolerates a missing ``real.hdf5`` while ``--mode traj`` requires it, and the
    report of what was skipped is more useful than a silent omission.
    """
    if source == "input":
        pair_dir = osp.abspath(args.input.rstrip("/"))
        if osp.isfile(pair_dir):
            pair_dir = osp.dirname(pair_dir)
        if not osp.isdir(pair_dir):
            raise NotADirectoryError(f"--input is not a pair directory: {args.input}")
        return [build_pair(pair_dir, osp.basename(pair_dir), args.sim_dir)]

    root = osp.abspath(args.input_dir.rstrip("/"))
    manifest = osp.join(root, "manifest.csv")
    if osp.isfile(manifest):
        with open(manifest, newline="") as f:
            rows = [row for row in csv.DictReader(f)
                    if row.get("status", "").strip() == "generated"]
        repo_root = osp.dirname(osp.dirname(root))
        return [build_pair(osp.join(root, row["pair_id"].strip()),
                           row["pair_id"].strip(), args.sim_dir, row, repo_root)
                for row in sorted(rows, key=lambda row: row["pair_id"])]
    pair_dirs = [current for current, _dirs, files in os.walk(root)
                 if REAL_NAME in files or TRAJECTORY_NAME in files]
    if not pair_dirs:
        raise FileNotFoundError(f"No pair directories found under {root}")
    return [build_pair(d, osp.relpath(d, root), args.sim_dir) for d in sorted(pair_dirs)]


def report_skipped(pairs, mode, strict):
    """Split candidates into runnable pairs and a report of the rest."""
    runnable = [p for p in pairs if p.ready(mode)]
    skipped = [{"pair": p.name, "pair_dir": p.pair_dir, "missing": p.why_not_ready(mode)}
               for p in pairs if not p.ready(mode)]
    print(f"Mode={mode}. {len(pairs)} candidate pairs, {len(runnable)} runnable")
    if skipped:
        print(f"Missing inputs for {len(skipped)} pairs:")
        for item in skipped[:10]:
            print(f"  {item['pair']}: missing {', '.join(item['missing'])}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
        if strict:
            raise FileNotFoundError("missing pair inputs (see the report above)")
    return runnable, skipped


# ---------------------------------------------------------------------------
# Main flow, in call order
# ---------------------------------------------------------------------------

def load_model(args, device):
    """Load the LDM, its DDIM sampler and the statistics it was trained with."""
    config_path = eval_args.resolve_ldm_config(args)
    print(f"Loading LDM ({args.mode}) from {args.ldm_ckpt}\n  config: {config_path}")
    model, sampler, _ = models.load_ldm(
        config_path, args.ldm_ckpt, device,
        vae_ckpt=args.vae_ckpt, scale_factor=args.scale_factor,
    )
    ldm_stats = stats.require_velocity(
        stats.resolve_stats(args.ldm_stats, None, what="ldm_stats")
    )
    print(f"  scale_factor={float(model.scale_factor)}\n  stats: {ldm_stats['path']}")
    return config_path, model, sampler, ldm_stats


def load_pair_inputs(pair, mode):
    """Trajectory, orientation and IMU arrays for one pair, per ``--mode``."""
    if mode == "traj":
        time, position, orientation, real_imu = sequences.load_hdf5_world(pair.real)
        return time, position, orientation, None, real_imu
    return sequences.load_pair_sequence(pair.trajectory, pair.synthetic, pair.real)


def generate_pair(args, pair, model, sampler, ldm_stats, device, window, stride,
                  config_path, out_path):
    """Generate one sequence and write its gen_world HDF5."""
    generate.seed_everything(args.seed)
    time, position, orientation, sim_imu, real_imu = load_pair_inputs(pair, args.mode)
    generated = generate.ldm_generate_dataset(
        model, sampler, ldm_stats, time, position, sim_imu, real_imu, device,
        window, stride, args.latent_length, args.batch_size,
        args.ddim_steps, args.ddim_eta, not args.no_ema, args.trim_tail,
    )
    n = len(generated)
    has_real = pair.real is not None
    metadata = {
        "pair_id": pair.name,
        "output_stem": pair.stem,
        "mode": args.mode,
        "generator": "trajectory-conditioned LDM" if args.mode == "traj"
        else "sim-conditioned LDM",
        "synthetic_source": None if args.mode == "traj" else osp.abspath(pair.synthetic),
        "real_source": osp.abspath(pair.real) if has_real else None,
        "trajectory_source": osp.abspath(pair.real if args.mode == "traj"
                                         else pair.trajectory),
        "ldm_config": osp.abspath(config_path),
        "ldm_ckpt": osp.abspath(args.ldm_ckpt),
        "vae_ckpt": osp.abspath(args.vae_ckpt),
        "ldm_stats": ldm_stats["path"],
        "seed": args.seed,
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
        "window_sec": args.window_sec,
        "stride_sec": window / args.sample_rate if args.stride_sec is None
        else args.stride_sec,
        "tail_mode": "real" if has_real and not args.trim_tail else "trim",
    }
    report.write_gen_world_hdf5(out_path, time[:n], position[:n], orientation[:n],
                                generated, metadata)
    return n


def generate_all(args, pairs, model, sampler, ldm_stats, device, window, stride,
                 config_path):
    """Generate every runnable pair, skipping the ones already on disk."""
    generated, failures = [], []
    for index, pair in enumerate(pairs, 1):
        out_path = osp.join(args.outdir, f"{pair.stem}.hdf5")
        print(f"[{index}/{len(pairs)}] {pair.name} -> {pair.stem}.hdf5")

        if not args.force and osp.isfile(out_path) and report.gen_world_is_valid(out_path):
            print("    [skip] already complete")
            generated.append(pair.stem)
            continue

        try:
            n = generate_pair(args, pair, model, sampler, ldm_stats, device,
                              window, stride, config_path, out_path)
            print(f"    wrote {n} samples")
            generated.append(pair.stem)
        except Exception as exc:  # one bad pair should not sink the dataset
            print(f"    FAILED: {exc}")
            failures.append({"pair": pair.name, "error": repr(exc)})
            if args.strict:
                raise
    return generated, failures


def write_summary(args, config_path, generated, skipped, failures):
    """Split lists plus ``generation_summary.json`` describing the run."""
    n_train, n_val = report.write_lists(args.outdir, generated, args.val_fraction,
                                        args.seed)
    report.write_json({
        "mode": args.mode,
        "outdir": osp.abspath(args.outdir),
        "ldm_config": osp.abspath(config_path),
        "ldm_ckpt": osp.abspath(args.ldm_ckpt),
        "vae_ckpt": osp.abspath(args.vae_ckpt),
        "seed": args.seed,
        "n_generated": len(generated),
        "train_sequences": n_train,
        "val_sequences": n_val,
        "generated": sorted(generated),
        "skipped": skipped,
        "failures": failures,
    }, args.outdir, "generation_summary.json", what="Generation summary")
    print(f"\nDone: {len(generated)} HDF5s ({n_train} train, {n_val} val), "
          f"{len(failures)} failed, {len(skipped)} skipped")


def main():
    parser = build_parser()
    args = parser.parse_args()
    source, window, stride = validate(parser, args)

    pairs = discover_pairs(args, source)
    runnable, skipped = report_skipped(pairs, args.mode, args.strict)
    if args.dry_run:
        for pair in runnable:
            print(f"  [dry-run] {pair.name} -> {pair.stem}.hdf5")
        return
    if not runnable:
        raise FileNotFoundError(
            f"No runnable pairs for --mode {args.mode}. sim_cond needs "
            f"{TRAJECTORY_NAME} and {SYNTHETIC_NAME} per pair ({REAL_NAME} optional); "
            f"traj needs {REAL_NAME}."
        )

    os.makedirs(args.outdir, exist_ok=True)
    generate.seed_everything(args.seed)
    device = eval_args.resolve_device(args)
    print(f"Device: {device}")

    config_path, model, sampler, ldm_stats = load_model(args, device)
    generated, failures = generate_all(args, runnable, model, sampler, ldm_stats,
                                       device, window, stride, config_path)
    write_summary(args, config_path, generated, skipped, failures)


if __name__ == "__main__":
    main()
