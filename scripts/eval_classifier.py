#!/usr/bin/env python3
"""Evaluate the carrying-type classifier.

The base run classifies consecutive 10s windows of each recording. Two
orthogonal flags extend it:

    --ldm_ckpt        generate IMU with the LDM first, then classify it.
                      --n_noise > 1 repeats the generation from independent
                      noise inits and reports per-window agreement.
    --compare_sources classify real, synthetic and generated IMU for the same
                      trajectory and plot them against each other.

Inputs follow the shared vocabulary: ``--input`` for one recording or pair
directory, ``--input_dir`` for a directory. A directory of class subfolders is
recognised as a labeled dataset and evaluated as a stratified ``--split``.

Examples:
    python scripts/eval_classifier.py \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --clf_ckpt logs/carrying_classifier_world_lr5/best_classifier.pt \\
        --input_dir data/hdf5_data --imu_frame world --outdir outputs/clf_eval

    python scripts/eval_classifier.py --compare_sources \\
        --vae_ckpt logs/vae_1d/world_full_dataset/checkpoints/best-064.ckpt \\
        --clf_ckpt logs/carrying_classifier_world_lr5/best_classifier.pt \\
        --ldm_ckpt logs/ldm_1d_sim_cond/ldm_world_sim_cond/checkpoints/best-019.ckpt \\
        --ldm_stats data/real_sim_paired_data_processed_ldm_training/stats.pt \\
        --mode sim_cond --imu_frame world \\
        --input_dir data/real_sim_paired_data_ldm --outdir outputs/clf_compare
"""

import argparse
import os
import os.path as osp
from collections import Counter


import numpy as np
import torch
from torch.utils.data import DataLoader

from ldm.data.carrying_dataset import (
    CarryingTypeDataset, build_file_list, split_files_stratified,
)
from ldm.data.carrying_labels import CARRYING_CLASS_NAMES, derive_window_labels
from ldm.evaluation import args as eval_args
from ldm.evaluation import (
    batch, generate, metrics, models, paths, plots, report, sequences, stats, windows,
)
from ldm.evaluation.constants import SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_SEC
from ldm.evaluation.plots import SOURCE_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate the carrying-type classifier")
    eval_args.add_input_args(parser, dataset_dir=False, pair=True, split=True)
    eval_args.add_vae_args(parser)
    parser.add_argument("--clf_ckpt", type=str, required=True,
                        help="Classifier checkpoint (best_classifier.pt)")
    eval_args.add_ldm_args(parser, required=False)
    eval_args.add_sampling_args(parser, n_noise=True)
    eval_args.add_frame_args(parser, required=False)
    eval_args.add_windowing_args(parser, stride=True)
    eval_args.add_output_args(parser, default_outdir="outputs/classifier_eval")
    eval_args.add_batch_args(parser)
    parser.add_argument("--compare_sources", action="store_true",
                        help="Classify real, synthetic and LDM-generated IMU for the "
                             "same trajectory and plot them overlayed")
    parser.add_argument("--activity_threshold", type=float, default=None,
                        help="Fraction of a window that must be stationary to "
                             "override its carrying label (default: from --clf_ckpt)")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Validation fraction when splitting a labeled directory")
    parser.add_argument("--per_file", action="store_true",
                        help="Classify every recording in a labeled directory "
                             "one at a time instead of scoring a train/val split. "
                             "Includes placements the classifier does not predict")
    parser.add_argument("--train_results", type=str, default=None,
                        help="Classifier training results.json with history "
                             "(default: results.json next to --clf_ckpt)")
    return parser


def validate(parser, args):
    """Cross-argument checks; returns the input flag that was used."""
    source = eval_args.validate_inputs(parser, args, allow_dataset_dir=False)
    eval_args.validate_windowing(parser, args)
    if args.activity_threshold is not None and not 0.0 <= args.activity_threshold < 1.0:
        parser.error("--activity_threshold must be in [0, 1)")

    if args.compare_sources:
        if not args.ldm_ckpt:
            parser.error("--compare_sources needs --ldm_ckpt to generate IMU")
        if args.n_noise > 1:
            parser.error("--compare_sources samples the LDM once per trajectory; "
                         "drop --n_noise and run the sweep separately")
    if not args.ldm_ckpt:
        if args.strength is not None:
            parser.error("--strength requires --ldm_ckpt and --mode traj")
        if args.n_noise > 1:
            parser.error("--n_noise only applies to LDM generation; pass --ldm_ckpt")
        return source

    eval_args.validate_sampling(parser, args)
    if args.n_noise < 1:
        parser.error("--n_noise must be >= 1")
    if args.n_noise > 1 and args.strength == 0.0:
        parser.error("--strength 0 is a deterministic VAE reconstruction, so every "
                     "noise init would be identical; use --strength > 0")
    if args.sim_input and args.input_dir is not None:
        parser.error("--sim_input is a single file and would condition every "
                     "recording in --input_dir on it")
    return source


def classify_windows(imu, ctx):
    """Classify consecutive non-overlapping windows of [N, 6] accel|gyro IMU.

    Returns (predictions, logits, windows) with the windows left unstandardized
    so they can be plotted in physical units.
    """
    stacked = windows.split_imu_windows(imu, ctx.window_samples)
    mean = ctx.imu_mean.view(1, 6, 1).to(ctx.device)
    std = ctx.imu_std.view(1, 6, 1).to(ctx.device)
    batches = []
    with torch.no_grad():
        for start in range(0, stacked.shape[0], ctx.args.batch_size):
            x = torch.from_numpy(stacked[start:start + ctx.args.batch_size]).to(ctx.device)
            batches.append(ctx.clf((x - mean) / std).cpu().numpy())
    logits = np.concatenate(batches, axis=0)
    return [int(p) for p in logits.argmax(1)], logits, stacked


def window_result(predictions, logits, class_names):
    """Per-window predictions, confidences and majority votes, ready for JSON."""
    probs = metrics.softmax(logits)
    majority_idx, majority_count = Counter(predictions).most_common(1)[0]
    carrying = [p for p in predictions if class_names[p] in CARRYING_CLASS_NAMES]
    if carrying:
        carrying_idx, carrying_count = Counter(carrying).most_common(1)[0]
        carrying_vote = class_names[carrying_idx]
    else:
        carrying_vote, carrying_count = None, 0
    return {
        "n_windows": len(predictions),
        "predictions": [class_names[p] for p in predictions],
        "logits": np.asarray(logits).tolist(),
        "softmax": probs.tolist(),
        "confidence": [float(probs[i, p]) for i, p in enumerate(predictions)],
        "majority_vote": class_names[majority_idx],
        "majority_count": majority_count,
        "carrying_majority_vote": carrying_vote,
        "carrying_majority_count": carrying_count,
    }


def derive_eval_labels(ctx, activity_imu, pos, base_class, n_windows):
    """Per-window ground truth, with stationary overriding the carrying label.

    Classifiers trained before the activity class have no stationary output, so
    for those the base class simply repeats.
    """
    if "stationary" not in ctx.class_names:
        return [base_class] * n_windows, []
    return derive_window_labels(
        activity_imu, pos, base_class, n_windows, ctx.window_samples, ctx.window_samples,
        sample_rate=ctx.args.sample_rate, activity_threshold=ctx.args.activity_threshold,
    )


def write_window_plots(window_imus, outdir, labels, n_plot, gt_labels=None,
                        window_sec=WINDOW_SEC, sample_rate=SAMPLE_RATE):
    """Per-window IMU plots annotated with their predicted label."""
    if n_plot <= 0:
        return
    plot_dir = osp.join(outdir, "window_plots")
    os.makedirs(plot_dir, exist_ok=True)
    for i in range(n_plot):
        gt = "N/A" if gt_labels is None or gt_labels[i] is None else gt_labels[i]
        plots.plot_imu_window(window_imus[i], plot_dir, i, gt_label=gt,
                              pred_label=labels[i], time_offset=i * window_sec, sample_rate=sample_rate)
    print(f"Saved {n_plot} window plots to {plot_dir}")


def write_confusion(labels, predictions, class_names, outdir, filename, title):
    """Confusion matrix plot plus its JSON-ready summary, or None if unlabeled."""
    confusion = metrics.confusion(labels, predictions, class_names)
    if confusion is None:
        return None
    plots.plot_confusion_matrix(confusion["matrix"], class_names, outdir,
                                filename=filename, title=title)
    return {
        "matrix": confusion["matrix"].tolist(),
        "class_names": list(class_names),
        "n_windows": confusion["n_windows"],
        "accuracy": confusion["accuracy"],
        "plot": filename,
    }


def resolve_train_results(train_results, clf_ckpt):
    """Explicit training ``results.json``, else the one beside the checkpoint."""
    if train_results is not None:
        if not osp.isfile(train_results):
            raise FileNotFoundError(f"--train_results not found: {train_results}")
        return train_results
    guess = osp.join(osp.dirname(osp.abspath(clf_ckpt)), "results.json")
    return guess if osp.isfile(guess) else None


class EvalContext:
    """Models, statistics and options shared by every evaluation mode."""

    def __init__(self, args, clf, class_names, imu_mean, imu_std, device, ldm=None):
        self.args = args
        self.clf = clf
        self.class_names = class_names
        self.imu_mean = imu_mean
        self.imu_std = imu_std
        self.device = device
        self.ldm = ldm
        self._warned_placements = set()
        self.window_samples = int(args.window_sec * args.sample_rate)

    def check_recording_windowing(self):
        stride = self.args.stride_sec
        if stride is not None and stride != self.args.window_sec:
            raise ValueError("Per-recording classification uses non-overlapping windows; "
                             "--stride_sec is supported for labeled splits only")

    def n_window_plots(self, n_windows):
        return n_windows if self.args.n_plot < 0 else min(self.args.n_plot, n_windows)

    def gt_label(self, placement):
        """A placement this classifier can be scored against, else None.

        Datasets carry placements the model was never trained on (head against
        a four-class checkpoint). Those recordings are still classified and
        plotted; they are only left out of accuracy and confusion matrices.
        """
        label = paths.scoreable_label(placement, self.class_names)
        if placement and label is None and placement not in self._warned_placements:
            self._warned_placements.add(placement)
            print(f"  [note] placement '{placement}' is not one of this "
                  f"classifier's classes {self.class_names}; these recordings are "
                  f"still classified, just left out of accuracy")
        return label

    def load_imu(self, path):
        """Classifier-frame IMU plus the device-frame IMU used for activity labels."""
        args = self.args
        imu, pos, time = sequences.load_trajectory_imu(
            path, imu_frame=args.imu_frame,
            already_world=args.already_world,
            parquet_from_world=args.parquet_from_world,
            world_heading=args.world_heading,
        )
        activity_imu = sequences.load_activity_imu(
            path, imu, imu_frame=args.imu_frame,
            already_world=args.already_world,
            parquet_from_world=args.parquet_from_world,
            world_heading=args.world_heading,
        )
        return imu, activity_imu, pos, time

    def generate(self, imu, pos, time, sim_imu, seed):
        """Generate world-frame classifier-order IMU from one noise seed."""
        generate.seed_everything(seed)
        if pos is None:
            raise ValueError(
                "LDM generation needs a trajectory (tango_pos / agent_pos)"
            )
        args = self.args
        n = imu.shape[0]
        pos = np.asarray(pos)[:n]
        ts = np.arange(n, dtype=np.float64) / args.sample_rate if time is None \
            else np.asarray(time, dtype=np.float64)[:n]

        if (args.mode == "sim_cond" or args.strength < 1) and sim_imu is None:
            raise ValueError(
                "--mode sim_cond needs synthetic IMU (--sim_input, or a "
                "synthetic.parquet beside the recording)"
            )
        generated = generate.ldm_generate(
            windows.swap_imu_channels(np.asarray(imu, dtype=np.float32)),
            ts, pos, self.ldm["model"], self.ldm["sampler"], self.ldm["stats"],
            self.device,
            sim_features=None if args.mode != "sim_cond"
            else np.asarray(sim_imu, dtype=np.float32)[:n],
            ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
            use_ema=not args.no_ema, strength=args.strength,
            initial_features=sim_imu if args.mode == "traj" and args.strength < 1 else None,
            window=self.window_samples, latent_length=args.latent_length,
        )
        return windows.swap_imu_channels(generated)


# ---------------------------------------------------------------------------
# Main flow, in call order
# ---------------------------------------------------------------------------

def load_models(parser, args, device):
    """Load the VAE, the classifier head and, when requested, the LDM."""
    print("Loading VAE ...")
    vae, _config = models.load_vae(args.vae_config, args.vae_ckpt, device)

    print("Loading classifier ...")
    clf, class_names, clf_ckpt = models.load_classifier(args.clf_ckpt, vae, device)
    class_names = list(class_names)
    args.imu_frame = eval_args.resolve_imu_frame(
        parser, args, clf_ckpt, "the classifier checkpoint"
    )
    if vae.imu_frame is not None and vae.imu_frame != args.imu_frame:
        parser.error("The VAE checkpoint frame does not match the classifier frame")
    if args.imu_frame == "local" and (args.already_world or args.ldm_ckpt):
        parser.error("A local classifier cannot consume generated world-frame IMU")
    if args.activity_threshold is None:
        args.activity_threshold = clf_ckpt.get("activity_threshold", 0.5)
    print(f"  classes: {class_names}")
    print(f"  activity_threshold: {args.activity_threshold}")

    # The classifier's own statistics win: they are what its head was trained on.
    if "imu_mean" in clf_ckpt and "imu_std" in clf_ckpt:
        imu_mean = clf_ckpt["imu_mean"].view(-1, 1)
        imu_std = clf_ckpt["imu_std"].view(-1, 1)
        print("  stats: from the classifier checkpoint")
    else:
        vae_stats = stats.resolve_stats(
            args.vae_stats, None, model=vae, outdir=args.outdir, what="vae_stats",
        )
        imu_mean, imu_std = vae_stats["imu_mean"], vae_stats["imu_std"]
        print(f"  stats: {vae_stats['path']}")

    ldm = None
    if args.ldm_ckpt:
        config_path = eval_args.resolve_ldm_config(args)
        print(f"Loading LDM ({args.mode}) from {args.ldm_ckpt}\n  config: {config_path}")
        model, sampler, _ = models.load_ldm(
            config_path, args.ldm_ckpt, device,
            vae_ckpt=args.vae_ckpt, scale_factor=args.scale_factor,
        )
        ldm = {
            "model": model,
            "sampler": sampler,
            "stats": stats.require_velocity(
                stats.resolve_stats(args.ldm_stats, None, what="ldm_stats")
            ),
        }
        print(f"  scale_factor={float(model.scale_factor)}")
    return EvalContext(args, clf, class_names, imu_mean, imu_std, device, ldm)


def write_recording_result(ctx, recording, outdir, predictions, logits, window_imus,
                           pos, time, plot=True, extra=None, window_gt=None,
                           label_details=None):
    """Timeline, window plots and ``results.json`` for one classified IMU."""
    class_names = ctx.class_names
    n_windows = len(predictions)
    os.makedirs(outdir, exist_ok=True)
    plots.plot_timeline(predictions, class_names, outdir, logits=logits, pos=pos,
                        time=time, window_sec=ctx.args.window_sec,
                        sample_rate=ctx.args.sample_rate)

    result = window_result(predictions, logits, class_names)
    limit = n_windows if ctx.args.n_plot < 0 else max(8, ctx.args.n_plot)
    report.save_plot_data(outdir, windows=window_imus[:limit], pos=pos, time=time)
    if plot:
        write_window_plots(
            window_imus, outdir,
            [f"{result['predictions'][i]} ({result['confidence'][i]:.2f})"
             for i in range(n_windows)],
            ctx.n_window_plots(n_windows), gt_labels=window_gt,
            window_sec=ctx.args.window_sec, sample_rate=ctx.args.sample_rate,
        )

    gt_class = ctx.gt_label(recording.label)
    print(f"Majority vote: {result['majority_vote']} "
          f"({result['majority_count']}/{n_windows} windows)"
          + (f"  |  GT(from name): {gt_class}" if gt_class else ""))
    if result["carrying_majority_vote"] is not None:
        print(f"Carrying-only vote: {result['carrying_majority_vote']} "
              f"({result['carrying_majority_count']} carrying windows)")

    result = {
        "input": osp.abspath(recording.path),
        **result,
        "placement": recording.label,
        "gt_from_name": gt_class,
        "class_names": list(class_names),
        "imu_frame": ctx.args.imu_frame,
        "window_sec": ctx.args.window_sec,
        "sample_rate": ctx.args.sample_rate,
    }
    if window_gt is not None:
        result["window_gt"] = list(window_gt[:n_windows])
        scored = [(g, p) for g, p in zip(window_gt, result["predictions"]) if g is not None]
        if scored:
            result["window_accuracy"] = float(np.mean([g == p for g, p in scored]))
            print(f"Six-class window accuracy: {result['window_accuracy']:.3f} "
                  f"({len(scored)} labeled windows)")
    if label_details:
        result["window_label_details"] = label_details[:n_windows]
    if extra:
        result.update(extra)
    report.write_json(result, outdir, "results.json")
    return result


def classify_recording(ctx, recording, outdir, plot=True):
    """Classify one recording, generating the IMU first when an LDM is loaded.

    Returns a result dict, or None when the recording is shorter than a window.
    """
    args = ctx.args
    ctx.check_recording_windowing()
    print(f"  {recording.path}")
    imu, activity_imu, pos, time = ctx.load_imu(recording.path)
    n_samples = imu.shape[0]

    sim_imu = None
    if ctx.ldm is not None and (args.mode == "sim_cond" or args.strength < 1):
        sim_path = recording.sim_path or (recording.path if recording.is_parquet else None)
        if sim_path is None:
            raise FileNotFoundError(
                f"--mode sim_cond: no synthetic parquet for {recording.path}; "
                f"pass --sim_input"
            )
        print(f"  sim IMU: {sim_path}")
        if time is None:
            raise ValueError("LDM generation needs timestamps")
        sim_imu = sequences.load_sim_on_timeline(sim_path, time)
        # The sim recording conditions every window, so it bounds the usable span
        if sim_imu.shape[0] < n_samples:
            print(f"  [trim] sim IMU is shorter ({sim_imu.shape[0]} samples); "
                  f"truncating real IMU {n_samples} -> {sim_imu.shape[0]}")
            n_samples = sim_imu.shape[0]
            imu, activity_imu = imu[:n_samples], activity_imu[:n_samples]
            pos = None if pos is None else np.asarray(pos)[:n_samples]
            time = None if time is None else np.asarray(time)[:n_samples]

    n_windows = n_samples // ctx.window_samples
    if n_windows == 0:
        print(f"  Recording too short for a full 10s window ({n_samples} samples)")
        return None
    print(f"  {n_samples} samples ({n_samples / args.sample_rate:.1f}s), "
          f"{n_windows} consecutive windows")

    window_gt, label_details = derive_eval_labels(
        ctx, activity_imu, pos, ctx.gt_label(recording.label), n_windows,
    )

    def classify_and_write(source_imu, dest, extra):
        predictions, logits, window_imus = classify_windows(source_imu, ctx)
        return write_recording_result(
            ctx, recording, dest, predictions, logits, window_imus, pos, time,
            plot=plot, extra=extra, window_gt=window_gt[:len(predictions)],
            label_details=label_details[:len(predictions)],
        )

    if ctx.ldm is None:
        return classify_and_write(imu, outdir, extra=None)

    if args.n_noise == 1:
        print(f"  LDM generate (seed={args.seed}) ...")
        generated = ctx.generate(imu, pos, time, sim_imu, args.seed)
        return classify_and_write(
            generated, outdir, {"ldm_seed": args.seed, "n_noise": 1, "mode": args.mode},
        )

    runs = []
    for index in range(args.n_noise):
        seed = args.seed + index
        print(f"\n  LDM noise {index + 1}/{args.n_noise} (seed={seed}) ...")
        generated = ctx.generate(imu, pos, time, sim_imu, seed)
        runs.append(classify_and_write(
            generated, osp.join(outdir, f"noise_{index:03d}_seed{seed}"),
            {"ldm_seed": seed, "noise_index": index, "mode": args.mode},
        ))
    return write_noise_sweep(ctx, recording, outdir, runs, pos, time)


def write_noise_sweep(ctx, recording, outdir, runs, pos, time):
    """Aggregate one trajectory's per-noise-init runs into a single summary."""
    class_names = ctx.class_names
    n_noise = len(runs)
    report.save_plot_data(outdir, pos=pos, time=time)
    plots.plot_noise_comparison(runs, class_names, outdir, pos=pos, time=time,
                                window_sec=ctx.args.window_sec, sample_rate=ctx.args.sample_rate)

    pred_matrix = np.array(
        [metrics.prediction_indices(r["predictions"], class_names) for r in runs]
    )
    per_window = [Counter(pred_matrix[:, w]).most_common(1)[0]
                  for w in range(pred_matrix.shape[1])]
    # One verdict per physical window, so an aggregate confusion matrix counts
    # each window once rather than once per noise init.
    aggregate_predictions = [class_names[int(cls)] for cls, _ in per_window]
    carrying = [p for p in aggregate_predictions if p in CARRYING_CLASS_NAMES]
    if carrying:
        carrying_vote, carrying_count = Counter(carrying).most_common(1)[0]
    else:
        carrying_vote, carrying_count = None, 0
    majority_vote, majority_count = Counter(
        r["majority_vote"] for r in runs
    ).most_common(1)[0]

    summary = {
        "input": osp.abspath(recording.path),
        "mode": ctx.args.mode,
        "n_noise": n_noise,
        "seeds": [r["ldm_seed"] for r in runs],
        "n_windows": runs[0]["n_windows"],
        "mean_window_agreement": float(np.mean([c / n_noise for _, c in per_window])),
        "per_window_agreement": [c / n_noise for _, c in per_window],
        "predictions": aggregate_predictions,
        "window_gt": runs[0].get("window_gt"),
        "window_label_details": runs[0].get("window_label_details"),
        "majority_votes": [r["majority_vote"] for r in runs],
        "majority_vote": majority_vote,
        "majority_count": majority_count,
        "carrying_majority_vote": carrying_vote,
        "carrying_majority_count": carrying_count,
        "gt_from_name": runs[0].get("gt_from_name"),
        "runs": [
            {"seed": r["ldm_seed"], "predictions": r["predictions"],
             "majority_vote": r["majority_vote"], "confidence": r["confidence"]}
            for r in runs
        ],
        "class_names": list(class_names),
        "imu_frame": ctx.args.imu_frame,
        "window_sec": ctx.args.window_sec,
        "sample_rate": ctx.args.sample_rate,
    }
    for name in ("noise_summary.json", "results.json"):
        report.write_json(summary, outdir, name)
    print(f"\n  Mean per-window agreement across {n_noise} noise inits: "
          f"{summary['mean_window_agreement']:.3f}")
    print(f"  Majority-of-runs: {majority_vote} ({majority_count}/{n_noise})")
    return summary


def compare_sources(ctx, recording, outdir, plot=True):
    """Classify real, synthetic and generated IMU for one trajectory."""
    args = ctx.args
    ctx.check_recording_windowing()
    class_names = ctx.class_names
    if recording.sim_path is None:
        raise FileNotFoundError(
            f"no synthetic parquet beside {recording.path}; pass --sim_input"
        )
    print(f"  {recording.path}\n  sim IMU: {recording.sim_path}")

    real_imu, activity_imu, pos, time = ctx.load_imu(recording.path)
    sim_imu = sequences.load_sim_on_timeline(recording.sim_path, time)

    n_samples = min(len(real_imu), len(sim_imu))
    if len(real_imu) != len(sim_imu):
        print(f"  [trim] real {len(real_imu)} / sim {len(sim_imu)} samples "
              f"-> {n_samples}")
    real_imu, sim_imu = real_imu[:n_samples], sim_imu[:n_samples]
    activity_imu = activity_imu[:n_samples]
    pos = None if pos is None else np.asarray(pos)[:n_samples]
    time = None if time is None else np.asarray(time)[:n_samples]

    n_windows = n_samples // ctx.window_samples
    if n_windows == 0:
        print(f"  Recording too short for a full 10s window ({n_samples} samples)")
        return None
    print(f"  {n_samples} samples ({n_samples / args.sample_rate:.1f}s), "
          f"{n_windows} consecutive windows")

    print(f"  LDM generate ({args.mode}, seed={args.seed}) ...")
    gen_imu = ctx.generate(real_imu, pos, time, sim_imu, args.seed)[:n_samples]

    # sim and gen render the placement the pair folder names, but the real IMU
    # was captured however the phone happened to be carried that session.
    gt_class = ctx.gt_label(recording.label)
    native_gt_class = ctx.gt_label(paths.infer_native_gt_class(recording.path))
    if native_gt_class != gt_class:
        print(f"  real recording's actual carrying type: {native_gt_class or 'unknown'}"
              f" (folder names {gt_class} for sim/gen)")
    source_gt = {"real": native_gt_class, "sim": gt_class, "gen": gt_class}

    source_window_gt, label_details = {}, None
    for name, base_class in source_gt.items():
        labels, details = derive_eval_labels(
            ctx, activity_imu, pos, base_class, n_windows,
        )
        source_window_gt[name] = labels
        if label_details is None:
            label_details = details

    sources = {"real": real_imu, "sim": sim_imu, "gen": gen_imu}
    per_source, window_imus = {}, {}
    for name, imu in sources.items():
        predictions, logits, wins = classify_windows(imu, ctx)
        per_source[name] = window_result(predictions, logits, class_names)
        window_imus[name] = wins
        print(f"    {name:4s}: majority {per_source[name]['majority_vote']} "
              f"({per_source[name]['majority_count']}/{len(predictions)})")

    limit = n_windows if args.n_plot < 0 else max(8, args.n_plot)
    report.save_plot_data(outdir, pos=pos, time=time,
                         **{name: wins[:limit] for name, wins in window_imus.items()})
    plots.plot_source_comparison(
        per_source, class_names, outdir, pos=pos, time=time, gt_class=gt_class,
        title=recording.name, window_sec=args.window_sec, sample_rate=args.sample_rate,
    )
    n_plot = ctx.n_window_plots(n_windows) if plot else 0
    if n_plot > 0:
        plot_dir = osp.join(outdir, "window_plots")
        os.makedirs(plot_dir, exist_ok=True)
        for i in range(n_plot):
            labels = {
                name: f"{per_source[name]['predictions'][i]} "
                      f"({per_source[name]['confidence'][i]:.2f})"
                for name in sources
            }
            plots.plot_source_overlay({n: window_imus[n][i] for n in sources},
                                      plot_dir, i, labels, time_offset=i * args.window_sec,
                                      sample_rate=args.sample_rate)
        print(f"Saved {n_plot} overlay window plots to {plot_dir}")

    def agreement(a, b):
        return float(np.mean([x == y for x, y in zip(
            per_source[a]["predictions"], per_source[b]["predictions"])]))

    summary = {
        "input": osp.abspath(recording.path),
        "sim_input": osp.abspath(recording.sim_path),
        "n_windows": n_windows,
        "gt_from_name": gt_class,
        "gt_native_real": native_gt_class,
        "gt_per_source": source_gt,
        "window_gt_per_source": source_window_gt,
        "window_label_details": label_details,
        "mode": args.mode,
        "ldm_seed": args.seed,
        "sources": per_source,
        "window_agreement": {
            "gen_vs_real": agreement("gen", "real"),
            "sim_vs_real": agreement("sim", "real"),
            "gen_vs_sim": agreement("gen", "sim"),
        },
        "majority_votes": {n: per_source[n]["majority_vote"] for n in sources},
        "carrying_majority_votes": {
            n: per_source[n]["carrying_majority_vote"] for n in sources
        },
        # Each source is scored against the placement it actually represents.
        "window_accuracy_vs_gt": {
            n: float(np.mean([
                p == g for p, g in zip(per_source[n]["predictions"], source_window_gt[n])
                if g is not None
            ]))
            for n in SOURCE_NAMES if any(g is not None for g in source_window_gt[n])
        },
        # The real IMU's verdict doubles as the trajectory-level vote in summaries
        "majority_vote": per_source["real"]["majority_vote"],
        "majority_count": per_source["real"]["majority_count"],
        "carrying_majority_vote": per_source["real"]["carrying_majority_vote"],
        "carrying_majority_count": per_source["real"]["carrying_majority_count"],
        "class_names": list(class_names),
        "imu_frame": ctx.args.imu_frame,
        "window_sec": ctx.args.window_sec,
        "sample_rate": ctx.args.sample_rate,
    }
    for name in ("source_comparison.json", "results.json"):
        report.write_json(summary, outdir, name)

    ag = summary["window_agreement"]
    print(f"  Per-window agreement: gen~real {ag['gen_vs_real']:.3f}, "
          f"sim~real {ag['sim_vs_real']:.3f}, gen~sim {ag['gen_vs_sim']:.3f}")
    return summary


def evaluate_labeled_split(ctx):
    """Evaluate the stratified split of a class-labeled directory.

    This mirrors training, so it reads only the folders named for the classes
    this classifier predicts. Placement folders it was not trained on (head
    against a four-class model) are reported and skipped; use ``--per_file`` to
    classify every recording in the directory instead.
    """
    args = ctx.args
    file_list = build_file_list(args.input_dir)
    skipped = [
        osp.basename(d) for d in paths.list_label_dirs(args.input_dir)
        if paths.normalize_placement(osp.basename(d)) not in ctx.class_names
    ]
    if skipped:
        print(f"  skipping {', '.join(skipped)}: not among this classifier's "
              f"classes {ctx.class_names} (use --per_file to classify them anyway)")

    train_files, val_files = split_files_stratified(
        file_list, val_fraction=args.val_fraction, seed=args.seed,
    )
    files = train_files if args.split == "train" else val_files
    print(f"  {len(files)} files in the {args.split} split")
    if not files:
        raise ValueError(
            f"the {args.split} split of {args.input_dir} is empty "
            f"({len(file_list)} scorable file(s) found"
            + (f", {len(skipped)} placement folder(s) skipped" if skipped else "")
            + "). Use --per_file to classify every recording without splitting."
        )

    dataset = CarryingTypeDataset(
        files, ctx.imu_mean, ctx.imu_std,
        window_sec=args.window_sec, sample_rate=args.sample_rate,
        latent_length=args.latent_length,
        stride_sec=args.stride_sec if args.stride_sec is not None else 2.0,
        imu_frame=args.imu_frame, activity_threshold=args.activity_threshold,
        class_names=ctx.class_names,
    )
    print(f"  {len(dataset)} windows (imu_frame={args.imu_frame})")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    # Windows are only kept for the plots, so stop hoarding IMU once the
    # --n_plot budget is covered; a full split does not fit comfortably in RAM.
    plot_budget = None if args.n_plot < 0 else max(args.n_plot, 0)
    all_preds, all_labels, plot_imu, n_kept = [], [], [], 0
    with torch.no_grad():
        for item in loader:
            logits = ctx.clf(item["imu"].float().to(ctx.device))
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(item["label"])
            if plot_budget is None or n_kept < plot_budget:
                take = item["imu"] if plot_budget is None \
                    else item["imu"][:plot_budget - n_kept]
                plot_imu.append(take)
                n_kept += len(take)

    predictions = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    plot_imu = torch.cat(plot_imu) if plot_imu else torch.empty(0)

    from sklearn.metrics import classification_report, confusion_matrix

    accuracy = float((predictions == labels).mean())
    class_ids = list(range(len(ctx.class_names)))
    print(f"\n=== {args.split} results ({len(predictions)} windows) ===")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(classification_report(labels, predictions, labels=class_ids,
                                target_names=ctx.class_names, digits=3, zero_division=0))
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    plots.plot_confusion_matrix(matrix, ctx.class_names, args.outdir)

    if len(plot_imu):
        plot_dir = osp.join(args.outdir, "window_plots")
        os.makedirs(plot_dir, exist_ok=True)
        for i in range(len(plot_imu)):
            plots.plot_imu_window(plot_imu[i].numpy(), plot_dir, i,
                                  gt_label=ctx.class_names[labels[i]],
                                  pred_label=ctx.class_names[predictions[i]])
        print(f"Saved {len(plot_imu)} window plots to {plot_dir}")

    result = {
        "input_dir": osp.abspath(args.input_dir),
        "split": args.split,
        "n_windows": len(predictions),
        "accuracy": accuracy,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(ctx.class_names),
    }
    report.save_plot_data(args.outdir, windows=plot_imu.numpy(),
                         labels=labels[:len(plot_imu)], predictions=predictions[:len(plot_imu)])
    result["sample_rate"] = args.sample_rate
    result["window_sec"] = args.window_sec
    report.write_json(result, args.outdir, "results.json", what="Results")
    return result


def evaluate_directory(ctx, recordings, layout):
    """Classify every recording under ``--input_dir`` and summarize the batch."""
    args = ctx.args
    plot = args.n_plot != 0
    run_one = compare_sources if args.compare_sources else classify_recording
    results, failures = batch.run_batch(
        recordings, args.outdir,
        lambda item, item_outdir: run_one(ctx, item, item_outdir, plot=plot),
        kind="pair" if args.compare_sources else "recording",
        force=args.force, dry_run=args.dry_run, plots_only=args.plots_only,
        strict=args.strict,
    )
    if args.dry_run:
        return None
    extra = {
        "input_dir": osp.abspath(args.input_dir),
        "layout": layout,
        "imu_frame": args.imu_frame,
        "already_world": args.already_world,
        "mode": args.mode if ctx.ldm else None,
        "n_noise": args.n_noise if ctx.ldm else 1,
    }
    if args.compare_sources:
        return summarize_comparison(ctx, results, failures, extra)
    return summarize_classification(ctx, results, failures, extra)


def summarize_classification(ctx, results, failures, extra):
    """Per-file votes, accuracy against the filename class and a confusion matrix."""
    args = ctx.args
    per_file, cm_labels, cm_preds = [], [], []
    n_correct = n_labeled = 0
    for name, result in results:
        gt = result.get("gt_from_name")
        carrying_vote = result.get("carrying_majority_vote")
        correct = gt is not None and carrying_vote == gt
        per_file.append({
            "input": result["input"],
            "stem": name,
            "majority_vote": result["majority_vote"],
            "majority_count": result["majority_count"],
            "carrying_majority_vote": carrying_vote,
            "carrying_majority_count": result.get("carrying_majority_count", 0),
            "n_windows": result["n_windows"],
            "gt_from_name": gt,
            "n_noise": result.get("n_noise"),
            "mean_window_agreement": result.get("mean_window_agreement"),
            "correct": correct,
        })
        if gt is not None:
            n_labeled += 1
            n_correct += int(correct)
        for label, pred in zip(result.get("window_gt", []), result["predictions"]):
            if label is not None:
                cm_labels.append(label)
                cm_preds.append(pred)

    summary = {**extra, "n_evaluated": len(per_file), "failures": failures,
               "files": per_file}
    if n_labeled:
        summary["majority_accuracy_vs_name"] = n_correct / n_labeled
        summary["n_labeled_from_name"] = n_labeled
        print(f"\nCarrying-only majority accuracy vs filename class: "
              f"{n_correct}/{n_labeled} = {n_correct / n_labeled:.3f}")

    confusion = write_confusion(
        cm_labels, cm_preds, ctx.class_names, args.outdir, "confusion_matrix.png",
        "Six-class per-window confusion",
    )
    if confusion is not None:
        summary["window_confusion_matrix"] = confusion
        print(f"Per-window accuracy over {confusion['n_windows']} labeled windows: "
              f"{confusion['accuracy']:.3f}")
    report.write_json(summary, args.outdir, "summary.json", what="Directory summary")
    return summary


def summarize_comparison(ctx, results, failures, extra):
    """Agreement and per-source accuracy across every compared pair."""
    args = ctx.args
    per_pair = []
    cm_labels = {n: [] for n in SOURCE_NAMES}
    cm_preds = {n: [] for n in SOURCE_NAMES}
    n_unlabeled = Counter()
    for name, result in results:
        per_pair.append({
            "pair": name,
            "gt_from_name": result["gt_from_name"],
            "gt_native_real": result["gt_native_real"],
            "gt_per_source": result["gt_per_source"],
            "n_windows": result["n_windows"],
            "majority_votes": result["majority_votes"],
            "carrying_majority_votes": result["carrying_majority_votes"],
            "window_agreement": result["window_agreement"],
            "window_accuracy_vs_gt": result["window_accuracy_vs_gt"],
        })
        # sim and gen are scored against the placement the simulator rendered;
        # real against what the phone was actually doing, which is unknown for
        # some sequences, and those are left out of the real matrix.
        for source in SOURCE_NAMES:
            labels = result["window_gt_per_source"][source]
            if not any(label is not None for label in labels):
                n_unlabeled[source] += 1
                continue
            for label, pred in zip(labels, result["sources"][source]["predictions"]):
                if label is not None:
                    cm_labels[source].append(label)
                    cm_preds[source].append(pred)

    summary = {**extra, "n_evaluated": len(per_pair), "ldm_seed": args.seed,
               "failures": failures, "pairs": per_pair}
    if per_pair:
        summary["mean_window_agreement"] = {
            key: float(np.mean([p["window_agreement"][key] for p in per_pair]))
            for key in ("gen_vs_real", "sim_vs_real", "gen_vs_sim")
        }
        labeled = {n: [p for p in per_pair if p["gt_per_source"][n] is not None]
                   for n in SOURCE_NAMES}
        summary["mean_window_accuracy_vs_gt"] = {
            n: float(np.mean([p["window_accuracy_vs_gt"][n] for p in pairs]))
            for n, pairs in labeled.items() if pairs
        }
        summary["majority_accuracy_vs_gt"] = {
            n: float(np.mean([p["carrying_majority_votes"][n] == p["gt_per_source"][n]
                              for p in pairs]))
            for n, pairs in labeled.items() if pairs
        }
        summary["n_labeled"] = {n: len(pairs) for n, pairs in labeled.items()}

    matrices = {}
    for name in SOURCE_NAMES:
        against = "actual carrying type" if name == "real" else "rendered carrying type"
        confusion = write_confusion(
            cm_labels[name], cm_preds[name], ctx.class_names, args.outdir,
            f"confusion_matrix_{name}.png",
            f"Per-window confusion — {name} vs {against}",
        )
        if confusion is not None:
            confusion["n_pairs_unlabeled"] = n_unlabeled[name]
            matrices[name] = confusion
    if matrices:
        summary["window_confusion_matrix"] = matrices
        print("\nSix-class per-window accuracy:")
        for name, confusion in matrices.items():
            skipped = (f", {confusion['n_pairs_unlabeled']} pairs skipped for unknown "
                       f"carrying type" if confusion["n_pairs_unlabeled"] else "")
            print(f"  {name:4s} {confusion['accuracy']:.3f} over "
                  f"{confusion['n_windows']} windows{skipped}")

    if per_pair:
        ag = summary["mean_window_agreement"]
        print(f"\nMean per-window agreement over {len(per_pair)} pairs: "
              f"gen~real {ag['gen_vs_real']:.3f}, sim~real {ag['sim_vs_real']:.3f}, "
              f"gen~sim {ag['gen_vs_sim']:.3f}")
        accuracy = summary["majority_accuracy_vs_gt"]
        if accuracy:
            print("Majority accuracy vs GT: " + ", ".join(
                f"{n} {accuracy[n]:.3f}" for n in SOURCE_NAMES if n in accuracy))
    report.write_json(summary, args.outdir, "summary.json", what="Comparison summary")
    return summary


def replot_recording(_item, outdir, result, args):
    """Rebuild classifier plots solely from saved numeric inputs/results."""
    data = report.load_plot_data(outdir)
    classes = result["class_names"]
    kwargs = dict(window_sec=result.get("window_sec", 10.0),
                  sample_rate=result.get("sample_rate", 200))
    trajectory = dict(pos=data.get("pos"), time=data.get("time"), **kwargs)
    if "confusion_matrix" in result:
        plots.plot_confusion_matrix(np.asarray(result["confusion_matrix"]), classes, outdir)
        labels = [classes[int(i)] for i in data.get("labels", [])]
        predictions = [classes[int(i)] for i in data.get("predictions", [])]
    elif "sources" in result:
        plots.plot_source_comparison(result["sources"], classes, outdir,
                                     gt_class=result.get("gt_from_name"), **trajectory)
        count = min(len(data[name]) for name in SOURCE_NAMES)
        count = count if args.n_plot < 0 else min(args.n_plot, count)
        dest = osp.join(outdir, "window_plots")
        os.makedirs(dest, exist_ok=True)
        for i in range(count):
            plots.plot_source_overlay(
                {name: data[name][i] for name in SOURCE_NAMES}, dest, i,
                {name: result["sources"][name]["predictions"][i] for name in SOURCE_NAMES},
                sample_rate=kwargs["sample_rate"], time_offset=i * kwargs["window_sec"],
            )
        return
    elif "runs" in result:
        plots.plot_noise_comparison(result["runs"], classes, outdir, **trajectory)
        for i, run in enumerate(result["runs"]):
            dest = osp.join(outdir, f"noise_{i:03d}_seed{run['seed']}")
            replot_recording(None, dest, report.read_json(osp.join(dest, "results.json")), args)
        return
    else:
        predictions = result["predictions"]
        labels = result.get("window_gt")
        plots.plot_timeline(predictions, classes, outdir,
                            logits=np.asarray(result["logits"]), **trajectory)
    count = len(data["windows"])
    count = count if args.n_plot < 0 else min(args.n_plot, count)
    write_window_plots(data["windows"], outdir, predictions, count,
                        gt_labels=labels, **kwargs)


def replot_summary(args, results, failures):
    from types import SimpleNamespace

    if not results:
        return
    ctx = SimpleNamespace(args=args, class_names=results[0][1]["class_names"])
    extra = {"input_dir": osp.abspath(args.input_dir)}
    if args.compare_sources:
        summarize_comparison(ctx, results, failures, extra)
    else:
        summarize_classification(ctx, results, failures, extra)


def main():
    parser = build_parser()
    args = parser.parse_args()
    source = validate(parser, args)

    if args.dry_run or args.plots_only:
        labeled_split = (source == "input_dir" and not args.per_file
                         and not args.compare_sources and not args.ldm_ckpt
                         and paths.classify_dir(args.input_dir) == "labeled")
        batch.handle_offline(
            args, lambda: paths.resolve_input_dir(args.input_dir, pairs_only=args.compare_sources),
            lambda item, dest, result: replot_recording(item, dest, result, args),
            single_output=labeled_split,
            summarize=lambda results, failures: replot_summary(args, results, failures),
        )
        return
    generate.seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = eval_args.resolve_device(args)
    print(f"Device: {device}")

    ctx = load_models(parser, args, device)

    train_results = resolve_train_results(args.train_results, args.clf_ckpt)
    if train_results is not None:
        print(f"Plotting training curves from {train_results}")
        plots.plot_classifier_training_curves(train_results, args.outdir)

    if source == "input":
        recording = paths.resolve_input(args.input, sim_path=args.sim_input)
        if args.compare_sources:
            compare_sources(ctx, recording, args.outdir)
        else:
            classify_recording(ctx, recording, args.outdir)
    else:
        layout = paths.classify_dir(args.input_dir)
        if layout == "labeled" and not args.per_file \
                and not args.compare_sources and ctx.ldm is None:
            print(f"\nEvaluating the labeled {args.split} split of {args.input_dir}")
            evaluate_labeled_split(ctx)
        else:
            layout, recordings = paths.resolve_input_dir(
                args.input_dir, pairs_only=args.compare_sources,
            )
            print(f"\nEvaluating {len(recordings)} recordings under "
                  f"{args.input_dir} ({layout} layout)")
            evaluate_directory(ctx, recordings, layout)

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
