"""Every figure produced by the evaluation scripts.

Grouped by what they show: IMU signals, RoNIN trajectories, classifier output,
method comparisons and training curves.
"""

import os
import os.path as osp

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from ldm.evaluation.constants import (  # noqa: E402
    CHANNEL_NAMES,
    RONIN_CHANNEL_NAMES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SEC,
)
from ldm.evaluation.metrics import (  # noqa: E402
    prediction_indices,
    rte_title_suffix,
    softmax,
    window_rmse,
)

SOURCE_NAMES = ("real", "sim", "gen")
SOURCE_STYLES = {
    "real": {"color": "k", "linewidth": 0.9, "alpha": 0.85},
    "sim": {"color": "tab:orange", "linewidth": 0.9, "alpha": 0.8},
    "gen": {"color": "tab:blue", "linewidth": 0.9, "alpha": 0.8},
}

METHOD_LABELS = [
    "real_imu",
    "noise_generated",
    "synthetic_imu",
    "strength_generated",
    "sim_cond_generated",
]
METHOD_COLORS = {
    "real_imu": "blue",
    "noise_generated": "red",
    "synthetic_imu": "green",
    "strength_generated": "orange",
    "sim_cond_generated": "purple",
}


def save(fig, outdir, filename, what, dpi=150):
    """Write a figure under ``outdir`` and report where it went."""
    os.makedirs(outdir, exist_ok=True)
    path = osp.join(outdir, filename)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {what} to {path}")
    return path


# ---------------------------------------------------------------------------
# IMU signals
# ---------------------------------------------------------------------------

def plot_signal_overlays(inputs, outputs, outdir, n_plot, sample_rate=SAMPLE_RATE,
                         title_prefix="reconstruction", input_label="input",
                         output_label="output", channel_names=CHANNEL_NAMES):
    """Per-window channel overlays of [B, 6, T] arrays plus a per-channel RMSE bar.

    ``n_plot`` of -1 plots every window; 0 plots none but still writes the bar.
    """
    overlay_dir = osp.join(outdir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    t = np.arange(inputs.shape[-1]) / sample_rate
    n = inputs.shape[0] if n_plot < 0 else min(n_plot, inputs.shape[0])

    for i in range(n):
        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True,
                                 constrained_layout=True)
        for c, ax in enumerate(axes):
            ax.plot(t, inputs[i, c], label=input_label, linewidth=0.8, alpha=0.9)
            ax.plot(t, outputs[i, c], label=output_label, linewidth=0.8, alpha=0.9)
            ax.set_ylabel(channel_names[c], fontsize=9)
            ax.set_title(f"{channel_names[c]}  "
                         f"(RMSE={window_rmse(inputs[i, c], outputs[i, c]):.4e})",
                         fontsize=9, loc="left")
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"{title_prefix} — sample {i}", fontsize=12)
        fig.savefig(osp.join(overlay_dir, f"sample_{i:03d}.png"), dpi=140)
        plt.close(fig)

    n_bar = max(n, 1)
    rmse = np.sqrt(((inputs[:n_bar] - outputs[:n_bar]) ** 2).mean(axis=(0, 2)))
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.bar(channel_names, rmse)
    ax.set_ylabel("RMSE")
    ax.set_title(f"Per-channel RMSE over {n_bar} samples")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(osp.join(outdir, "per_channel_rmse.png"), dpi=140)
    plt.close(fig)
    print(f"Saved {n} overlay plots under {overlay_dir}")


def plot_full_sequence_windows(features_a, features_b, outdir, n_plot=4,
                               label_a="original", label_b="generated",
                               window=WINDOW_SAMPLES, sample_rate=SAMPLE_RATE,
                               channel_names=RONIN_CHANNEL_NAMES,
                               subdir="imu_overlays"):
    """Channel overlays of two full-length [N, 6] sequences, window by window."""
    n_windows = features_a.shape[0] // window
    n_plot = n_windows if n_plot < 0 else min(n_plot, n_windows)
    if n_plot <= 0:
        return

    overlay_dir = osp.join(outdir, subdir)
    os.makedirs(overlay_dir, exist_ok=True)
    t = np.arange(window) / sample_rate

    for wi in range(n_plot):
        s, e = wi * window, (wi + 1) * window
        a, b = features_a[s:e], features_b[s:e]
        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True,
                                 constrained_layout=True)
        for c, ax in enumerate(axes):
            ax.plot(t, a[:, c], lw=0.7, alpha=0.9, label=label_a)
            ax.plot(t, b[:, c], lw=0.7, alpha=0.9, label=label_b)
            ax.set_ylabel(channel_names[c], fontsize=9)
            ax.set_title(f"{channel_names[c]}  (RMSE={window_rmse(a[:, c], b[:, c]):.4e})",
                         fontsize=9, loc="left")
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"IMU window {wi} — {label_a} vs {label_b}", fontsize=12)
        fig.savefig(osp.join(overlay_dir, f"window_{wi:03d}.png"), dpi=140)
        plt.close(fig)

    print(f"Saved {n_plot} IMU overlay plots to {overlay_dir}")


def plot_imu_window(imu, outdir, idx, gt_label=None, pred_label=None,
                    sample_rate=SAMPLE_RATE, time_offset=0.0):
    """Six-channel plot of one [6, T] window, annotated with GT and prediction."""
    t = np.arange(imu.shape[-1]) / sample_rate + time_offset
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True,
                             constrained_layout=True)
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
    fig.savefig(osp.join(outdir, f"window_{idx:03d}.png"), dpi=140)
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
    fig.savefig(osp.join(outdir, f"window_{idx:03d}.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RoNIN trajectories
# ---------------------------------------------------------------------------

def plot_trajectories(res_orig, res_other, outdir, other_label="LDM gen",
                      rte_delta_sec=10.0, title=None,
                      filename="trajectory_comparison.png"):
    """Ground truth vs original vs modified-IMU trajectories, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    ax = axes[0]
    ax.plot(res_orig["pos_gt"][:, 0], res_orig["pos_gt"][:, 1], "k-", lw=1.2, label="GT")
    ax.plot(res_orig["pos_pred"][:, 0], res_orig["pos_pred"][:, 1], "b-", lw=1.0,
            alpha=0.85, label="RoNIN")
    ax.set_title(f"Original IMU\n{rte_title_suffix(res_orig, rte_delta_sec)}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(res_other["pos_gt"][:, 0], res_other["pos_gt"][:, 1], "k-", lw=1.2, label="GT")
    ax.plot(res_other["pos_pred"][:, 0], res_other["pos_pred"][:, 1], "r-", lw=1.0,
            alpha=0.85, label=f"RoNIN ({other_label})")
    ax.set_title(f"{other_label} IMU\n{rte_title_suffix(res_other, rte_delta_sec)}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(res_orig["pos_gt"][:, 0], res_orig["pos_gt"][:, 1], "k-", lw=1.2, label="GT")
    ax.plot(res_orig["pos_pred"][:, 0], res_orig["pos_pred"][:, 1], "b-", lw=0.9,
            alpha=0.7, label="Original")
    ax.plot(res_other["pos_pred"][:, 0], res_other["pos_pred"][:, 1], "r-", lw=0.9,
            alpha=0.7, label=other_label)
    ax.set_title("Overlay")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    fig.suptitle(title or f"RoNIN trajectory: Original vs {other_label} IMU", fontsize=13)
    fig.tight_layout()
    return save(fig, outdir, filename, "trajectory plot")


def plot_position_error(res_orig, res_other, outdir, other_label="LDM gen",
                        sample_rate=SAMPLE_RATE, filename="position_error.png"):
    """Cumulative position error against ground truth over time."""
    err_orig = np.linalg.norm(res_orig["pos_pred"] - res_orig["pos_gt"], axis=1)
    err_other = np.linalg.norm(res_other["pos_pred"] - res_other["pos_gt"], axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(err_orig)) / sample_rate, err_orig, "b-", lw=0.8, alpha=0.85,
            label=f"Original (ATE={res_orig['ate']:.3f})")
    ax.plot(np.arange(len(err_other)) / sample_rate, err_other, "r-", lw=0.8, alpha=0.85,
            label=f"{other_label} (ATE={res_other['ate']:.3f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Cumulative position error vs ground truth")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save(fig, outdir, filename, "position error plot")


# ---------------------------------------------------------------------------
# Method comparison (compare_ldm_modes.py)
# ---------------------------------------------------------------------------

def plot_method_overlay(methods, outdir, pair_id, rte_delta_sec=10.0,
                        filename="five_method_trajectory_overlay.png"):
    """All available methods' trajectories for one pair, over the same GT."""
    from ldm.evaluation.metrics import format_rte_sec

    short = format_rte_sec(rte_delta_sec)
    fig, ax = plt.subplots(figsize=(10, 10))
    gt = next(iter(methods.values()))["pos_gt"]
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2.0, label="GT", zorder=10)
    for label in METHOD_LABELS:
        if label not in methods:
            continue
        m = methods[label]
        ax.plot(m["pos_pred"][:, 0], m["pos_pred"][:, 1],
                color=METHOD_COLORS[label], lw=1.0, alpha=0.8,
                label=(f"{label} (ATE={m['ate']:.3f}, RTE_60s={m['rte']:.3f}, "
                       f"RTE_{short}={m['rte_short']:.3f})"))
    ax.set_title(f"Trajectory Comparison — {pair_id}", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save(fig, outdir, filename, "method overlay")


def plot_method_bars(methods, outdir, pair_id, rte_delta_sec=10.0,
                     filename="five_method_ate_rte.png"):
    """Grouped ATE / RTE bars per method for one pair."""
    from ldm.evaluation.metrics import format_rte_sec

    present = [label for label in METHOD_LABELS if label in methods]
    short = format_rte_sec(rte_delta_sec)
    x = np.arange(len(present))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    series = [
        (x - width, [methods[l]["ate"] for l in present], "ATE", "steelblue"),
        (x, [methods[l]["rte"] for l in present], "RTE 60s", "salmon"),
        (x + width, [methods[l]["rte_short"] for l in present], f"RTE {short}", "seagreen"),
    ]
    for positions, values, label, color in series:
        bars = ax.bar(positions, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=6, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Error (m)")
    ax.set_title(f"ATE / RTE Comparison — {pair_id}", fontsize=11)
    ax.legend()
    fig.tight_layout()
    return save(fig, outdir, filename, "method bar chart")


def plot_aggregate_bars(per_method, n_trajectories, outdir,
                        filename="aggregate_ate_rte.png"):
    """Mean +/- std ATE and RTE per method across a whole dataset."""
    present = [label for label in METHOD_LABELS if per_method[label]["ate"]]
    if not present:
        return None

    x = np.arange(len(present))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    series = [
        (x - width, "ate", "ATE", "steelblue"),
        (x, "rte", "RTE 60s", "salmon"),
        (x + width, "rte_short", "RTE short", "seagreen"),
    ]
    for positions, key, label, color in series:
        means = [np.mean(per_method[l][key]) for l in present]
        stds = [np.std(per_method[l][key]) for l in present]
        bars = ax.bar(positions, means, width, yerr=stds, label=label,
                      color=color, capsize=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=6, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Error (m)")
    ax.set_title(f"Mean ATE / RTE Across {n_trajectories} Trajectories", fontsize=11)
    ax.legend()
    fig.tight_layout()
    return save(fig, outdir, filename, "aggregate bar chart")


# ---------------------------------------------------------------------------
# Classifier output
# ---------------------------------------------------------------------------

def _class_colors(class_names):
    return plt.cm.Set2(np.linspace(0, 1, len(class_names)))


def _traj_velocity(pos, time, sample_rate=SAMPLE_RATE):
    """Horizontal velocity from differentiating trajectory position.

    Uses the first two position columns (HDF5 tango x/y, parquet agent x/z).
    Returns (t_sec, vel_xy [N,2], speed [N]) aligned with the position samples.
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


def has_velocity(pos):
    """Whether a velocity panel can be drawn (matches :func:`_traj_velocity`)."""
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


def plot_timeline(predictions, class_names, outdir, window_sec=WINDOW_SEC,
                  logits=None, pos=None, time=None, sample_rate=SAMPLE_RATE):
    """Predicted class vs time, with velocity, logits and confidence panels."""
    n = len(predictions)
    t_start = np.arange(n) * window_sec
    t_mid = t_start + window_sec / 2
    colors = _class_colors(class_names)

    show_logits = logits is not None and len(logits) == n
    show_vel = has_velocity(pos)
    probs = softmax(logits) if show_logits else None

    n_rows = 1 + int(show_vel) + (2 if show_logits else 0)
    height = 4 + (3 if show_vel else 0) + (5 if show_logits else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(8, n * 0.6), height),
                             sharex=True, constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]
    for i, pred in enumerate(predictions):
        ax.barh(0, window_sec, left=t_start[i], height=0.6,
                color=colors[pred], edgecolor="white", linewidth=0.5)
        label = class_names[pred]
        if show_logits:
            label = f"{label}\n{probs[i, pred]:.2f}"
        ax.text(t_mid[i], 0, label, ha="center", va="center", fontsize=7)
    ax.set_yticks([])
    ax.set_title("Predicted class" + (" (label = class, conf)" if show_logits else ""))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(class_names))]
    ax.legend(handles, class_names, loc="upper right", fontsize=8)

    row = 1
    if show_vel:
        _plot_velocity_panel(axes[row], pos, time, n, window_sec, sample_rate,
                             shade_by=predictions, colors=colors, window_mean=True)
        axes[row].set_title("Trajectory velocity (d/dt position, window-aligned)")
        row += 1

    if show_logits:
        logits = np.asarray(logits, dtype=np.float64)
        for values, ylabel, title, ylim in (
            (logits, "logit", "Class logits per window", None),
            (probs, "softmax prob", "Class confidence (softmax)", (-0.05, 1.05)),
        ):
            ax = axes[row]
            for c, name in enumerate(class_names):
                ax.plot(t_mid, values[:, c], marker="o", markersize=4,
                        linewidth=1.2, color=colors[c], label=name)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            row += 1

    axes[-1].set_xlabel("time (s)")
    return save(fig, outdir, "timeline.png", "timeline")


def plot_noise_comparison(runs, class_names, outdir, window_sec=WINDOW_SEC,
                          pos=None, time=None, sample_rate=SAMPLE_RATE):
    """Classifier labels across LDM noise inits for one trajectory."""
    n_runs = len(runs)
    pred_mat = np.array(
        [prediction_indices(r["predictions"], class_names) for r in runs], dtype=np.int64
    )
    n_win = pred_mat.shape[1]
    colors = _class_colors(class_names)
    t_start = np.arange(n_win) * window_sec

    show_vel = has_velocity(pos)
    heights = [0.9 + 0.12 * n_runs] + ([2.2] if show_vel else []) + [2.4]
    height = (2.2 + 0.28 * n_runs) + (3.0 if show_vel else 0) + 2.4
    fig, axes = plt.subplots(len(heights), 1, figsize=(max(10, n_win * 0.55), height),
                             sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": heights})

    _plot_pred_heatmap(
        fig, axes[0], pred_mat, [f"seed {r['ldm_seed']}" for r in runs],
        class_names, colors, window_sec, "Predicted class vs LDM noise init",
    )

    row = 1
    if show_vel:
        _plot_velocity_panel(axes[row], pos, time, n_win, window_sec, sample_rate)
        axes[row].set_title("Trajectory velocity")
        row += 1

    ax = axes[row]
    # Bars share the x axis with the panels above, so use seconds (not indices)
    x = t_start + window_sec / 2
    bottom = np.zeros(n_win)
    for c, name in enumerate(class_names):
        frac = (pred_mat == c).mean(axis=0)
        ax.bar(x, frac, bottom=bottom, color=colors[c], width=window_sec * 0.9, label=name)
        bottom += frac
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of noise inits")
    ax.set_title("Per-window class agreement across noise inits")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}" for t in t_start], fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xlabel("window start time (s)")

    return save(fig, outdir, "noise_comparison.png", "noise comparison")


def plot_source_comparison(per_source, class_names, outdir, window_sec=WINDOW_SEC,
                           pos=None, time=None, sample_rate=SAMPLE_RATE,
                           gt_class=None, title=None):
    """Classifier output on real vs sim vs generated IMU for one trajectory."""
    names = [n for n in SOURCE_NAMES if n in per_source]
    pred_mat = np.array(
        [prediction_indices(per_source[n]["predictions"], class_names) for n in names],
        dtype=np.int64,
    )
    n_win = pred_mat.shape[1]
    colors = _class_colors(class_names)
    t_start = np.arange(n_win) * window_sec
    t_mid = t_start + window_sec / 2

    show_vel = has_velocity(pos)
    heights = [0.5 + 0.3 * len(names), 2.4] + ([2.2] if show_vel else []) + [2.4]
    fig, axes = plt.subplots(
        len(heights), 1,
        figsize=(min(max(11, n_win * 0.6), 40), sum(heights) + 1.5),
        sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": heights},
    )

    _plot_pred_heatmap(
        fig, axes[0], pred_mat, names, class_names, colors, window_sec,
        "Predicted class per window" + ("" if gt_class is None else f"   (GT: {gt_class})"),
    )

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
    if show_vel:
        _plot_velocity_panel(axes[row], pos, time, n_win, window_sec, sample_rate)
        axes[row].set_title("Trajectory velocity")
        row += 1

    # Probability each source assigns to the class predicted from real IMU
    ax = axes[row]
    ref = np.array(prediction_indices(per_source["real"]["predictions"], class_names))
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
    return save(fig, outdir, "source_comparison.png", "source comparison")


def plot_confusion_matrix(matrix, class_names, outdir,
                          filename="confusion_matrix.png", title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    return save(fig, outdir, filename, "confusion matrix")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def _event_accumulator(logdir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed; skipping training-curve plots")
        return None, []
    ea = EventAccumulator(logdir)
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    if not scalars:
        print(f"No scalar tags found in {logdir}")
        return None, []
    return ea, scalars


def _series(ea, scalars, tag):
    if tag not in scalars:
        return None
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def plot_vae_training_curves(logdir, outdir):
    """Total / reconstruction / KL loss from a VAE Lightning event file."""
    ea, scalars = _event_accumulator(logdir)
    if ea is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, name in zip(axes, ["total_loss", "rec_loss", "kl_loss"]):
        train = _series(ea, scalars, f"train/{name}_epoch")
        val = _series(ea, scalars, f"val/{name}")
        if train is not None:
            ax.plot(train[0], train[1], label="train (epoch)", alpha=0.85)
        if val is not None:
            ax.plot(val[0], val[1], label="val", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if name != "kl_loss":
            ax.set_yscale("log")
    return save(fig, outdir, "training_curves.png", "training curves")


def plot_ldm_training_curves(logdir, outdir):
    """Epoch and step diffusion loss from an LDM Lightning event file."""
    ea, scalars = _event_accumulator(logdir)
    if ea is None:
        return None

    train_epoch = _series(ea, scalars, "train/loss_epoch")
    train_step = _series(ea, scalars, "train/loss_step")
    val = _series(ea, scalars, "val/loss")
    val_ema = _series(ea, scalars, "val/loss_ema")
    epoch_map = _series(ea, scalars, "epoch")
    if train_epoch is None and train_step is None and val is None:
        print(f"No loss tags found in {logdir}")
        return None

    def to_epochs(steps):
        """Map TensorBoard global steps onto epochs via the logged 'epoch' scalar."""
        if epoch_map is None or len(steps) == 0:
            return steps
        return np.interp(steps.astype(float), epoch_map[0].astype(float),
                         epoch_map[1].astype(float))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    if train_epoch is not None:
        ax.plot(to_epochs(train_epoch[0]), train_epoch[1], label="train",
                color="C0", alpha=0.85, linewidth=1.2)
    if val is not None:
        ax.plot(to_epochs(val[0]), val[1], label="val", color="C1",
                alpha=0.9, linewidth=1.4)
    if val_ema is not None:
        ax.plot(to_epochs(val_ema[0]), val_ema[1], label="val (EMA)", color="C2",
                alpha=0.9, linewidth=1.4)
    ax.set_title("Epoch loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if train_step is not None:
        ax.plot(train_step[0], train_step[1], label="train (step)", color="C0",
                alpha=0.55, linewidth=0.9)
    if val is not None:
        ax.plot(val[0], val[1], label="val", color="C1", alpha=0.9, linewidth=1.4)
    if val_ema is not None:
        ax.plot(val_ema[0], val_ema[1], label="val (EMA)", color="C2",
                alpha=0.9, linewidth=1.4)
    ax.set_title("Step loss")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return save(fig, outdir, "training_curves.png", "training curves")


def plot_classifier_training_curves(results_path, outdir):
    """Train/val loss and accuracy from a classifier training ``results.json``."""
    import json

    with open(results_path) as f:
        results = json.load(f)
    history = results.get("history")
    if not history:
        print(f"No history found in {results_path}; skipping training curves")
        return None

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

    return save(fig, outdir, "training_curves.png", "training curves")
