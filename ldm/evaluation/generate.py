"""VAE reconstruction and LDM generation over full-length recordings.

Both operate on non-overlapping 10s windows of a ``[N, 6]`` feature array in
RoNIN channel order ``[gyro, accel]``, since that is what the RoNIN sequence
loaders produce; the channel swap to the model's ``[accel, gyro]`` order and
back happens here. Samples in a trailing partial window are left untouched.
"""

from contextlib import nullcontext

import numpy as np
import torch

from ldm.evaluation.constants import LATENT_LENGTH, WINDOW_SAMPLES
from ldm.evaluation.windows import build_conditioning, latent_shape, swap_imu_channels


@torch.no_grad()
def vae_reconstruct(features, vae, stats, device, window=WINDOW_SAMPLES):
    """Reconstruct [N, 6] RoNIN-order IMU window by window through the VAE."""
    n_windows = features.shape[0] // window
    recon = features.copy()
    if n_windows == 0:
        print(f"  [warn] Sequence too short for a full window ({features.shape[0]} samples)")
        return recon

    mean = stats["imu_mean"].to(device)
    std = stats["imu_std"].to(device)
    for i in range(n_windows):
        s, e = i * window, (i + 1) * window
        x = torch.tensor(features[s:e], dtype=torch.float32)
        x = swap_imu_channels(x).T.unsqueeze(0).to(device)  # [1, 6, W] model order
        x_recon, _ = vae((x - mean) / std, sample_posterior=False)
        x_recon = x_recon * std + mean
        recon[s:e] = swap_imu_channels(x_recon.squeeze(0).T.cpu()).numpy()

    _report_tail(features.shape[0], n_windows, window)
    return recon


@torch.no_grad()
def ldm_generate(features, ts, gt_pos, model, sampler, stats, device,
                 sim_features=None, ddim_steps=50, ddim_eta=0.0, use_ema=True,
                 strength=1.0, window=WINDOW_SAMPLES, latent_length=LATENT_LENGTH,
                 initial_features=None):
    """Generate [N, 6] RoNIN-order IMU with a trajectory-conditioned LDM.

    ``sim_features`` (a synthetic recording in model ``[accel, gyro]`` order)
    switches on sim-conditioned generation by adding the VAE-encoded synthetic
    latent to the conditioning.

    Traj-only ``strength < 1`` starts from ``initial_features``: synthetic
    world-frame IMU in model order. Sim conditioning always starts from noise.
    """
    if not 0 <= strength <= 1:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if sim_features is not None and strength != 1:
        raise ValueError("Sim-conditioned generation always starts from pure noise")
    if strength < 1 and initial_features is None:
        raise ValueError("Trajectory img2img requires synthetic world-frame initial_features")

    n_samples = features.shape[0]
    if initial_features is not None and len(initial_features) < n_samples:
        raise ValueError("Synthetic img2img input is shorter than the trajectory")
    n_windows = n_samples // window
    generated = features.copy()
    if n_windows == 0:
        print(f"  [warn] Sequence too short for a full window ({n_samples} samples)")
        return generated

    if sim_features is not None and sim_features.shape[0] < n_samples:
        raise ValueError(
            f"sim IMU length {sim_features.shape[0]} < IMU length {n_samples}"
        )

    mean = stats["imu_mean"].to(device)
    std = stats["imu_std"].to(device)
    vel_mean = stats["vel_mean"].to(device)
    vel_std = stats["vel_std"].to(device)
    shape = latent_shape(model, latent_length)
    mode = "VAE recon" if strength == 0.0 else "full gen" if strength >= 1.0 else "img2img"
    print(f"  LDM strength={strength} ({mode})"
          + ("" if sim_features is None else ", sim-conditioned"))

    context = model.ema_scope("eval") if (use_ema and model.use_ema) else nullcontext()
    with context:
        for i in range(n_windows):
            s, e = i * window, (i + 1) * window
            available = min(e, len(ts), len(gt_pos))
            if available - s < window:
                print(f"  [warn] Window {i} truncated for conditioning; keeping original IMU")
                continue

            cond = build_conditioning(
                ts, gt_pos, [s], window, vel_mean, vel_std, device,
                latent_length=latent_length,
            )
            if sim_features is not None:
                sim_window = torch.tensor(
                    sim_features[s:e].T, dtype=torch.float32
                ).unsqueeze(0).to(device)
                cond["sim_latent"] = model.scale_factor * model.encode_first_stage(
                    (sim_window - mean) / std
                ).mode()

            if strength < 1.0:
                imu_window = torch.tensor(initial_features[s:e].T, dtype=torch.float32)
                imu_window = imu_window.unsqueeze(0).to(device)
                z0 = model.scale_factor * model.encode_first_stage(
                    (imu_window - mean) / std
                ).mode()
                if strength == 0.0:
                    samples = z0
                else:
                    samples, _ = sampler.sample_img2img(
                        S=ddim_steps, x0=z0, conditioning=cond,
                        strength=strength, eta=ddim_eta, verbose=False,
                    )
            else:
                samples, _ = sampler.sample(
                    S=ddim_steps, batch_size=1, shape=shape, conditioning=cond,
                    eta=ddim_eta, verbose=False,
                )

            decoded = model.decode_first_stage(samples / model.scale_factor)
            decoded = decoded * std + mean  # [1, 6, W] model order
            generated[s:e] = swap_imu_channels(decoded.squeeze(0).T.cpu()).numpy()

    _report_tail(n_samples, n_windows, window)
    return generated


@torch.no_grad()
def ldm_generate_dataset(model, sampler, stats, time, position, sim_imu, real_imu,
                         device, window, stride, latent_length, batch_size,
                         ddim_steps, ddim_eta, use_ema, trim_tail):
    """Generate a full world-frame sequence in ``[accel, gyro]`` order.

    Unlike :func:`ldm_generate` this batches windows and supports a stride, so
    it is the path used when writing whole gen_world datasets. Windows are
    generated from noise; a remainder shorter than one window either keeps the
    real IMU (default, when a real recording is available) or is trimmed.
    """
    from ldm.evaluation.windows import window_starts

    n_samples = len(time)
    starts = window_starts(n_samples, window, stride)
    if not starts:
        raise ValueError(f"sequence has {n_samples} samples, fewer than window {window}")

    if real_imu is None:
        output = np.zeros((n_samples, 6), dtype=np.float32)
        retain_real_tail = False
    else:
        if len(real_imu) != n_samples:
            raise ValueError(
                f"real IMU length {len(real_imu)} != trajectory length {n_samples}"
            )
        output = real_imu.copy()
        retain_real_tail = not trim_tail

    mean = stats["imu_mean"].to(device)
    std = stats["imu_std"].to(device)
    vel_mean = stats["vel_mean"].to(device)
    vel_std = stats["vel_std"].to(device)
    shape = latent_shape(model, latent_length)

    context = model.ema_scope("dataset generation") if use_ema and model.use_ema \
        else nullcontext()
    with context:
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            cond = build_conditioning(
                time, position, batch_starts, window, vel_mean, vel_std, device,
                latent_length=latent_length,
            )
            if sim_imu is not None:
                sim_windows = np.stack([
                    sim_imu[start:start + window].T for start in batch_starts
                ])
                sim_windows = torch.from_numpy(sim_windows).float().to(device)
                cond["sim_latent"] = model.scale_factor * model.encode_first_stage(
                    (sim_windows - mean) / std
                ).mode()

            samples, _ = sampler.sample(
                S=ddim_steps, batch_size=len(batch_starts), shape=shape,
                conditioning=cond, eta=ddim_eta, verbose=False,
            )
            generated = model.decode_first_stage(samples / model.scale_factor)
            generated = (generated * std + mean).cpu().numpy()
            if generated.shape[-1] != window:
                raise ValueError(
                    f"model decoded {generated.shape[-1]} samples; expected {window}"
                )
            for item, start in enumerate(batch_starts):
                output[start:start + window] = generated[item].T

            done = min(offset + len(batch_starts), len(starts))
            print(f"    generated {done}/{len(starts)} windows", end="\r", flush=True)
    print()

    tail = n_samples - (starts[-1] + window)
    if tail > 0:
        if retain_real_tail:
            print(f"    retained {tail} trailing real IMU samples")
        else:
            output = output[:-tail]
            print(f"    trimmed {tail} trailing samples")
    return output.astype(np.float32)


def seed_everything(seed):
    """Reseed torch and numpy so one noise init is reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _report_tail(n_samples, n_windows, window):
    tail = n_samples - n_windows * window
    if tail > 0:
        print(f"  [info] {tail} trailing samples kept unchanged (< 1 window)")
