"""Synthetic-IMU partial-noising (sim-to-real) inference.

This script encodes synthetic IMU through the VAE, adds diffusion noise
at a configurable strength, then reverse-denoises conditioned on trajectory.
This is an out-of-distribution inference experiment - the model has NOT been
trained on synthetic IMU. It relies on the latent space learned from real data.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_1d import DDIMSampler1D
from ldm.data.imu_dataset import SyntheticIMUDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="LDM checkpoint path")
    parser.add_argument("--synthetic_data", type=str, required=True, help="Path to synthetic parquet file")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to stats.pt from preprocessing")
    parser.add_argument("--outdir", type=str, default="outputs/sim2real")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--use_posterior_mean", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)

    sd = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    sampler = DDIMSampler1D(model)

    # Load synthetic data
    dataset = SyntheticIMUDataset(
        args.synthetic_data, args.stats_path,
        window_sec=10, sample_rate=200, latent_length=100,
    )

    n = min(args.n_samples, len(dataset))
    batch = [dataset[i] for i in range(n)]
    imu = torch.stack([b["imu"] for b in batch]).to(device)
    velocity = torch.stack([b["velocity"] for b in batch]).to(device)
    physical_time = torch.stack([b["physical_time"] for b in batch]).to(device)
    imu_raw = torch.stack([b["imu_raw"] for b in batch]).cpu()

    cond = {"velocity": velocity, "physical_time": physical_time}

    results = {"strengths": args.strengths, "samples": [], "synthetic_raw": imu_raw}

    with torch.no_grad():
        # Encode synthetic IMU
        posterior = model.encode_first_stage(imu)
        if args.use_posterior_mean:
            z_sim = posterior.mode()
        else:
            z_sim = posterior.sample()
        z_sim = model.scale_factor * z_sim

        # VAE reconstruction (strength=0 reference)
        z_recon = z_sim / model.scale_factor
        recon = model.decode_first_stage(z_recon)
        results["vae_reconstruction"] = recon.cpu()

        for strength in args.strengths:
            print(f"\n--- Strength: {strength} ---")
            if strength == 0.0:
                generated = recon
            elif strength >= 1.0:
                # Full generation from noise
                samples, _ = sampler.sample(
                    S=args.ddim_steps,
                    batch_size=n,
                    shape=(8, 100),
                    conditioning=cond,
                    eta=args.ddim_eta,
                )
                z_gen = samples / model.scale_factor
                generated = model.decode_first_stage(z_gen)
            else:
                samples, _ = sampler.sample_img2img(
                    S=args.ddim_steps,
                    x0=z_sim,
                    conditioning=cond,
                    strength=strength,
                    eta=args.ddim_eta,
                )
                z_gen = samples / model.scale_factor
                generated = model.decode_first_stage(z_gen)

            results["samples"].append({
                "strength": strength,
                "imu_generated": generated.cpu(),
            })
            print(f"  Generated shape: {generated.shape}")

    save_path = os.path.join(args.outdir, "sim2real_results.pt")
    torch.save(results, save_path)
    print(f"\nSaved results for {len(args.strengths)} strengths to {save_path}")


if __name__ == "__main__":
    main()
