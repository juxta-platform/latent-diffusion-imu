"""Pure trajectory-conditioned IMU generation from Gaussian noise."""

import argparse
import os


import torch
import numpy as np
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_1d import DDIMSampler1D
from ldm.data.imu_dataset import IMUDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="LDM checkpoint path")
    parser.add_argument("--data_dir", type=str, required=True, help="Preprocessed data directory (for conditioning)")
    parser.add_argument("--outdir", type=str, default="outputs/sample_generate")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=8)
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

    # Load conditioning from validation set
    stats_path = os.path.join(args.data_dir, "stats.pt")
    dataset = IMUDataset(args.data_dir, stats_path, split="val")

    generated = []
    idx = 0
    while len(generated) < args.n_samples:
        bs = min(args.batch_size, args.n_samples - len(generated))
        batch_indices = list(range(idx, min(idx + bs, len(dataset))))
        if len(batch_indices) == 0:
            break
        idx += bs

        batch = [dataset[i] for i in batch_indices]
        velocity = torch.stack([b["velocity"] for b in batch]).to(device)
        physical_time = torch.stack([b["physical_time"] for b in batch]).to(device)

        cond = {"velocity": velocity, "physical_time": physical_time}

        with torch.no_grad():
            samples, _ = sampler.sample(
                S=args.ddim_steps,
                batch_size=len(batch_indices),
                shape=(8, 100),
                conditioning=cond,
                eta=args.ddim_eta,
                verbose=True,
            )

            # Decode
            z = samples / model.scale_factor
            imu_standardized = model.decode_first_stage(z)

        generated.append({
            "imu_generated": imu_standardized.cpu(),
            "velocity": velocity.cpu(),
            "physical_time": physical_time.cpu(),
        })

    save_path = os.path.join(args.outdir, "generated_samples.pt")
    torch.save(generated, save_path)
    print(f"Saved {args.n_samples} generated samples to {save_path}")


if __name__ == "__main__":
    main()
