"""Train the sim-conditioned 1D latent diffusion model.

Uses paired real/sim IMU data. The synthetic IMU is encoded by the same frozen
VAE and concatenated as conditioning alongside trajectory (19-channel U-Net input).

Example usage:
python scripts/train_ldm_1d_sim_cond.py --config configs/imu/ldm_1d_sim_cond.yaml --name sim_cond_exp \
    model.params.first_stage_ckpt=logs/vae_1d/world/checkpoints/best-394.ckpt \
    data.params.data_dir=data/real_sim_pairs_processed

Preprocessing:
python scripts/preprocess_real_sim_imu_pairs.py \
    --pairs_root data/real_sim_imu_pairs \
    --output_dir data/real_sim_pairs_processed
"""

import argparse


import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from ldm.util import instantiate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="logs/ldm_1d_sim_cond")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (used as log folder instead of version_N)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args, unknown = parser.parse_known_args()

    pl.seed_everything(args.seed)

    config = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)

    model = instantiate_from_config(config.model)
    data = instantiate_from_config(config.data)

    callbacks = []
    if "callbacks" in config.lightning:
        for name, cb_cfg in config.lightning.callbacks.items():
            if not cb_cfg.get("enabled", True):
                print(f"Skipping disabled callback: {name}")
                continue
            callbacks.append(instantiate_from_config(cb_cfg))
    if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
        callbacks.append(ModelCheckpoint(monitor="val/loss", save_top_k=3, mode="min"))
    if not any(isinstance(cb, LearningRateMonitor) for cb in callbacks):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if args.name:
        logger = TensorBoardLogger(save_dir=args.logdir, name="", version=args.name)
    else:
        logger = True

    trainer_kwargs = OmegaConf.to_container(config.lightning.trainer, resolve=True)
    if args.resume:
        trainer_kwargs["resume_from_checkpoint"] = args.resume
    trainer = pl.Trainer(
        default_root_dir=args.logdir,
        callbacks=callbacks,
        logger=logger,
        **trainer_kwargs,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
