"""Train the 1D IMU VAE (AutoencoderKL1D).

Example usage:
python scripts/train_vae_1d.py --config configs/imu/vae_1d.yaml --name my_exp data.params.data_dir=data/dataset_processed_overlapped
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ldm.util import instantiate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="logs/vae_1d")
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
        for cb_cfg in config.lightning.callbacks.values():
            callbacks.append(instantiate_from_config(cb_cfg))
    if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
        callbacks.append(ModelCheckpoint(monitor="val/total_loss", save_top_k=3, mode="min"))
    if not any(isinstance(cb, LearningRateMonitor) for cb in callbacks):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if args.name:
        logger = TensorBoardLogger(save_dir=args.logdir, name="", version=args.name)
    else:
        logger = True  # default: lightning_logs/version_N

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
