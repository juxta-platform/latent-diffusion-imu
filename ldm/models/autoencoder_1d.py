import torch
import torch.nn as nn
import pytorch_lightning as pl

from ldm.modules.diffusionmodules.model_1d import Encoder1D, Decoder1D
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class AutoencoderKL1D(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 kl_weight=1e-6,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="imu",
                 monitor=None,
                 learning_rate=1e-4,
                 imu_frame=None,
                 ):
        super().__init__()
        self.input_key = input_key
        self.embed_dim = embed_dim
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        if imu_frame not in (None, "local", "world"):
            raise ValueError(f"Invalid imu_frame: {imu_frame!r}")
        self.imu_frame = imu_frame

        self.encoder = Encoder1D(**ddconfig)
        self.decoder = Decoder1D(**ddconfig)

        self.quant_conv = nn.Conv1d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)

        in_ch = ddconfig.get("in_channels", 6)
        self.register_buffer('imu_mean', torch.zeros(in_ch))
        self.register_buffer('imu_std', torch.ones(in_ch))
        self.register_buffer('stats_set', torch.tensor(False))

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        self.on_load_checkpoint(sd)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        self.load_state_dict(sd, strict=False)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["imu_frame"] = self.imu_frame or "world"

    def on_load_checkpoint(self, checkpoint):
        self.imu_frame = checkpoint.get("imu_frame", self.imu_frame)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch):
        x = batch[self.input_key]
        return x.float()

    def _kl_loss(self, posterior):
        return 0.5 * torch.sum(
            posterior.mean ** 2 + posterior.var - 1.0 - posterior.logvar,
            dim=[1, 2],
        ).mean()

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        rec_loss = torch.nn.functional.mse_loss(reconstructions, inputs)
        kl_loss = self._kl_loss(posterior)
        loss = rec_loss + self.kl_weight * kl_loss

        log_dict = {
            "train/total_loss": loss.detach(),
            "train/rec_loss": rec_loss.detach(),
            "train/kl_loss": kl_loss.detach(),
        }
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        rec_loss = torch.nn.functional.mse_loss(reconstructions, inputs)
        kl_loss = self._kl_loss(posterior)
        loss = rec_loss + self.kl_weight * kl_loss

        log_dict = {
            "val/total_loss": loss.detach(),
            "val/rec_loss": rec_loss.detach(),
            "val/kl_loss": kl_loss.detach(),
        }
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def set_stats(self, stats):
        """Store IMU standardization stats so they survive in the checkpoint."""
        self.imu_mean.copy_(stats['imu_mean'].view(-1))
        self.imu_std.copy_(stats['imu_std'].view(-1))
        self.stats_set.fill_(True)

    def standardize(self, x):
        """Standardize IMU input x ([B, C, T]) using embedded stats."""
        return (x - self.imu_mean[None, :, None]) / self.imu_std[None, :, None]

    def inverse_standardize(self, x):
        """Undo standardization on x ([B, C, T])."""
        return x * self.imu_std[None, :, None] + self.imu_mean[None, :, None]

    def get_last_layer(self):
        return self.decoder.conv_out.weight
