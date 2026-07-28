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
                 ):
        super().__init__()
        self.input_key = input_key
        self.embed_dim = embed_dim
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate

        self.encoder = Encoder1D(**ddconfig)
        self.decoder = Decoder1D(**ddconfig)

        self.quant_conv = nn.Conv1d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        self.load_state_dict(sd, strict=False)

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

    def get_last_layer(self):
        return self.decoder.conv_out.weight
