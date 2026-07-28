import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial

from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config


class LatentDiffusion1D(pl.LightningModule):
    def __init__(self,
                 unet_config,
                 first_stage_config,
                 first_stage_ckpt,
                 timesteps=1000,
                 beta_schedule="linear",
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 loss_type="l2",
                 scale_factor=1.0,
                 scale_by_std=False,
                 learning_rate=1e-4,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor="val/loss",
                 use_ema=True,
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale_by_std = scale_by_std
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.use_ema = use_ema
        self.monitor = monitor

        self.first_stage_model = instantiate_from_config(first_stage_config)
        sd = torch.load(first_stage_ckpt, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = self.first_stage_model.load_state_dict(sd, strict=False)
        if missing:
            print(f"VAE missing keys: {missing}")
        if unexpected:
            print(f"VAE unexpected keys: {unexpected}")
        self.first_stage_model.eval()
        self.first_stage_model.train = lambda self, mode=True: self
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        self.model = instantiate_from_config(unet_config)

        self.register_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=None):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        if ignore_keys:
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print(f"Deleting key {k} from state_dict.")
                        del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def register_schedule(self, beta_schedule, timesteps, linear_start, linear_end, cosine_s):
        betas = make_beta_schedule(beta_schedule, timesteps,
                                   linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)

    def apply_model(self, x_noisy, t, cond):
        """
        x_noisy: [B, latent_ch, T_latent] noisy latent
        t: [B] timestep indices
        cond: dict with 'velocity' [B, 2, T_latent] and 'physical_time' [B, 1, T_latent]

        Concatenates conditions along channel dim and runs U-Net.
        """
        model_input = torch.cat([x_noisy, cond['velocity'], cond['physical_time']], dim=1)
        return self.model(model_input, t)

    def p_losses(self, x_start, t, cond, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.loss_type == "l2":
            loss = torch.nn.functional.mse_loss(model_output, noise)
        elif self.loss_type == "l1":
            loss = torch.nn.functional.l1_loss(model_output, noise)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type}")
        return loss

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _shared_step(self, batch):
        imu = batch['imu']
        posterior = self.encode_first_stage(imu)
        z = posterior.sample()
        z = self.scale_factor * z

        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=z.device).long()
        cond = {
            'velocity': batch['velocity'],
            'physical_time': batch['physical_time'],
        }
        loss = self.p_losses(z, t, cond)
        return loss

    def training_step(self, batch, batch_idx):
        if self.scale_by_std and self.global_step == 0 and self.scale_factor == 1.0:
            with torch.no_grad():
                imu = batch['imu']
                posterior = self.encode_first_stage(imu)
                z = posterior.sample()
                self.scale_factor = 1.0 / z.std()
                print(f"Setting scale_factor to {self.scale_factor}")

        loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_no_ema = self._shared_step(batch)
        self.log("val/loss", loss_no_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        if self.use_ema:
            with self.ema_scope():
                loss_ema = self._shared_step(batch)
            self.log("val/loss_ema", loss_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return loss_no_ema

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        return optimizer
