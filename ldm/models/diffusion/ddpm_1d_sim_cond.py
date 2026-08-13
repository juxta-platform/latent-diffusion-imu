"""Latent Diffusion with additional synthetic IMU conditioning.

Subclass of LatentDiffusion1D that encodes paired synthetic IMU through the
same frozen VAE and concatenates the resulting latent as extra U-Net conditioning
channels. The diffusion target (noise on real IMU latent) is unchanged.
"""

import torch

from ldm.models.diffusion.ddpm_1d import LatentDiffusion1D


class LatentDiffusion1DSimCond(LatentDiffusion1D):
    """LDM conditioned on trajectory + synthetic IMU latent.

    U-Net input channels: 8 (noisy real latent) + 8 (sim latent) + 2 (vel) + 1 (time) = 19
    U-Net output channels: 8 (noise prediction on real latent)
    """

    def apply_model(self, x_noisy, t, cond):
        model_input = torch.cat([
            x_noisy,
            cond['sim_latent'],
            cond['velocity'],
            cond['physical_time'],
        ], dim=1)
        return self.model(model_input, t)

    def _shared_step(self, batch):
        imu = batch['imu']
        sim_imu = batch['sim_imu']

        # Encode real IMU
        posterior = self.encode_first_stage(imu)
        z = posterior.sample()
        z = self.scale_factor * z

        # Encode synthetic IMU with the same frozen VAE
        sim_posterior = self.encode_first_stage(sim_imu)
        z_sim = sim_posterior.sample()
        z_sim = self.scale_factor * z_sim

        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=z.device).long()
        cond = {
            'sim_latent': z_sim,
            'velocity': batch['velocity'],
            'physical_time': batch['physical_time'],
        }
        loss = self.p_losses(z, t, cond)
        return loss

    def training_step(self, batch, batch_idx):
        if self.scale_by_std and self.global_step == 0 and float(self.scale_factor) == 1.0:
            with torch.no_grad():
                imu = batch['imu']
                posterior = self.encode_first_stage(imu)
                z = posterior.sample()
                self.scale_factor.fill_(float(1.0 / z.flatten().std()))
                print(f"Setting scale_factor to {float(self.scale_factor)}")

        loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
