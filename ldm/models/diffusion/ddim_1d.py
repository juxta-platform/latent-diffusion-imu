"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler1D(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.betas.device:
                attr = attr.to(self.model.betas.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, \
            'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.betas.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(torch.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(torch.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(torch.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(torch.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(torch.sqrt(1. / alphas_cumprod - 1)))

        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) *
            (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning, eta=0.,
               x_T=None, verbose=False, temperature=1., noise_dropout=0.,
               log_every_t=100, **kwargs):
        """
        shape: (C, T) e.g. (8, 100)
        conditioning: dict with 'velocity' and 'physical_time'
        """
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        C, T = shape
        size = (batch_size, C, T)
        if verbose:
            print(f'Data shape for DDIM 1D sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(
            conditioning, size, x_T=x_T,
            temperature=temperature, noise_dropout=noise_dropout,
            log_every_t=log_every_t,
        )
        return samples, intermediates

    @torch.no_grad()
    def sample_img2img(self, S, x0, conditioning, strength=0.5, eta=0.,
                       verbose=False, temperature=1., noise_dropout=0., **kwargs):
        """
        Img2img-style partial denoising: encode input, add noise at strength, denoise.

        x0: [B, C, T] clean latent (already scaled)
        strength: float 0-1 (0=no change, 1=full generation)
        """
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        t_start = int(strength * len(self.ddim_timesteps))
        if t_start == 0:
            return x0, {}

        timesteps = self.ddim_timesteps[:t_start]

        noise = torch.randn_like(x0)
        t_enc = torch.full((x0.shape[0],), self.ddim_timesteps[t_start - 1],
                           device=x0.device, dtype=torch.long)
        x_noisy = self.model.q_sample(x0, t_enc, noise=noise)

        samples, intermediates = self.ddim_sampling(
            conditioning, x_noisy.shape, x_T=x_noisy,
            timesteps=timesteps,
            temperature=temperature, noise_dropout=noise_dropout,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, x_T=None, timesteps=None,
                      temperature=1., noise_dropout=0., log_every_t=100):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddim_timesteps

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        iterator = tqdm(time_range, desc='DDIM Sampler 1D', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            img, pred_x0 = self.p_sample_ddim(
                img, cond, ts, index=index,
                temperature=temperature, noise_dropout=noise_dropout,
            )
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False,
                      temperature=1., noise_dropout=0.):
        b, *_, device = *x.shape, x.device
        e_t = self.model.apply_model(x, t, c)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        # 1D data is (B, C, T) so broadcast shape is (B, 1, 1)
        a_t = torch.full((b, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1), sqrt_one_minus_alphas[index], device=device)

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
