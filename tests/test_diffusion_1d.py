"""Smoke tests for 1D diffusion training and inference pipeline."""


import torch
import numpy as np
import pytest


@pytest.fixture
def vae():
    from ldm.models.autoencoder_1d import AutoencoderKL1D
    ddconfig = dict(
        ch=64, in_channels=6, out_channels=6, z_channels=8,
        ch_mult=(1, 2, 4), num_res_blocks=2,
        downsample_factors=(2, 2, 5), upsample_factors=(5, 2, 2),
        double_z=True,
    )
    return AutoencoderKL1D(ddconfig=ddconfig, embed_dim=8)


@pytest.fixture
def unet():
    from ldm.modules.diffusionmodules.unet_1d import UNetModel1D
    return UNetModel1D(
        in_channels=11, out_channels=8,
        model_channels=128, channel_mult=(1, 2, 4),
        num_res_blocks=2, attention_levels=(1, 2),
        num_heads=8,
    )


def test_diffusion_forward_backward(unet):
    """Smoke test: forward pass, loss computation, backward pass."""
    from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor

    betas = make_beta_schedule("linear", 1000, linear_start=1e-4, linear_end=2e-2)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32)
    sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod), dtype=torch.float32)

    B = 2
    z_0 = torch.randn(B, 8, 100)
    velocity = torch.randn(B, 2, 100)
    physical_time = torch.randn(B, 1, 100)
    t = torch.randint(0, 1000, (B,))
    noise = torch.randn_like(z_0)

    # q_sample
    z_t = (extract_into_tensor(sqrt_alphas_cumprod, t, z_0.shape) * z_0 +
           extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, z_0.shape) * noise)

    model_input = torch.cat([z_t, velocity, physical_time], dim=1)
    assert model_input.shape == (B, 11, 100)

    noise_pred = unet(model_input, t)
    assert noise_pred.shape == (B, 8, 100)

    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    loss.backward()
    for name, p in unet.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


def test_ddim_inference_smoke(vae, unet):
    """Smoke test: encode synthetic, add noise, DDIM denoise, decode."""
    from ldm.models.diffusion.ddpm_1d import LatentDiffusion1D
    from ldm.models.diffusion.ddim_1d import DDIMSampler1D

    # We need a minimal LatentDiffusion1D-like object for the sampler
    # Create a lightweight mock that has the required attributes
    class MockLDM:
        def __init__(self, unet, vae):
            self.model = unet
            self.first_stage_model = vae
            self.scale_factor = 1.0
            self.num_timesteps = 1000

            betas = make_beta_schedule("linear", 1000, linear_start=1e-4, linear_end=2e-2)
            alphas = 1. - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

            self.betas = torch.tensor(betas, dtype=torch.float32)
            self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
            self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32)
            self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32)
            self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod), dtype=torch.float32)

        @property
        def device(self):
            return self.betas.device

        def apply_model(self, x_noisy, t, cond):
            model_input = torch.cat([x_noisy, cond['velocity'], cond['physical_time']], dim=1)
            return self.model(model_input, t)

        def q_sample(self, x_start, t, noise=None):
            from ldm.modules.diffusionmodules.util import extract_into_tensor
            if noise is None:
                noise = torch.randn_like(x_start)
            return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                    extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    from ldm.modules.diffusionmodules.util import make_beta_schedule

    mock_ldm = MockLDM(unet, vae)
    sampler = DDIMSampler1D(mock_ldm)

    B = 2
    synthetic_imu = torch.randn(B, 6, 2000)
    velocity = torch.randn(B, 2, 100)
    physical_time = torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(0).expand(B, 1, 100)

    cond = {"velocity": velocity, "physical_time": physical_time}

    with torch.no_grad():
        # Encode
        posterior = vae.encode(synthetic_imu)
        z_sim = posterior.mode()
        assert z_sim.shape == (B, 8, 100)

        # DDIM img2img with small number of steps
        samples, _ = sampler.sample_img2img(
            S=5, x0=z_sim, conditioning=cond, strength=0.5, eta=0.0,
        )
        assert samples.shape == (B, 8, 100)

        # Decode
        output = vae.decode(samples)
        assert output.shape == (B, 6, 2000), f"Expected (2, 6, 2000), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_conditions_unchanged_through_diffusion(unet):
    """Verify trajectory conditions are not modified by the diffusion process."""
    B = 2
    velocity = torch.randn(B, 2, 100)
    physical_time = torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(0).expand(B, 1, 100).clone()

    velocity_orig = velocity.clone()
    physical_time_orig = physical_time.clone()

    # Run U-Net
    z_t = torch.randn(B, 8, 100)
    t = torch.randint(0, 1000, (B,))
    model_input = torch.cat([z_t, velocity, physical_time], dim=1)
    _ = unet(model_input, t)

    # Conditions should be unchanged
    assert torch.equal(velocity, velocity_orig), "Velocity was modified during diffusion"
    assert torch.equal(physical_time, physical_time_orig), "Physical time was modified during diffusion"


def test_pure_generation(unet):
    """Test generation from pure Gaussian noise."""
    from ldm.models.diffusion.ddim_1d import DDIMSampler1D
    from ldm.modules.diffusionmodules.util import make_beta_schedule

    class MockLDM:
        def __init__(self, unet):
            self.model = unet
            self.num_timesteps = 1000
            betas = make_beta_schedule("linear", 1000, linear_start=1e-4, linear_end=2e-2)
            alphas = 1. - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
            self.betas = torch.tensor(betas, dtype=torch.float32)
            self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
            self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32)

        def apply_model(self, x_noisy, t, cond):
            model_input = torch.cat([x_noisy, cond['velocity'], cond['physical_time']], dim=1)
            return self.model(model_input, t)

    mock = MockLDM(unet)
    sampler = DDIMSampler1D(mock)

    B = 2
    velocity = torch.randn(B, 2, 100)
    physical_time = torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(0).expand(B, 1, 100)
    cond = {"velocity": velocity, "physical_time": physical_time}

    with torch.no_grad():
        samples, _ = sampler.sample(
            S=5, batch_size=B, shape=(8, 100),
            conditioning=cond, eta=0.0,
        )
    assert samples.shape == (B, 8, 100)
    assert torch.isfinite(samples).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
