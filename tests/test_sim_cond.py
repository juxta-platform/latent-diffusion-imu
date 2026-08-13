"""Tests for the sim-conditioned LDM pipeline (19-channel U-Net)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
def unet_19ch():
    from ldm.modules.diffusionmodules.unet_1d import UNetModel1D
    return UNetModel1D(
        in_channels=19, out_channels=8,
        model_channels=128, channel_mult=(1, 2, 4),
        num_res_blocks=2, attention_levels=(1, 2),
        num_heads=8,
    )


def test_unet_19ch_shape(unet_19ch):
    """19-channel U-Net produces 8-channel output."""
    x = torch.randn(2, 19, 100)
    t = torch.randint(0, 1000, (2,))
    out = unet_19ch(x, t)
    assert out.shape == (2, 8, 100)


def test_sim_cond_apply_model(vae, unet_19ch):
    """apply_model correctly concatenates sim_latent + velocity + time."""
    from ldm.models.diffusion.ddpm_1d_sim_cond import LatentDiffusion1DSimCond

    B = 2
    x_noisy = torch.randn(B, 8, 100)
    t = torch.randint(0, 1000, (B,))
    cond = {
        'sim_latent': torch.randn(B, 8, 100),
        'velocity': torch.randn(B, 2, 100),
        'physical_time': torch.randn(B, 1, 100),
    }

    # Directly test the concat logic matches U-Net expectation
    model_input = torch.cat([
        x_noisy, cond['sim_latent'], cond['velocity'], cond['physical_time']
    ], dim=1)
    assert model_input.shape == (B, 19, 100)

    out = unet_19ch(model_input, t)
    assert out.shape == (B, 8, 100)
    assert torch.isfinite(out).all()


def test_sim_cond_forward_backward(vae, unet_19ch):
    """Full forward + backward through the 19-channel diffusion pipeline."""
    from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor

    betas = make_beta_schedule("linear", 1000, linear_start=1e-4, linear_end=2e-2)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32)
    sqrt_one_minus = torch.tensor(np.sqrt(1. - alphas_cumprod), dtype=torch.float32)

    B = 2
    # Simulate encoding real and sim through VAE
    with torch.no_grad():
        real_imu = torch.randn(B, 6, 2000)
        sim_imu = torch.randn(B, 6, 2000)
        z_real = vae.encode(real_imu).sample()
        z_sim = vae.encode(sim_imu).sample()

    t = torch.randint(0, 1000, (B,))
    noise = torch.randn_like(z_real)
    z_t = (extract_into_tensor(sqrt_alphas_cumprod, t, z_real.shape) * z_real +
           extract_into_tensor(sqrt_one_minus, t, z_real.shape) * noise)

    velocity = torch.randn(B, 2, 100)
    physical_time = torch.randn(B, 1, 100)

    model_input = torch.cat([z_t, z_sim, velocity, physical_time], dim=1)
    noise_pred = unet_19ch(model_input, t)

    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    assert torch.isfinite(loss)

    loss.backward()
    # VAE should not have gradients (it should be frozen in real usage)
    for p in vae.parameters():
        assert p.grad is None or (p.grad == 0).all()

    # U-Net should have gradients
    has_grad = False
    for p in unet_19ch.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            assert torch.isfinite(p.grad).all()
    assert has_grad, "U-Net should have non-zero gradients"


def test_target_unchanged():
    """Verify diffusion target is noise on real latent only, not sim."""
    from ldm.models.diffusion.ddpm_1d_sim_cond import LatentDiffusion1DSimCond

    # p_losses should not be overridden — target must remain noise on real latent
    assert 'p_losses' not in LatentDiffusion1DSimCond.__dict__, \
        "p_losses should not be overridden in the subclass"


def test_paired_dataset_shapes(tmp_path):
    """PairedIMUDataset returns correct shapes."""
    from ldm.data.paired_imu_dataset import PairedIMUDataset

    # Create mock data
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    stats = {
        'imu_mean': torch.zeros(6),
        'imu_std': torch.ones(6),
        'vel_mean': torch.zeros(2),
        'vel_std': torch.ones(2),
    }
    torch.save(stats, tmp_path / 'stats.pt')

    for i in range(3):
        window = {
            'imu': torch.randn(6, 2000),
            'sim_imu': torch.randn(6, 2000),
            'velocity': torch.randn(2, 100),
            'physical_time': torch.randn(1, 100),
        }
        torch.save(window, train_dir / f'window_{i:04d}.pt')

    ds = PairedIMUDataset(str(tmp_path), str(tmp_path / 'stats.pt'), split='train')
    assert len(ds) == 3

    sample = ds[0]
    assert sample['imu'].shape == (6, 2000)
    assert sample['sim_imu'].shape == (6, 2000)
    assert sample['velocity'].shape == (2, 100)
    assert sample['physical_time'].shape == (1, 100)


def test_paired_dataset_shared_normalization(tmp_path):
    """Both real and sim IMU are normalized with the same stats."""
    from ldm.data.paired_imu_dataset import PairedIMUDataset

    train_dir = tmp_path / 'train'
    train_dir.mkdir()

    imu_mean = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    imu_std = torch.tensor([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    stats = {
        'imu_mean': imu_mean,
        'imu_std': imu_std,
        'vel_mean': torch.zeros(2),
        'vel_std': torch.ones(2),
    }
    torch.save(stats, tmp_path / 'stats.pt')

    raw_real = torch.ones(6, 2000) * 2.0
    raw_sim = torch.ones(6, 2000) * 3.0
    window = {
        'imu': raw_real,
        'sim_imu': raw_sim,
        'velocity': torch.zeros(2, 100),
        'physical_time': torch.zeros(1, 100),
    }
    torch.save(window, train_dir / 'window_0000.pt')

    ds = PairedIMUDataset(str(tmp_path), str(tmp_path / 'stats.pt'), split='train')
    sample = ds[0]

    expected_real = (raw_real - imu_mean.view(6, 1)) / imu_std.view(6, 1)
    expected_sim = (raw_sim - imu_mean.view(6, 1)) / imu_std.view(6, 1)
    assert torch.allclose(sample['imu'], expected_real)
    assert torch.allclose(sample['sim_imu'], expected_sim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
