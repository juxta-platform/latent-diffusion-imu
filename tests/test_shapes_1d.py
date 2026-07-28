"""Shape tests for 1D VAE and U-Net models."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


def test_vae_encoder_shape():
    from ldm.modules.diffusionmodules.model_1d import Encoder1D

    encoder = Encoder1D(
        ch=64, in_channels=6, z_channels=8,
        ch_mult=(1, 2, 4), num_res_blocks=2,
        downsample_factors=(2, 2, 5), double_z=True,
    )
    x = torch.randn(2, 6, 2000)
    out = encoder(x)
    assert out.shape == (2, 16, 100), f"Expected (2, 16, 100), got {out.shape}"


def test_vae_decoder_shape():
    from ldm.modules.diffusionmodules.model_1d import Decoder1D

    decoder = Decoder1D(
        ch=64, out_channels=6, z_channels=8,
        ch_mult=(1, 2, 4), num_res_blocks=2,
        upsample_factors=(5, 2, 2),
    )
    z = torch.randn(2, 8, 100)
    out = decoder(z)
    assert out.shape == (2, 6, 2000), f"Expected (2, 6, 2000), got {out.shape}"


def test_vae_full_roundtrip():
    from ldm.models.autoencoder_1d import AutoencoderKL1D

    ddconfig = dict(
        ch=64, in_channels=6, out_channels=6, z_channels=8,
        ch_mult=(1, 2, 4), num_res_blocks=2,
        downsample_factors=(2, 2, 5), upsample_factors=(5, 2, 2),
        double_z=True,
    )
    vae = AutoencoderKL1D(ddconfig=ddconfig, embed_dim=8)

    x = torch.randn(2, 6, 2000)

    # Encode
    posterior = vae.encode(x)
    assert posterior.mean.shape == (2, 8, 100), f"Posterior mean: {posterior.mean.shape}"
    assert posterior.logvar.shape == (2, 8, 100)

    # Sample
    z = posterior.sample()
    assert z.shape == (2, 8, 100)

    # Decode
    recon = vae.decode(z)
    assert recon.shape == (2, 6, 2000), f"Reconstruction: {recon.shape}"


def test_unet_shape():
    from ldm.modules.diffusionmodules.unet_1d import UNetModel1D

    unet = UNetModel1D(
        in_channels=11, out_channels=8,
        model_channels=128, channel_mult=(1, 2, 4),
        num_res_blocks=2, attention_levels=(1, 2),
        dropout=0.0, num_heads=8,
    )

    x = torch.randn(2, 11, 100)
    t = torch.randint(0, 1000, (2,))
    out = unet(x, t)
    assert out.shape == (2, 8, 100), f"Expected (2, 8, 100), got {out.shape}"


def test_unet_input_channels():
    """Verify the U-Net correctly handles concatenated condition channels."""
    from ldm.modules.diffusionmodules.unet_1d import UNetModel1D

    unet = UNetModel1D(in_channels=11, out_channels=8)

    z_t = torch.randn(2, 8, 100)
    velocity = torch.randn(2, 2, 100)
    physical_time = torch.randn(2, 1, 100)
    model_input = torch.cat([z_t, velocity, physical_time], dim=1)

    assert model_input.shape == (2, 11, 100)
    t = torch.randint(0, 1000, (2,))
    out = unet(model_input, t)
    assert out.shape == (2, 8, 100)


def test_conditions_not_in_output():
    """The U-Net only predicts noise for latent channels, not conditions."""
    from ldm.modules.diffusionmodules.unet_1d import UNetModel1D

    unet = UNetModel1D(in_channels=11, out_channels=8)
    x = torch.randn(2, 11, 100)
    t = torch.randint(0, 1000, (2,))
    out = unet(x, t)
    # Output is 8 channels (latent noise), not 11
    assert out.shape[1] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
