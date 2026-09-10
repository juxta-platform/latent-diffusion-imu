"""Checkpoint loading for the four models the eval scripts touch.

Each loader raises with the flag that fixes an architecture mismatch, because
the most common failure is a checkpoint paired with the wrong config.
"""

import os
import os.path as osp
import sys

import torch
from omegaconf import OmegaConf

from ldm.evaluation.constants import DEFAULT_LDM_CONFIGS, DEFAULT_RONIN_ROOT
from ldm.util import instantiate_from_config


def resolve_ronin_root(ronin_root=None):
    """Locate the juxta-ronin checkout that supplies the RoNIN networks.

    Precedence: explicit --ronin_root, then $JUXTA_RONIN_ROOT, then the
    sibling-directory default. juxta-ronin is only needed for the network
    definitions; the sequence loaders in this package are self-contained.
    """
    root = ronin_root or os.environ.get("JUXTA_RONIN_ROOT") or DEFAULT_RONIN_ROOT
    if not osp.isfile(osp.join(root, "source", "model_resnet1d.py")):
        raise FileNotFoundError(
            f"No juxta-ronin checkout at {root} (looked for "
            f"source/model_resnet1d.py). The default assumes juxta-ronin is a "
            f"sibling of this repo; pass --ronin_root or set $JUXTA_RONIN_ROOT."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _load_state_dict(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    return state["state_dict"] if "state_dict" in state else state


def checkpoint_imu_frame(checkpoint):
    """Frame metadata from our checkpoints or Lightning hyperparameters."""
    frame = checkpoint.get("imu_frame")
    if frame is None:
        frame = checkpoint.get("hyper_parameters", {}).get("imu_frame")
    if frame is not None and frame not in ("local", "world"):
        raise ValueError(f"Invalid checkpoint imu_frame: {frame!r}")
    return frame


def require_world_vae(checkpoint, path):
    if checkpoint_imu_frame(checkpoint) == "local":
        raise ValueError(f"{path} records imu_frame='local'; the LDM requires a world-frame VAE")


def load_vae(config_path, ckpt_path, device):
    """Load a 1D IMU autoencoder in eval mode."""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load VAE weights from {ckpt_path} into the architecture in "
            f"{config_path}. Pass the matching config via --vae_config.\n{exc}"
        ) from exc
    model.imu_frame = checkpoint_imu_frame(checkpoint) or getattr(model, "imu_frame", None)
    return model.to(device).eval(), config


def default_ldm_config(mode):
    """Stock config for an LDM ``--mode`` when ``--ldm_config`` is omitted."""
    return DEFAULT_LDM_CONFIGS[mode]


def load_ldm(config_path, ckpt_path, device, vae_ckpt=None, scale_factor=None):
    """Load a latent diffusion model plus its DDIM sampler.

    ``vae_ckpt`` overrides the first stage recorded in the config, which is how
    the scripts keep ``--vae_ckpt`` authoritative for both models.
    """
    from ldm.models.diffusion.ddim_1d import DDIMSampler1D

    config = OmegaConf.load(config_path)
    if vae_ckpt is not None:
        config.model.params.first_stage_ckpt = vae_ckpt
    vae_path = config.model.params.first_stage_ckpt
    vae_checkpoint = torch.load(vae_path, map_location="cpu", weights_only=False)
    require_world_vae(vae_checkpoint, vae_path)
    configured_frame = config.model.params.first_stage_config.params.get("imu_frame")
    if configured_frame == "local":
        raise ValueError("The LDM first_stage_config specifies a local-frame VAE")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if checkpoint.get("first_stage_imu_frame") == "local":
        raise ValueError(f"{ckpt_path} was trained with a local-frame VAE")
    model = instantiate_from_config(config.model)
    try:
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
        # A Lightning LDM checkpoint includes frozen first-stage weights too.
        # Restore the explicitly selected VAE after loading those weights.
        if vae_ckpt is not None:
            model.first_stage_model.load_state_dict(
                vae_checkpoint.get("state_dict", vae_checkpoint), strict=False,
            )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load LDM weights from {ckpt_path} into the architecture in "
            f"{config_path}. Pass the matching config via --ldm_config.\n{exc}"
        ) from exc
    if scale_factor is not None:
        model.register_buffer("scale_factor", torch.tensor(float(scale_factor)))
    model.first_stage_model.imu_frame = "world"
    model = model.to(device).eval()
    return model, DDIMSampler1D(model), config


def load_classifier(clf_ckpt_path, vae, device):
    """Load the latent MLP head and wrap it with its frozen VAE encoder.

    Returns (classifier, class_names, checkpoint) so callers can read the
    ``imu_frame`` / ``activity_threshold`` / stats the checkpoint was trained
    with rather than re-specifying them on the command line.
    """
    from ldm.data.carrying_labels import CLASS_NAMES
    from ldm.models.carrying_classifier import CarryingTypeClassifier, LatentMLPClassifier

    ckpt = torch.load(clf_ckpt_path, map_location="cpu")
    mlp = LatentMLPClassifier(
        in_dim=ckpt["in_dim"],
        hidden=tuple(ckpt["hidden"]),
        n_classes=ckpt["n_classes"],
        dropout=ckpt.get("dropout", 0.0),
    )
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    classifier = CarryingTypeClassifier(vae, mlp.to(device).eval())
    classifier.eval()
    return classifier, ckpt.get("class_names", CLASS_NAMES), ckpt


def _build_ronin_2d(arch, window_size, ronin_root=None):
    """2D RoNIN ResNet1D; imports from juxta-ronin without its training deps."""
    resolve_ronin_root(ronin_root)
    from source.model_resnet1d import BasicBlock1D, FCOutputModule, ResNet1D

    fc_config = {"fc_dim": 512, "in_dim": window_size // 32 + 1,
                 "dropout": 0.5, "trans_planes": 128}
    if arch == "resnet18":
        groups = [2, 2, 2, 2]
    elif arch == "resnet50":
        groups = [3, 4, 6, 3]
        fc_config["fc_dim"] = 1024
    elif arch == "resnet101":
        groups = [3, 4, 23, 3]
        fc_config["fc_dim"] = 1024
    else:
        raise ValueError(f"Unknown --ronin_arch {arch}")
    return ResNet1D(6, 2, BasicBlock1D, groups, base_plane=64,
                    output_block=FCOutputModule, kernel_size=3, **fc_config)


def _build_ronin_3d(arch, window_size, ronin_root=None):
    resolve_ronin_root(ronin_root)
    from source.model_resnet1d_3d import BasicBlock1D, ResNet1D

    if arch == "resnet18":
        groups = [2, 2, 2, 2]
    elif arch == "resnet50":
        groups = [3, 4, 6, 3]
    elif arch == "resnet101":
        groups = [3, 4, 23, 3]
    else:
        raise ValueError(f"Unknown --ronin_arch {arch}")
    return ResNet1D(BasicBlock1D, 6, 3, groups, window_size // 32 + 1)


def load_ronin(ckpt_path, device, arch="resnet18", window_size=200, use_3d=False,
               ronin_root=None):
    """Load a RoNIN velocity regressor in eval mode."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    network = _build_ronin_3d(arch, window_size, ronin_root) if use_3d \
        else _build_ronin_2d(arch, window_size, ronin_root)
    network.load_state_dict(ckpt["model_state_dict"])
    return network.eval().to(device)
