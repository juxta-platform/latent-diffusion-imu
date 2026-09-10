"""Regression tests at the recording/model/RoNIN frame boundaries."""

from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from ldm.evaluation import args as eval_args
from ldm.evaluation import generate, models, paths, report, ronin, sequences, windows
from ldm.evaluation.constants import (
    ACCEL_LOCAL_COLS, ACCEL_WORLD_COLS, GYRO_LOCAL_COLS, GYRO_WORLD_COLS, PHONE_ROT_COLS,
)
from scripts import eval_classifier, eval_ldm, eval_vae


@pytest.fixture
def recording(tmp_path):
    n = 2400
    time = np.arange(n, dtype=np.float64) / 200
    pos = np.column_stack([time, time * .2, time * 0])
    local = np.column_stack([np.sin(time) + 1, time * 0 + 2, time * 0 + 9,
                             time * 0 + .1, np.cos(time), time * 0 + .3])
    rotation = Rotation.from_euler("xyz", np.column_stack([time * 3, time * 2, time * 5]),
                                   degrees=True)
    quat = rotation.as_quat()
    world = np.column_stack([rotation.apply(local[:, :3]), rotation.apply(local[:, 3:])])
    hdf5 = tmp_path / "real.h5"
    with h5py.File(hdf5, "w") as f:
        for name, values in dict(time=time, tango_pos=pos, game_rv=quat[:, [3, 0, 1, 2]],
                                 acce=local[:, :3], gyro=local[:, 3:]).items():
            f.create_dataset("synced/" + name, data=values)
    heading = .35
    # Independent simulator export: Y-up -> Z-up, then a heading rotation.
    axis_map = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    yaw = Rotation.from_euler("z", heading)
    exported = np.column_stack([yaw.apply(world[:, :3] @ axis_map.T),
                                yaw.apply(world[:, 3:] @ axis_map.T)])
    df = pd.DataFrame(dict(time=time, agent_pos_x=pos[:, 0], agent_pos_z=pos[:, 1]))
    for cols, values in ((ACCEL_LOCAL_COLS + GYRO_LOCAL_COLS, local),
                         (ACCEL_WORLD_COLS + GYRO_WORLD_COLS, exported),
                         (PHONE_ROT_COLS, quat)):
        for i, col in enumerate(cols):
            df[col] = values[:, i]
    parquet = tmp_path / "synthetic.parquet"
    df.to_parquet(parquet)
    return SimpleNamespace(hdf5=str(hdf5), parquet=str(parquet), local=local,
                           world=world, exported=exported, time=time, pos=pos,
                           heading=heading, df=df)


@pytest.mark.parametrize("frame", ["local", "world"])
@pytest.mark.parametrize("kind", ["hdf5", "parquet", "parquet_fallback"])
def test_model_input_and_world_roundtrip(recording, frame, kind):
    r = recording
    path = r.hdf5 if kind == "hdf5" else r.parquet
    if kind == "parquet_fallback":
        r.df.drop(columns=ACCEL_LOCAL_COLS + GYRO_LOCAL_COLS).to_parquet(path)
    imu, _, _ = sequences.load_trajectory_imu(path, frame, world_heading=r.heading)
    expected_world = r.world if kind == "hdf5" else r.exported
    np.testing.assert_allclose(imu, r.local if frame == "local" else expected_world, atol=1e-12)
    np.testing.assert_allclose(sequences.imu_to_world(path, imu, frame, r.heading),
                               expected_world, atol=1e-12)


def test_generated_hdf5_classifier_frames_and_activity(recording):
    r = recording
    with h5py.File(r.hdf5, "a") as f:
        f["synced/acce"][:] = r.world[:, :3]
        f["synced/gyro"][:] = r.world[:, 3:]
        f.attrs["imu_frame"] = "world"
    imu, _, _ = sequences.load_trajectory_imu(r.hdf5, "world")
    np.testing.assert_allclose(imu, r.world)
    np.testing.assert_array_equal(sequences.load_activity_imu(r.hdf5, imu, "world"), imu)
    with pytest.raises(ValueError, match="world-frame"):
        sequences.load_trajectory_imu(r.hdf5, "local")
    with pytest.raises(ValueError, match="world-frame"):
        sequences.load_trajectory_imu(r.parquet, "local", already_world=True)
    # Legacy generated exports are also handled by the explicit override.
    with h5py.File(r.hdf5, "a") as f:
        del f.attrs["imu_frame"]
    ds = ronin.load_strided_dataset(r.hdf5, already_world=True)
    np.testing.assert_allclose(windows.swap_imu_channels(ds.features[0]), r.world)


@pytest.mark.parametrize("already_world", [False, True])
def test_generated_hdf5_export_keeps_tail_in_world(recording, tmp_path, already_world):
    r = recording
    if already_world:
        # A legacy generated file has no frame attribute.
        with h5py.File(r.hdf5, "a") as f:
            f["synced/acce"][:] = r.world[:, :3]
            f["synced/gyro"][:] = r.world[:, 3:]
    generated = r.world[:100] * 2
    dest = str(tmp_path / "export.hdf5")
    report.write_generated_hdf5(r.hdf5, dest, windows.swap_imu_channels(generated),
                               already_world=already_world)
    actual = sequences.load_hdf5(dest, "world").imu
    np.testing.assert_allclose(actual[:100], generated)
    np.testing.assert_allclose(actual[100:], r.world[100:])


class ProbeVAE:
    def __init__(self):
        self.inputs = []
        self.posterior_flags = []

    def __call__(self, x, sample_posterior):
        self.inputs.append(x.clone())
        self.posterior_flags.append(sample_posterior)
        return 2 * x, None

    def _kl_loss(self, posterior):
        return torch.tensor(0.)


@pytest.mark.parametrize("frame", ["local", "world"])
@pytest.mark.parametrize("kind", ["hdf5", "parquet"])
def test_vae_ronin_uses_the_same_reconstruction(recording, frame, kind, tmp_path, monkeypatch):
    r = recording
    path = getattr(r, kind)
    st = {"imu_mean": torch.arange(6).float().view(6, 1), "imu_std": torch.ones(6, 1) * 2}
    args = SimpleNamespace(imu_frame=frame, already_world=False, parquet_from_world=False,
                           world_heading=r.heading, window_sec=5., sample_rate=200,
                           latent_length=50, batch_size=2, num_workers=0, n_plot=0,
                           max_batches=None, sample_posterior=True, ronin_step=10,
                           ronin_window=200, vae_ckpt="unused")
    captured = {}

    def compare(net, before, after, *a, **kw):
        captured.update(before=before, after=after)
        return {}, {}

    monkeypatch.setattr(ronin, "compare_against_baseline", compare)
    monkeypatch.setattr(eval_vae.plots, "plot_signal_overlays", lambda *a, **kw: None)
    monkeypatch.setattr(eval_vae.plots, "plot_full_sequence_windows", lambda *a, **kw: None)
    vae = ProbeVAE()
    result = eval_vae.evaluate_recording(paths.resolve_input(path), str(tmp_path / "out"),
                                         vae, st, object(), "cpu", args, .01)
    assert result["n_windows"] == 2
    assert len(vae.inputs) == 1  # No second VAE pass for trajectory evaluation.
    assert vae.inputs[0].shape == (2, 6, 1000)
    assert vae.posterior_flags == [True]
    model_imu = r.local if frame == "local" else (r.world if kind == "hdf5" else r.exported)
    expected = model_imu.copy()
    expected[:2000] = 2 * expected[:2000] - st["imu_mean"].numpy().T
    expected = sequences.imu_to_world(path, expected, frame, r.heading)
    np.testing.assert_allclose(windows.swap_imu_channels(captured["after"].features[0]),
                               expected, atol=3e-6)
    np.testing.assert_allclose(windows.swap_imu_channels(captured["before"].features[0]),
                               r.world if kind == "hdf5" else r.exported, atol=1e-12)


def test_h5_and_trimmed_scoring_timeline(recording):
    ds = ronin.load_strided_dataset(recording.hdf5)
    ronin.trim_dataset_to(ds, 1200)
    assert len(ds.features[0]) == len(ds.ts[0]) == len(ds.gt_pos[0]) == 1200
    assert len(ds.targets[0]) == 1000
    assert max(frame for _, frame in ds.index_map) < 1000
    predictions = np.ones((len(ds), 2))
    assert len(ronin.recon_traj_2d(ds, predictions)) == 1200


@pytest.mark.parametrize("checkpoint,explicit,expected", [
    ({}, None, "world"), ({"imu_frame": "local"}, None, "local"),
    ({"hyper_parameters": {"imu_frame": "local"}}, None, "local"),
    ({"imu_frame": "local"}, "world", "world"),
])
def test_checkpoint_frame_resolution(checkpoint, explicit, expected):
    assert eval_args.resolve_imu_frame(None, SimpleNamespace(imu_frame=explicit),
                                       checkpoint, "test") == expected


def test_ldm_rejects_local_vae_before_model_creation(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    vae = tmp_path / "vae.pt"
    torch.save({"imu_frame": "local", "state_dict": {}}, vae)
    config = OmegaConf.create({"model": {"params": {"first_stage_ckpt": str(vae)}}})
    monkeypatch.setattr(models.OmegaConf, "load", lambda _: config)
    with pytest.raises(ValueError, match="world-frame VAE"):
        models.load_ldm("unused", "unused", "cpu", vae_ckpt=str(vae))


def test_classifier_rejects_local_generation_before_loading_ldm(monkeypatch):
    parser = eval_classifier.build_parser()
    args = parser.parse_args(["--input", "unused", "--vae_ckpt", "unused", "--clf_ckpt", "unused",
                              "--ldm_ckpt", "unused", "--imu_frame", "local"])
    monkeypatch.setattr(models, "load_vae", lambda *a: (SimpleNamespace(imu_frame="local"), None))
    monkeypatch.setattr(models, "load_classifier", lambda *a: (None, ["chest"], {"imu_frame": "local"}))
    with pytest.raises(SystemExit):
        eval_classifier.load_models(parser, args, "cpu")


def test_traj_img2img_uses_sim_world_and_real_ronin_baseline(recording):
    parser = eval_ldm.build_parser()
    args = parser.parse_args(["--mode", "traj", "--input", recording.hdf5,
                              "--imu_input", recording.parquet, "--strength", "0",
                              "--vae_ckpt", "unused", "--ldm_ckpt", "unused"])
    eval_ldm.validate(parser, args)
    ds, original, _, _, _, cond_sim, initial, _ = eval_ldm.load_sequences(
        paths.resolve_input(args.input), args,
    )
    assert cond_sim is None
    np.testing.assert_allclose(windows.swap_imu_channels(original), recording.world, atol=1e-12)
    np.testing.assert_allclose(initial, recording.exported, atol=1e-12)


@pytest.mark.parametrize("module", [eval_ldm, eval_classifier])
def test_sim_cond_rejects_strength(module):
    parser = module.build_parser()
    flags = ["--input", "unused", "--vae_ckpt", "unused", "--ldm_ckpt", "unused",
             "--mode", "sim_cond", "--strength", "1"]
    if module is eval_classifier:
        flags += ["--clf_ckpt", "unused"]
    with pytest.raises(SystemExit):
        module.validate(parser, parser.parse_args(flags))


def test_conditioning_is_invariant_to_coordinate_offset():
    time = np.arange(2000) / 200
    position = np.column_stack([time, time * .4])
    a, _ = sequences.window_conditioning(time, position, 0, 2000)
    b, _ = sequences.window_conditioning(time, position + 1e6, 0, 2000)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
