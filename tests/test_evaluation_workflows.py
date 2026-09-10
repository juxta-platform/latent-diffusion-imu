"""Sampling, resume, manifest and offline evaluation integration regressions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ldm.evaluation import batch, generate, paths, report, stats
from scripts import compare_ldm_modes, eval_classifier, eval_ldm, eval_vae, generate_ronin_dataset


class IdentityLDM:
    use_ema = False
    num_timesteps = 10
    scale_factor = torch.tensor(2.)
    first_stage_model = SimpleNamespace(embed_dim=6)

    def encode_first_stage(self, x):
        return SimpleNamespace(mode=lambda: x, sample=lambda: x)

    def decode_first_stage(self, z):
        return z

    def p_losses(self, z, t, cond):
        return torch.tensor(0.)


class ProbeSampler:
    def __init__(self):
        self.calls = []

    def sample(self, S, batch_size, shape, conditioning, **kwargs):
        self.calls.append(("noise", conditioning))
        return torch.zeros(batch_size, *shape), None

    def sample_img2img(self, S, x0, conditioning, **kwargs):
        self.calls.append(("img2img", conditioning))
        return x0 + 1, None


@pytest.mark.parametrize("mode,strength,expected,path", [
    ("traj", 0., 3., None), ("traj", .5, 3.5, "img2img"),
    ("traj", 1., 0., "noise"), ("sim_cond", 1., 0., "noise"),
])
def test_split_sampling_uses_synthetic_initializer(mode, strength, expected, path):
    item = dict(imu=torch.ones(1, 6, 4) * 8, sim_imu=torch.ones(1, 6, 4) * 3,
                velocity=torch.zeros(1, 2, 4), physical_time=torch.zeros(1, 1, 4))
    args = SimpleNamespace(mode=mode, strength=strength, latent_length=4, no_ema=True,
                           max_batches=None, ddim_steps=5, ddim_eta=0.)
    sampler = ProbeSampler()
    _, _, generated = eval_ldm.evaluate_split_windows(IdentityLDM(), sampler, [item], "cpu", args)
    torch.testing.assert_close(generated, torch.full_like(generated, expected))
    if path is None:
        assert sampler.calls == []
    else:
        assert sampler.calls[0][0] == path
        cond = sampler.calls[0][1]
        assert ("sim_latent" in cond) == (mode == "sim_cond")
        if mode == "sim_cond":
            torch.testing.assert_close(cond["sim_latent"], item["sim_imu"] * 2)


def test_full_recording_img2img_never_encodes_real_input():
    raw = np.full((8, 6), 9.)
    sim = np.full((8, 6), 3.)
    st = dict(imu_mean=torch.zeros(6, 1), imu_std=torch.ones(6, 1),
              vel_mean=torch.zeros(2, 1), vel_std=torch.ones(2, 1))
    kw = dict(features=raw, ts=np.arange(8) / 200, gt_pos=np.zeros((8, 3)),
              model=IdentityLDM(), sampler=ProbeSampler(), stats=st, device="cpu",
              window=4, latent_length=4, strength=0.)
    generated = generate.ldm_generate(**kw, initial_features=sim)
    np.testing.assert_array_equal(generated, sim)
    with pytest.raises(ValueError, match="initial_features"):
        generate.ldm_generate(**kw)
    with pytest.raises(ValueError, match="pure noise"):
        generate.ldm_generate(**kw, sim_features=sim, initial_features=sim)


def test_manifest_paths_are_used_without_fixed_pair_filenames(tmp_path):
    import csv

    root = tmp_path / "data" / "pairs"
    root.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    for filename in ("motion.txt", "real.h5", "imu.parquet"):
        (external / filename).touch()
    row = dict(pair_id="chest/walk", status="generated", trajectory_txt="external/motion.txt",
               real_hdf5=str(external / "real.h5"), synthetic_parquet="external/imu.parquet")
    with (root / "manifest.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    pairs = generate_ronin_dataset.discover_pairs(SimpleNamespace(input_dir=str(root), sim_dir=None), "input_dir")
    assert len(pairs) == 1 and pairs[0].ready("traj") and pairs[0].ready("sim_cond")
    assert pairs[0].synthetic == str(external / "imu.parquet")


def test_comparison_default_and_resume_missing_method(tmp_path, monkeypatch):
    parser = compare_ldm_modes.build_parser()
    args = parser.parse_args(["--input", "unused", "--vae_ckpt", "unused", "--ronin_ckpt", "unused",
                              "--ldm_ckpt", "traj", "--ldm_sim_cond_ckpt", "sim",
                              "--outdir", str(tmp_path)])
    assert args.strength == .5
    args.force = False
    recording = paths.Recording(name="pair", path="real.hdf5", sim_path="sim.parquet")
    dest = tmp_path / "pair"
    report.write_json({}, str(dest), "metrics.json")
    for name in ("real_ldm", "synthetic_ldm"):
        report.write_json({}, str(dest / name), "metrics.json")
        (dest / name / "trajectories.npz").touch()
    calls = []
    monkeypatch.setattr(compare_ldm_modes, "run_pass", lambda name, *a: calls.append(name))
    monkeypatch.setattr(compare_ldm_modes, "plot_pair", lambda *a: {"methods": {}})
    batch.run_batch(
        [recording], str(tmp_path),
        lambda item, out: compare_ldm_modes.compare_pair(item, out, {"sim_cond": {}}, None, "cpu", args),
        complete=lambda item, out: all(
            batch.is_complete(str(dest / name), ("trajectories.npz",))
            for name, _ in compare_ldm_modes.plan_runs(item, args)),
    )
    assert calls == ["sim_cond_ldm"]


def test_comparison_noise_is_independent_of_previous_runs(tmp_path, monkeypatch):
    time = np.arange(400) / 200
    dataset = SimpleNamespace(features=[np.zeros((400, 6))], ts_full=[time],
                              gt_pos_full=[np.zeros((400, 3))])
    # run_pass only asks len(dataset) for logging before the mocked inference.
    class Dataset(SimpleNamespace):
        def __len__(self):
            return 20
    dataset = Dataset(**vars(dataset))
    monkeypatch.setattr(compare_ldm_modes.ronin, "load_strided_dataset", lambda *a, **kw: dataset)
    noise = []
    def sample(features, *a, **kw):
        noise.append(torch.randn(4))
        return features
    monkeypatch.setattr(generate, "ldm_generate", sample)
    monkeypatch.setattr(compare_ldm_modes.ronin, "with_features", lambda ds, features: ds)
    result = dict(ate=0., rte=0., rte_short=0., pos_gt=np.zeros((400, 2)), pos_pred=np.zeros((400, 2)))
    monkeypatch.setattr(compare_ldm_modes.ronin, "run_ronin_pipeline", lambda *a, **kw: result)
    args = SimpleNamespace(seed=42, ronin_step=10, ronin_window=200, ddim_steps=5, ddim_eta=0.,
                           no_ema=True, rte_delta_sec=10., ronin_3d=False, ronin_ckpt="unused")
    spec = dict(imu_path="real.hdf5", cond_path="real.hdf5", sim_path=None, strength=1., trim=False)
    bundle = dict(ldm=None, sampler=None, stats=None, ckpt="unused")
    for _ in range(2):
        torch.randn(33)
        compare_ldm_modes.run_pass("real_ldm", spec, bundle, None, str(tmp_path), "cpu", args)
    torch.testing.assert_close(noise[0], noise[1])


@pytest.mark.parametrize("module", [eval_vae, eval_ldm, eval_classifier])
def test_dry_run_never_loads_models(module, monkeypatch, tmp_path):
    flags = ["script", "--input", "missing.hdf5", "--vae_ckpt", "missing.ckpt", "--dry_run",
             "--outdir", str(tmp_path / "outputs")]
    if module is eval_classifier:
        flags += ["--clf_ckpt", "missing.pt"]
    elif module is eval_ldm:
        flags += ["--ldm_ckpt", "missing.ckpt"]
    monkeypatch.setattr("sys.argv", flags)
    monkeypatch.setattr(module, "load_models", lambda *a: pytest.fail("models loaded in dry run"))
    module.main()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("module", [eval_vae, eval_ldm])
def test_signal_plots_only_without_models(module, monkeypatch, tmp_path):
    x = np.ones((1, 6, 4), dtype=np.float32)
    report.save_signal_plot_data(str(tmp_path), x, x * 2, x, x * 2, 1)
    report.write_json({"source": "cached", "sample_rate": 200}, str(tmp_path), "metrics.json")
    calls = []
    monkeypatch.setattr(module.plots, "plot_signal_overlays", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(module, "load_models", lambda *a: pytest.fail("models loaded while replotting"))
    flags = ["script", "--input", "missing.hdf5", "--vae_ckpt", "missing.ckpt", "--plots_only",
             "--outdir", str(tmp_path)]
    if module is eval_ldm:
        flags += ["--ldm_ckpt", "missing.ckpt"]
    monkeypatch.setattr("sys.argv", flags)
    module.main()
    assert len(calls) == 2


def test_classifier_plots_only_without_models(monkeypatch, tmp_path):
    report.save_plot_data(str(tmp_path), windows=np.ones((1, 6, 4)))
    report.write_json(dict(class_names=["chest", "pocket"], predictions=["chest"],
                           logits=[[2., 1.]], window_gt=["chest"]), str(tmp_path), "results.json")
    monkeypatch.setattr(eval_classifier, "load_models", lambda *a: pytest.fail("models loaded while replotting"))
    calls = []
    monkeypatch.setattr(eval_classifier.plots, "plot_timeline", lambda *a, **kw: calls.append("timeline"))
    monkeypatch.setattr(eval_classifier.plots, "plot_imu_window", lambda *a, **kw: calls.append("window"))
    monkeypatch.setattr("sys.argv", ["script", "--input", "missing.hdf5", "--vae_ckpt", "missing",
                                     "--clf_ckpt", "missing", "--plots_only", "--outdir", str(tmp_path)])
    eval_classifier.main()
    assert calls == ["timeline", "window"]
