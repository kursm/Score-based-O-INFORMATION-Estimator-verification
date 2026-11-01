from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from soi_nuisance.config import ROOT, grid, load_config
from soi_nuisance.convergence import PlateauMonitor
from soi_nuisance.data import build_dataset, case_config, covariance, gaussian_truth
from soi_nuisance.runner import configure_environment, plan, run_one
from soi_nuisance.simple_experiment import DEFAULT_SETTINGS, RHO_VALUES, build_experiment
from conftest import extracted


def test_five_settings_produce_ten_true_information_points_and_original_optimizers():
    settings = json.loads((ROOT/"configs/experiment.json").read_text())
    assert set(settings) == set(DEFAULT_SETTINGS)
    config = build_experiment(settings)
    details = plan(config)
    assert details["runs"] == 30 and details["datasets"] == 10
    truth = [p["omega_true"] for p in details["curve_points"]]
    assert len(set(truth)) == 10 and min(truth) < 0 < max(truth)
    assert config["methods"] == ["soi", "line_mine", "line_nwj"]
    assert config["method_params"]["soi"]["optimizer"]["lr"] == .01
    for method in ["line_mine", "line_nwj"]:
        assert config["method_params"][method]["optimizer"]["lr"] == .001
        assert config["method_params"][method]["batch_size"] == 64
    other = plan(build_experiment(settings | dict(v_dim=3, k=64)))
    np.testing.assert_allclose([p["omega_true"] for p in other["curve_points"]], truth)
    scaled = plan(build_experiment(settings | dict(u_dim=3)))
    np.testing.assert_allclose([p["omega_true"] for p in scaled["curve_points"]], np.array(truth)*3)
    loaded = load_config("experiment", overrides=["n_samples=256", "k=1", "max_steps=3"])
    assert loaded["train_sizes"] == [256] and loaded["k_values"] == [1]
    assert all(p["training_steps"] == 3 for p in loaded["method_params"].values())


def test_mixed_curve_covariance_matches_authors_and_fixed_nuisance_across_rho():
    config = build_experiment(DEFAULT_SETTINGS | dict(n_samples=256))
    config.update(n_val=32, n_test=33)
    original = extracted("vendor/soi/src/benchmark/synthetic_dataset.py")
    for rho in [RHO_VALUES[0], RHO_VALUES[-1]]:
        expected = original["combined_cov"]([
            original["Task_redundant"](3, rho=rho).cov,
            original["Task_redundant"](4, rho=rho).cov,
            original["Task_synergy"](3, rho=.7).cov])
        np.testing.assert_array_equal(covariance(case_config(config, {"rho":rho})), expected)
    a, ma = build_dataset(case_config(config, {"rho":RHO_VALUES[0]}), 16, 77, 77)
    b, mb = build_dataset(case_config(config, {"rho":RHO_VALUES[-1]}), 16, 77, 77)
    assert ma["truth"] != mb["truth"]
    for split in ["train", "val", "test"]:
        np.testing.assert_array_equal(a[split][:, :, 1:], b[split][:, :, 1:])
        assert not np.array_equal(a[split][:, :, :1], b[split][:, :, :1])
    assert config["mixed_subsystems"][0]["rho"] == RHO_VALUES[0]


@pytest.mark.parametrize("method", ["soi", "line_mine", "line_nwj"])
def test_plateau_stops_before_cap_and_monitor_resumes_exactly(tiny, tmp_path, monkeypatch, method):
    import soi_nuisance.convergence as convergence
    cfg = deepcopy(tiny)
    cfg["methods"] = [method]
    cfg["method_params"][method]["training_steps"] = 20
    cfg["convergence"] = dict(every=2, min_steps=2, patience=2, relative_tolerance=.001,
                              smoothing=1., n_samples=32)
    monkeypatch.setattr(convergence, "validation_objective", lambda *args: 1.)
    job = next(grid(cfg))
    device = configure_environment(cfg)
    count = 0
    def stop_at_five():
        nonlocal count
        count += 1
        return count > 5
    with pytest.raises(KeyboardInterrupt):
        run_one(cfg, job, tmp_path/"resumed", device, stop_requested=stop_at_five)
    result = run_one(cfg, job, tmp_path/"resumed", device, resume=True)
    continuous = run_one(cfg, job, tmp_path/"continuous", device)
    assert result["status"] == "ok" and result["stop_reason"] == "converged"
    assert result["training_steps"] == continuous["training_steps"] == 6
    def checkpoint(directory, result):
        return torch.load(directory/"runs"/result["run_id"]/"last.pt", weights_only=False, map_location="cpu")
    a = checkpoint(tmp_path/"resumed", result)
    b = checkpoint(tmp_path/"continuous", continuous)
    assert a["state"]["convergence"] == b["state"]["convergence"]
    assert a["state"]["convergence"]["converged"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert run_one(cfg, job, tmp_path/"resumed", device, resume=True) == result


def test_plateau_warmup_and_improvement_reset_patience():
    monitor = PlateauMonitor(dict(min_steps=4, patience=2, relative_tolerance=.01, smoothing=1.))
    assert not monitor.update(1, 1.)
    assert not monitor.update(2, 1.)
    assert not monitor.update(4, .8)
    assert not monitor.update(5, .8)
    assert not monitor.update(6, .6)
    assert monitor.state["stale"] == 0
    assert not monitor.update(7, .6)
    assert monitor.update(8, .6)


@pytest.mark.parametrize("method", ["soi", "line_mine", "line_nwj"])
def test_real_validation_objective_is_fixed_and_does_not_change_training_rng(tiny, method):
    from soi_nuisance.convergence import validation_objective
    from soi_nuisance.runner import model_for
    cfg = deepcopy(tiny)
    cfg["methods"] = [method]
    device = configure_environment(cfg)
    model = model_for(cfg, next(grid(cfg)), device)
    data = torch.randn(32, 3, 1)
    parameters = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    a = validation_objective(model, data, 16, 123, 32)
    b = validation_objective(model, data, 16, 123, 32)
    assert np.isfinite(a) and a == b
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
