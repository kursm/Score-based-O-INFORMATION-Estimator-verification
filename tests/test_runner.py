from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from soi_nuisance.config import grid, load_config, method_steps
from soi_nuisance.runner import run_one, configure_environment, plan, rng_state, restore_rng
from soi_nuisance.report import aggregates, make_report


@pytest.mark.parametrize("method", ["soi", "line_mine", "line_nwj"])
def test_gaussian_full_train_evaluate_cycle(tiny, tmp_path, method):
    tiny["methods"] = [method]
    result = run_one(tiny, next(grid(tiny)), tmp_path, configure_environment(tiny))
    assert result["status"] == "ok", result["failure"]
    assert result["training_steps"] == 4
    assert result["x_presentations"] == 128
    assert result["unique_train_seen"] == 64
    assert np.isfinite(result["omega_hat"])
    assert result["tc_hat"]-result["dtc_hat"] == pytest.approx(result["omega_hat"])
    if method != "soi":
        assert len(result["mi_terms"]) == 5
    assert (tmp_path/"runs"/result["run_id"]/"last.pt").exists()


@pytest.mark.parametrize("method", ["soi", "line_mine", "line_nwj"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_exact_resume_model_optimizer_rng_and_batch_state(tiny, tmp_path, method, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    tiny["device"] = device_name
    tiny["methods"] = [method]
    device = configure_environment(tiny)
    job = next(grid(tiny))
    first = run_one(tiny, job, tmp_path/"resumed", device)
    extended = deepcopy(tiny)
    extended["method_params"][method]["training_steps"] = 8
    resumed = run_one(extended, job, tmp_path/"resumed", device, resume=True)
    full = run_one(extended, job, tmp_path/"continuous", device)
    def load(folder, result):
        return torch.load(tmp_path/folder/"runs"/result["run_id"]/"last.pt",
                          map_location="cpu", weights_only=False)
    a, b = load("resumed", resumed), load("continuous", full)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for param_id, state in a["optimizer"]["state"].items():
        for key in state:
            torch.testing.assert_close(state[key], b["optimizer"]["state"][param_id][key], rtol=0, atol=0)
    for key in ["order", "seen", "rng"]:
        assert torch.equal(a["stream"][key], b["stream"][key])
    assert a["stream"]["offset"] == b["stream"]["offset"]
    assert torch.equal(a["train_rng"], b["train_rng"])
    assert resumed["omega_hat"] == full["omega_hat"]
    assert resumed["x_presentations"] == full["x_presentations"] == 256


def test_global_rng_roundtrip():
    state = rng_state()
    expected = torch.randn(10), np.random.rand(10)
    restore_rng(state)
    actual = torch.randn(10), np.random.rand(10)
    assert torch.equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])


def test_equal_time_budget_applies_to_whole_estimator(tiny, tmp_path):
    tiny.update(methods=["line_nwj"], budget_mode="equal_time", equal_time_seconds=.02)
    result = run_one(tiny, next(grid(tiny)), tmp_path, configure_environment(tiny))
    assert result["status"] == "ok"
    assert result["training_steps"] >= 1
    assert result["training_seconds"] >= .02
    assert result["architecture"]["critics"] == 5


def test_failures_are_preserved_and_not_silently_retried(tiny, tmp_path, monkeypatch):
    from soi_nuisance.estimators import Line
    tiny["methods"] = ["line_mine"]
    monkeypatch.setattr(Line, "loss", lambda self, x, g: torch.tensor(float("nan")))
    job = next(grid(tiny))
    result = run_one(tiny, job, tmp_path, configure_environment(tiny))
    assert result["status"] == "nan"
    assert (tmp_path/"runs"/result["run_id"]/"diagnostic.pt").exists()
    assert run_one(tiny, job, tmp_path, torch.device("cpu"), resume=True) == result


def test_ci_clusters_nested_repeats_and_retains_failures():
    common = dict(method="soi", k=1, n_train=10, evaluation_mode="vp_upstream_constant",
                  status="ok", training_seconds=1., training_steps=2)
    rows = [common | dict(distribution_seed=outer, training_seed=inner,
                         signed_error=err, squared_error=err**2)
            for outer, err in [(1, 1.), (2, 3.)] for inner in range(5)]
    result = aggregates(rows)[0]
    assert result["outer_repetitions"] == 2
    assert result["bias"] == 2
    assert result["rmse"] == pytest.approx(np.sqrt(5))
    result = aggregates(rows+[common | dict(status="nan", distribution_seed=3)])[0]
    assert result["failed_runs"] == 1
    assert result["metric_population"] == "conditional_on_success"


def test_presets_and_dry_run_sizes(tiny):
    result = plan(tiny)
    assert result["runs"] == 3 and result["datasets"] == 1
    assert result["uncompressed_data_bytes"] == (64+32+33)*3*4
    assert result["total_target_updates"] == 12
    for preset in ["smoke", "pilot", "paper", "full", "gaussian_control"]:
        load_config(preset)
    cfg = load_config("paper")
    assert method_steps(cfg, "soi") == 195000
    assert method_steps(cfg, "line_mine") == 40000
    cfg["num_blocks"] = 10
    assert method_steps(cfg, "line_nwj") == 80000
    with pytest.raises(ValueError):
        load_config(overrides=["missing_key=1"])


def test_outer_replicas_have_independent_init_and_pair_across_k(tiny):
    from soi_nuisance.runner import model_for
    job = next(grid(tiny))
    initial = model_for(tiny, job, torch.device("cpu"))
    other_outer = model_for(tiny, job | dict(distribution_seed=999), torch.device("cpu"))
    other_k = model_for(tiny, job | dict(k=16), torch.device("cpu"))
    a, b, c = [next(model.parameters()).detach() for model in [initial, other_outer, other_k]]
    assert not torch.equal(a, b)
    assert torch.equal(a, c)
