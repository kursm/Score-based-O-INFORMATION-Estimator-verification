from collections import Counter
from copy import deepcopy
import itertools
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from soi_nuisance.estimators import SOI, Critic, Line
from soi_nuisance.sde import VPSDE, training_masks
from conftest import module_file, extracted


@pytest.mark.parametrize("kind,klass_name", [("mine", "MINE"), ("nwj", "NWJ")])
def test_mi_values_and_parameter_gradients_match_soi_upstream(kind, klass_name):
    up = module_file("vendor/soi/src/baseline/TC_estimation/mi_estimators.py")
    torch.manual_seed(12)
    ours = Critic(2, 3, 8, kind).double()
    original = getattr(up, klass_name)(2, 3, 8).double()
    getattr(original, "T_func" if kind == "mine" else "F_func").load_state_dict(ours.net.state_dict())
    x, y = torch.randn(7, 2, dtype=torch.float64), torch.randn(7, 3, dtype=torch.float64)
    torch.manual_seed(44)
    expected = original.forward(x, y) if kind == "mine" else original.forward_(x, y)
    expected.backward()
    expected_grads = [p.grad.clone() for p in original.parameters()]
    torch.manual_seed(44)
    actual = ours.bound(x, y, pair_chunk=5)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-13)
    for a, b in zip(ours.parameters(), expected_grads):
        torch.testing.assert_close(a.grad, b, atol=2e-14, rtol=2e-12)


@pytest.mark.parametrize("chunk", [1, 3, 10, 1000])
def test_nwj_chunk_value_and_input_parameter_gradients(chunk):
    torch.manual_seed(16)
    dense, split = Critic(2, 2, 5, "nwj").double(), Critic(2, 2, 5, "nwj").double()
    split.load_state_dict(dense.state_dict())
    x = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    xx, yy = x.detach().clone().requires_grad_(), y.detach().clone().requires_grad_()
    a, b = dense.bound(x, y, pair_chunk=0), split.bound(xx, yy, pair_chunk=chunk)
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-13)
    for left, right in [(x.grad, xx.grad), (y.grad, yy.grad)] + [
        (p.grad, q.grad) for p, q in zip(dense.parameters(), split.parameters())]:
        torch.testing.assert_close(left, right, atol=2e-14, rtol=2e-12)


def test_tc_original_mi_objectives(tiny):
    original = module_file("vendor/tc_estimation/mi_estimators.py")
    # TC-estimation NWJ has one hidden layer. SOI's fork has two; keep that distinction.
    ours = Critic(2, 1, 8, "nwj", hidden_layers=1).double()
    ref = original.NWJ(2, 1, 8).double()
    ref.F_func.load_state_dict(ours.net.state_dict())
    x, y = torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 1, dtype=torch.float64)
    torch.testing.assert_close(ours.bound(x, y, pair_chunk=3), ref(x, y), rtol=1e-13, atol=1e-14)


def test_soi_mlp_architecture_and_gradients_are_verbatim(tiny):
    params = tiny["method_params"]["soi"]
    ours = SOI(3, 1, params)
    original = module_file("vendor/soi/src/models/mlp.py").UnetMLP_simple(
        3, init_dim=16, time_dim=16, nb_var=3, dim_mults=[])
    assert list(ours.score.state_dict()) == list(original.state_dict())
    with torch.no_grad():
        ours.score.proj.weight.normal_(std=.01)
    original.load_state_dict(ours.score.state_dict())
    x, tau = torch.randn(7, 3), torch.rand(7, 3)
    actual, expected = ours.score(x, t=tau), original(x, t=tau)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in zip(ours.score.parameters(), original.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_importance_inverse_and_vp_coefficients_match_upstream():
    original = module_file("vendor/soi/src/libs/importance.py").VariancePreservingTruncatedSampling()
    ours = VPSDE()
    u = torch.tensor([0., .01, .05, .15, .5, .9, 1.])
    expected32 = original.inv_Phi(u, torch.tensor([1.]))
    actual = ours.inv_cdf(u)
    # Upstream float32 cancellation at epsilon perturbs r_eps and the normalizer.
    # Evaluate the unchanged original algebra in float64 to test the formula itself.
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        expected64 = original.inv_Phi(u.double(), torch.tensor([1.]))
        constant64 = original.normalizing_constant(torch.tensor([1.])).item()
    finally:
        torch.set_default_dtype(old_dtype)
    torch.testing.assert_close(actual, expected64, rtol=2e-11, atol=2e-13)
    assert ours.normalizer == pytest.approx(constant64, rel=2e-12)
    assert (actual.float()-expected32).abs().max().item() < 3e-5
    t = torch.linspace(.001, 1, 100, dtype=torch.float64)
    mean, std = ours.mean_std(t)
    torch.testing.assert_close(mean, original.mean_weight(t))
    torch.testing.assert_close(std, original.std(t))


def test_soi_mask_frequencies_loss_and_gradient_match_upstream(tiny, monkeypatch):
    from soi_nuisance.estimators import SOI
    importance = module_file("vendor/soi/src/libs/importance.py")
    util = extracted("vendor/soi/src/libs/util.py", ["concat_vect", "expand_mask"])
    namespace = extracted("vendor/soi/src/libs/SDE.py", ["VP_SDE"], dict(
        itertools=itertools, **util, sample_vp_truncated_q=importance.sample_vp_truncated_q))
    cls = namespace["VP_SDE"]
    ref = cls.__new__(cls)
    ref.nb_var, ref.weight_s_functions, ref.device = 3, True, "cpu"
    ref.masks = ref.get_masks_training(1)
    assert Counter(map(tuple, ref.masks.tolist())) == Counter(map(tuple, training_masks(3).tolist()))
    ref.rand_batch, ref.importance_sampling, ref.marginals, ref.var_sizes = True, True, 1, [1, 1, 1]
    ours = SOI(3, 1, tiny["method_params"]["soi"])
    ref.masks = ours.masks.clone()
    with torch.no_grad():
        ours.score.proj.weight.normal_(std=.01)
    b = 7
    x, noise, missing_noise = torch.randn(b, 3), torch.randn(b, 3), torch.randn(b, 3)
    t = torch.linspace(.01, .8, b)[:, None]
    indices = torch.arange(b) % len(ours.masks)
    monkeypatch.setattr(torch, "randint", lambda *a, **kw: indices+1)
    monkeypatch.setattr(torch, "randn_like", lambda *a, **kw: missing_noise)
    ref.sample_importance_sampling_t = lambda shape: t
    mean, std = ours.sde.mean_std(t)
    ref.sample = lambda x_0, t: (mean*x_0+std*noise, noise, mean, std)
    data = {str(i): x[:, i:i+1] for i in range(3)}
    expected = ref.train_step(data, lambda x, t, mask, std: ours.predict(x, t, mask)).mean()
    expected.backward()
    gradients = [p.grad.clone() if p.grad is not None else None for p in ours.score.parameters()]
    ours.zero_grad(set_to_none=True)
    actual = ours.loss_fixed(x.reshape(b, 3, 1), t, noise, ours.masks[indices], missing_noise)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for p, g in zip(ours.score.parameters(), gradients):
        if g is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, g, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("importance", [False, True])
def test_score_difference_integrals_match_upstream(importance):
    up_importance = module_file("vendor/soi/src/libs/importance.py")
    ns = extracted("vendor/soi/src/libs/info_measures.py", ["get_tc", "get_dtc"], dict(
        concat_vect=lambda d: torch.cat(list(d.values()), -1),
        get_normalizing_constant=up_importance.get_normalizing_constant))
    sde = VPSDE(importance_sampling=importance)
    t = torch.linspace(.01, .95, 9)[:, None]
    j, m, c = torch.randn(9, 6), torch.randn(9, 6), torch.randn(9, 6)
    std = sde.mean_std(t)[1]
    js, ms, cs = (j, m, c) if importance else (j/std, m/std, c/std)
    make_dict = lambda x: {str(i): x[:, i*2:(i+1)*2] for i in range(3)}
    g = sde.beta(t).sqrt()
    tc = ns["get_tc"](SimpleNamespace(device="cpu"), make_dict(js), make_dict(ms), g, importance)
    dtc = ns["get_dtc"](SimpleNamespace(device="cpu"), make_dict(js), make_dict(cs), g, importance)
    weight = sde.integral_weight(t)
    # Uniform time mode includes the interval-length correction; upstream omitted it.
    correction = 1 if importance else 1-sde.uniform_epsilon
    assert (weight*(j-m).double().square()).sum().item()/len(t) == pytest.approx(tc*correction, rel=3e-5)
    assert (weight*(j-c).double().square()).sum().item()/len(t) == pytest.approx(dtc*correction, rel=3e-5)


def test_line_decomposition_has_independent_duplicate_terms(tiny):
    model = Line(3, 1, tiny["method_params"]["line_mine"], "mine")
    assert len(model.critics) == 5
    assert model.pairs[0][:2] == ((1, 2), (0,))
    assert model.pairs[-1][:2] == ((0, 1), (2,))
    assert model.critics[2] is not model.critics[-1]


@pytest.mark.parametrize("kind,mi_e", [("mine", "MINE"), ("nwj", "NWJ")])
def test_full_baseline_assembly_loss_and_gradients_match_upstream(tiny, kind, mi_e):
    import types
    import torch.nn as nn
    up_mi = module_file("vendor/soi/src/baseline/TC_estimation/mi_estimators.py")
    ns = extracted("vendor/soi/src/baseline/baseline.py", ["O_Estimator"], dict(
        nn=nn, pl=types.SimpleNamespace(LightningModule=nn.Module),
        MI_CLASS={"MINE": up_mi.MINE, "NWJ": up_mi.NWJ}))
    torch.manual_seed(9)
    ref = ns["O_Estimator"](types.SimpleNamespace(dim=1, mi_e=mi_e), nb_var=3)
    torch.manual_seed(9)
    ours = Line(3, 1, tiny["method_params"]["line_"+kind], kind)
    originals = list(ref.mi_estimator_list_s)+list(ref.mi_estimator_list_tc)
    for critic, reference in zip(ours.critics, originals):
        for a, b in zip(critic.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    x = torch.randn(8, 3, 1)
    samples = [x[:, i] for i in range(3)]
    torch.manual_seed(71)
    expected = torch.stack(ref.learning_loss_tc(samples)+ref.learning_loss_s(samples)).mean()
    expected.backward()
    torch.manual_seed(71)
    actual = ours.loss(x, None)
    actual.backward()
    torch.testing.assert_close(actual.float(), expected, rtol=2e-5, atol=1e-6)
    for critic, reference in zip(ours.critics, originals):
        for a, b in zip(critic.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-7)


def test_ema_aliases_and_values_match_original(tiny):
    import torch.nn as nn
    ours = SOI(3, 1, tiny["method_params"]["soi"])
    ns = extracted("vendor/soi/src/libs/util.py", ["EMA"], dict(nn=nn, deepcopy=deepcopy))
    reference = ns["EMA"](ours.score, decay=.999)
    for _ in range(3):
        with torch.no_grad():
            for p in ours.score.parameters():
                p.add_(torch.randn_like(p)*.01)
        reference.update(ours.score)
        ours.update_ema()
        for key, value in ours.ema.state_dict().items():
            torch.testing.assert_close(value, reference.module.state_dict()[key], rtol=0, atol=0)
