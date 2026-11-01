import numpy as np
import pytest
from soi_nuisance.density import LinearDensity
from soi_nuisance.data import build_dataset, gaussian_truth, covariance, scalar_covariance
from conftest import extracted


@pytest.mark.parametrize("k", [1, 3, 16, 256])
def test_density_integrals_cdf_and_exact_sampler(k):
    d = LinearDensity.random(k, np.random.default_rng(11))
    points = np.linspace(0, k+1, (k+1)*20+1)
    assert np.all(d.pdf(points) >= 0)
    # Trapezoid integration is exact on each linear segment.
    assert np.trapezoid(d.pdf(points), points) == pytest.approx(1., abs=2e-15)
    assert np.array_equal(d.cdf([-np.inf, 0, k+1, np.inf]), [0, 0, 1, 1])
    assert np.all(np.diff(d.cdf(points)) >= -1e-15)
    values = d.sample(100000, np.random.default_rng(123))
    uniform = np.sort(d.cdf(values))
    ranks = np.arange(1, len(values)+1)/len(values)
    assert np.max(np.abs(uniform-ranks)) < .006
    assert abs(values.mean()-d.mean) < 6*d.std/np.sqrt(len(values))
    assert values.var() == pytest.approx(d.variance, rel=.025)
    standardized = d.sample(100000, np.random.default_rng(77), standardized=True)
    assert abs(standardized.mean()) < .02
    assert standardized.var() == pytest.approx(1., rel=.025)


def test_triangle_and_zero_mass_segments():
    a, b = LinearDensity([0, .1, 0]), LinearDensity([0, .9, 0])
    assert a.mean == pytest.approx(1)
    assert a.variance == pytest.approx(1/6)
    assert np.allclose(a.pdf(np.linspace(0, 2, 100)), b.pdf(np.linspace(0, 2, 100)))
    zero = LinearDensity([0, 0, 1, 0, 0])
    values = zero.sample(50000, np.random.default_rng(7))
    assert values.min() >= 1 and values.max() <= 3
    for heights in [[0, 0, 0], [1, 1, 0], [0, -1, 0]]:
        with pytest.raises(ValueError):
            LinearDensity(heights)


def test_fixed_densities_splits_nested_prefix_and_u_across_k(tiny):
    tiny["v_dim"] = 2
    arrays, meta = build_dataset(tiny, 1, 11, 21)
    arrays2, meta2 = build_dataset(tiny, 16, 11, 21)
    again, same = build_dataset(tiny, 1, 11, 21)
    for split in ["train", "val", "test"]:
        assert np.array_equal(arrays[split][:, :, :1], arrays2[split][:, :, :1])
        assert np.array_equal(arrays[split], again[split])
        assert meta["hashes"][split] == same["hashes"][split]
    assert not np.array_equal(arrays["train"][:32], arrays["val"])
    assert not np.array_equal(arrays["test"][:32], arrays["val"])
    assert meta["truth"] == meta2["truth"]
    assert len(meta["densities"]) == tiny["num_blocks"]*2
    # The pool is generated before choosing n; smaller n uses literal prefixes.
    assert np.array_equal(arrays["train"][:16], arrays["train"][:32][:16])


@pytest.mark.parametrize("kind", ["redundancy", "synergy", "mixed"])
@pytest.mark.parametrize("du", [1, 3])
def test_ground_truth_and_covariance_match_original_generator(tiny, kind, du):
    upstream = extracted("vendor/soi/src/benchmark/synthetic_dataset.py")
    n = 6 if kind == "mixed" else 3
    cfg = tiny | dict(benchmark=kind, num_blocks=n, u_dim=du)
    cov = covariance(cfg)
    if kind == "mixed":
        tasks = [upstream["Task_redundant"](3, rho=.6, dim=du),
                 upstream["Task_synergy"](3, rho=.8, dim=du)]
        task = upstream["Task_combination"](tasks, dim=du)
    else:
        cls = upstream["Task_redundant" if kind == "redundancy" else "Task_synergy"]
        task = cls(n, rho=cfg["rho"], dim=du)
    assert np.array_equal(cov, np.kron(task.cov, np.eye(du)))
    truth = gaussian_truth(cov, n, du)
    for key, value in [("tc_true", task.tc()), ("dtc_true", task.dtc()), ("omega_true", task.o_inf())]:
        assert truth[key] == pytest.approx(value, abs=2e-13)
    # Analytic independent Gaussian nuisance check, arranged inside each Xi block.
    full = np.zeros((n*(du+2), n*(du+2)))
    uids = np.array([i*(du+2)+j for i in range(n) for j in range(du)])
    vids = np.setdiff1d(np.arange(len(full)), uids)
    full[np.ix_(uids, uids)] = cov
    full[vids, vids] = np.linspace(.4, 3., len(vids))
    augmented = gaussian_truth(full, n, du+2)
    assert truth == pytest.approx(augmented, abs=5e-14)


def test_invalid_covariance_rejected_without_jitter(tiny):
    with pytest.raises(np.linalg.LinAlgError):
        scalar_covariance("redundancy", 3, 1.)
    with pytest.raises(np.linalg.LinAlgError):
        scalar_covariance("synergy", 3, 1.1)
    with pytest.raises(ValueError):
        covariance(tiny | dict(benchmark="mixed", num_blocks=5))
