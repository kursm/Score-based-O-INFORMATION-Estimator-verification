"""Fixed finite pools, independent RNG channels, blockwise analytic Gaussian truth."""
import hashlib
import copy
import json
from pathlib import Path
import time
import numpy as np
from .density import LinearDensity


def seed_for(*parts):
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63-1)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def scalar_covariance(kind, n, rho):
    if n < 3 or not np.isfinite(rho):
        raise ValueError("Each Gaussian subsystem needs >=3 blocks and finite rho")
    if kind == "redundancy":
        cov = (1-rho)*np.eye(n) + rho*np.ones((n, n))
    elif kind == "synergy":
        cov = np.eye(n)
        cov[0, 1] = cov[1, 0] = 1/np.sqrt(n-1)
        cov[1, 2:] = rho/np.sqrt(n-1)
        cov[2:, 1] = rho/np.sqrt(n-1)
    else:
        raise ValueError("Unknown Gaussian benchmark: "+str(kind))
    # No hidden jitter.
    np.linalg.cholesky(cov)
    return cov


def covariance(config):
    n, du = config["num_blocks"], config["u_dim"]
    subs = config["mixed_subsystems"] if config["benchmark"] == "mixed" else [
        dict(benchmark=config["benchmark"], size=n, rho=config["rho"])]
    if sum(s["size"] for s in subs) != n:
        raise ValueError("Subsystem sizes must sum to num_blocks")
    cov = np.zeros((n, n), dtype=np.float64)
    offset = 0
    for s in subs:
        size = s["size"]
        cov[offset:offset+size, offset:offset+size] = scalar_covariance(
            s["benchmark"], size, s.get("rho", config["rho"]))
        offset += size
    return np.kron(cov, np.eye(du))


def case_config(config, job):
    """Resolve one correlation point without mutating the shared experiment."""
    if "rho" not in job:
        return config
    resolved = dict(config, rho=job["rho"])
    if config["benchmark"] == "mixed":
        resolved["mixed_subsystems"] = copy.deepcopy(config["mixed_subsystems"])
        for index in config.get("rho_subsystem_indices", []):
            resolved["mixed_subsystems"][index]["rho"] = job["rho"]
    return resolved


def logdet(cov):
    return float(2*np.log(np.diag(np.linalg.cholesky(cov))).sum())


def gaussian_truth(cov, n, du):
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (n*du, n*du) or not np.allclose(cov, cov.T):
        raise ValueError("Invalid block covariance")
    joint = logdet(cov)
    marg, minus = [], []
    for i in range(n):
        ids = np.arange(i*du, (i+1)*du)
        others = np.setdiff1d(np.arange(n*du), ids)
        marg.append(logdet(cov[np.ix_(ids, ids)]))
        minus.append(logdet(cov[np.ix_(others, others)]))
    tc = .5*(sum(marg)-joint)
    dtc = .5*(sum(minus)-(n-1)*joint)
    return dict(tc_true=tc, dtc_true=dtc, omega_true=tc-dtc)


def dataset_spec(config, k, distribution_seed, data_seed, v_dim):
    return {key: config[key] for key in
            ["benchmark", "rho", "mixed_subsystems", "num_blocks", "u_dim",
             "normalization", "transformation"]} | dict(
        k=k, v_dim=v_dim, distribution_seed=distribution_seed, data_seed=data_seed,
        n_pool=max(config["train_sizes"]), n_val=config["n_val"], n_test=config["n_test"],
        generator_version=1)


def build_dataset(config, k, distribution_seed, data_seed, v_dim=None):
    start = time.perf_counter()
    vd = config["v_dim"] if v_dim is None else v_dim
    spec = dataset_spec(config, k, distribution_seed, data_seed, vd)
    n, du = config["num_blocks"], config["u_dim"]
    cov = covariance(config)
    densities = [LinearDensity.random(k, np.random.default_rng(seed_for(
        "density", distribution_seed, k, i, r)))
        for i in range(n) for r in range(vd)]
    arrays, hashes = {}, {}
    for split, count in [("train", spec["n_pool"]), ("val", spec["n_val"]),
                         ("test", spec["n_test"])]:
        # Outer seed is a namespace for independent outer replications.
        # Neither k nor vd occurs in U/permutation RNG keys.
        rng_u = np.random.default_rng(seed_for("U", distribution_seed, data_seed, split))
        u = (rng_u.standard_normal((count, n*du)) @ np.linalg.cholesky(cov).T
             ).reshape(count, n, du)
        v = np.empty((count, n, vd), dtype=np.float64)
        for i in range(n):
            for r in range(vd):
                rng_v = np.random.default_rng(seed_for(
                    "V", distribution_seed, data_seed, split, k, i, r))
                v[:, i, r] = densities[i*vd+r].sample(
                    count, rng_v, standardized=config["normalization"] == "population")
        x = np.concatenate([u, v], axis=2).astype(np.float32)
        permutation = np.random.default_rng(seed_for(
            "permutation", distribution_seed, data_seed, split)).permutation(count)
        arrays[split] = x[permutation]
        arrays[split+"_permutation"] = permutation
        hashes[split] = digest(arrays[split])
    meta = dict(spec=spec, densities=[d.metadata() for d in densities],
                covariance=cov.tolist(), truth=gaussian_truth(cov, n, du), hashes=hashes,
                generation_seconds=time.perf_counter()-start)
    return arrays, meta


def get_dataset(config, output, k, ds, xs, v_dim):
    spec = dataset_spec(config, k, ds, xs, v_dim)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(output)/"data"/key
    meta_path, arrays_path = directory/"metadata.json", directory/"arrays.npz"
    if meta_path.exists() and arrays_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if any(digest(arrays[s]) != meta["hashes"][s] for s in ["train", "val", "test"]):
            raise ValueError("Saved dataset hash mismatch")
        return arrays, meta, directory, True
    arrays, meta = build_dataset(config, k, ds, xs, v_dim)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return arrays, meta, directory, False
