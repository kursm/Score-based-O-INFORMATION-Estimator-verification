import copy
import json
import math
from pathlib import Path
from .data import covariance, case_config, gaussian_truth

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("soi", "line_mine", "line_nwj")


def merge(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        if key not in out:
            raise ValueError("Unknown config key: "+key)
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out[key], dict) else value
    return out


def read_config(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "n_samples" in data and "max_steps" in data:
        from .simple_experiment import build_experiment
        return build_experiment(data)
    parent = data.pop("extends", None)
    return merge(read_config(path.parent/parent), data) if parent else data


def load_config(preset="smoke", config=None, overrides=()):
    cfg = read_config(config or ROOT/"configs"/(preset+".json"))
    if cfg.get("simple_settings"):
        from .simple_experiment import DEFAULT_SETTINGS, build_experiment
        settings = dict(cfg["simple_settings"])
        remaining = []
        changed = False
        for item in overrides:
            key, _, value = item.partition("=")
            if key in DEFAULT_SETTINGS:
                settings[key] = json.loads(value)
                changed = True
            else:
                remaining.append(item)
        if changed:
            cfg = build_experiment(settings, cfg["output_dir"], cfg["resume"])
        overrides = remaining
    for item in overrides:
        if "=" not in item:
            raise ValueError("Override requires path=value")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in obj or not isinstance(obj[part], dict):
                raise ValueError("Unknown config path: "+key)
            obj = obj[part]
        if parts[-1] not in obj:
            raise ValueError("Unknown config path: "+key)
        obj[parts[-1]] = value
    if cfg["wall_time_seconds"] is not None:
        cfg["equal_time_seconds"] = cfg["wall_time_seconds"]
    if cfg["train_sizes"] is None:
        cfg["train_sizes"] = sorted(set(round(cfg["n_start"]*cfg["n_multiplier"]**i)
                                       for i in range(cfg["n_levels"])))
    validate(cfg)
    return cfg


def validate(cfg):
    if not cfg["methods"] or not set(cfg["methods"]) <= set(METHODS):
        raise ValueError("Only soi, line_mine, line_nwj are supported")
    for key in ["num_blocks", "u_dim", "n_val", "n_test", "eval_batch_size", "cpu_threads"]:
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(key+" must be a positive integer")
    if cfg["num_blocks"] < 3 or cfg["v_dim"] < 0 or not isinstance(cfg["v_dim"], int):
        raise ValueError("Need N>=3 and v_dim>=0")
    for key in ["train_sizes", "k_values", "distribution_seeds", "data_seeds", "training_seeds"]:
        if not cfg[key] or any(not isinstance(v, int) for v in cfg[key]) or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(key+" must contain distinct integers")
    if min(cfg["train_sizes"]) < 2 or min(cfg["k_values"]) < 1:
        raise ValueError("Need train_sizes>=2 and k_values>=1")
    cfg["train_sizes"] = sorted(cfg["train_sizes"])
    if cfg["normalization"] not in ["population", "none"] or cfg["transformation"] != "":
        raise ValueError("Supported normalization: population/none; transformation must be empty")
    if cfg["precision"] != "float32":
        raise ValueError("Networks use float32; sensitive accumulation always uses float64")
    if cfg["budget_mode"] not in ["paper", "equal_time"]:
        raise ValueError("Unknown budget_mode")
    if not cfg["budget_multipliers"] or any(b <= 0 for b in cfg["budget_multipliers"]):
        raise ValueError("Budget multipliers must be positive")
    for key in ["checkpoints_every", "evaluation_every", "diagnostic_samples"]:
        if not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(key+" must be a nonnegative integer")
    if cfg["equal_time_seconds"] <= 0 or (cfg["max_run_seconds"] is not None and cfg["max_run_seconds"] <= 0):
        raise ValueError("Time budgets must be positive")
    if cfg["epsilon"] is not None and cfg["epsilon"] < 0:
        raise ValueError("epsilon must be nonnegative")
    for method in cfg["methods"]:
        p = cfg["method_params"][method]
        if p["optimizer"]["name"] not in ["Adam", "AdamW"]:
            raise ValueError("Supported optimizer: Adam/AdamW")
        if not 1 <= p["batch_size"] <= min(cfg["train_sizes"]):
            raise ValueError("Batch size must fit every training prefix")
        if p["training_steps"] != "paper" and (not isinstance(p["training_steps"], int) or p["training_steps"] < 1):
            raise ValueError("training_steps must be a positive integer or paper")
        if p["optimizer"]["lr"] <= 0 or p["optimizer"]["eps"] <= 0:
            raise ValueError("Optimizer lr and eps must be positive")
        if method == "soi":
            if p["dim_mults"] and (any(m != 1 for m in p["dim_mults"])):
                raise ValueError("Upstream MLP only supports constant width dim_mults")
            if p["mc_iter"] < 1 or not 0 <= p["ema_decay"] < 1:
                raise ValueError("Invalid SOI MC/EMA parameters")
            if p["ema_timing"] not in ["before_step", "after_step"]:
                raise ValueError("Unknown EMA timing")
            if p["sde"]["inference_weighting"] not in ["upstream_constant", "exact"]:
                raise ValueError("Unknown importance weighting")
        else:
            if p["pair_chunk"] < 0 or p["negative_sampling"] not in ["with_replacement", "permutation"]:
                raise ValueError("Invalid baseline negative sampling/chunk")
            if p["eval_aggregation"] not in ["paper_batch_mean", "global_moment"]:
                raise ValueError("Unknown evaluation aggregation")
    covariance(cfg)
    values = cfg.get("rho_values")
    if values is not None:
        if not values or len(set(values)) != len(values):
            raise ValueError("rho_values must contain distinct correlation values")
        for rho in values:
            covariance(case_config(cfg, {"rho": rho}))
    convergence = cfg.get("convergence")
    if convergence:
        for key in ["every", "min_steps", "patience", "n_samples"]:
            if type(convergence[key]) is not int or convergence[key] < 1:
                raise ValueError("Invalid convergence parameter: "+key)
        if not 0 < convergence["smoothing"] <= 1 or convergence["relative_tolerance"] < 0:
            raise ValueError("Invalid convergence smoothing/tolerance")


def method_steps(cfg, method, multiplier=1.):
    steps = cfg["method_params"][method]["training_steps"]
    if steps == "paper":
        n = cfg["num_blocks"]
        steps = 195000 if method == "soi" else (40000 if n == 6 else 80000 if n == 10 else 8000*n)
    return math.ceil(steps*multiplier)


def grid(cfg):
    # Control changes dimension and is labeled separately.
    conditions = [(k, cfg["v_dim"], "main") for k in cfg["k_values"]]
    if cfg["control_v_dim_zero"] and cfg["v_dim"] != 0:
        conditions += [(1, 0, "gaussian_control")]
    import itertools
    for (k, vd, condition), n, ds, xs, ts, mult in itertools.product(
        conditions, cfg["train_sizes"], cfg["distribution_seeds"], cfg["data_seeds"],
        cfg["training_seeds"], cfg["budget_multipliers"]):
        job = dict(k=k, v_dim=vd, condition=condition, n_train=n, distribution_seed=ds,
                   data_seed=xs, training_seed=ts, budget_multiplier=mult)
        if cfg.get("rho_values") is None:
            for method in cfg["methods"]:
                yield dict(job, method=method)
        else:
            for index, rho in enumerate(cfg["rho_values"]):
                for method in cfg["methods"]:
                    yield dict(job, method=method, rho=rho, point_index=index)
