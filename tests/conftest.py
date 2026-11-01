import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal
from numpy.random import default_rng
from soi_nuisance.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def module_file(relative):
    path = ROOT/relative
    spec = importlib.util.spec_from_file_location("reference_"+path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extracted(relative, names=None, extra=None):
    source = (ROOT/relative).read_text(encoding="utf-8-sig")
    parsed = ast.parse(source)
    nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and (names is None or node.name in names)]
    scope = dict(np=np, torch=torch, default_rng=default_rng, multivariate_normal=multivariate_normal,
                 Dataset=torch.utils.data.Dataset)
    scope.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/relative), "exec"), scope)
    return scope


@pytest.fixture
def tiny():
    torch.set_num_threads(1)
    cfg = load_config("smoke")
    cfg.update(train_sizes=[64], n_val=32, n_test=33, k_values=[1], v_dim=0,
               control_v_dim_zero=False, checkpoints_every=2, evaluation_every=2,
               diagnostic_samples=0, log_every=0, eval_batch_size=16, device="cpu")
    for params in cfg["method_params"].values():
        params.update(training_steps=4, batch_size=32)
    cfg["method_params"]["soi"].update(width=16, time_dim=16, mc_iter=1)
    return cfg
