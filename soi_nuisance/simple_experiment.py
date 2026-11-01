"""Five user settings; the authors' Gaussian benchmark and optimizers are fixed."""
import copy
from pathlib import Path

DEFAULT_SETTINGS = dict(u_dim=1, v_dim=1, n_samples=10000, k=16, max_steps=20000)
# Ten entries from the official jobs/o_inf.sh sweep; synergy stays at rho=.7.
RHO_VALUES = [.10, .20, .25, .30, .35, .375, .40, .45, .50, .60]


def settings_from_config(value):
    if value.get("simple_settings"):
        return copy.deepcopy(value["simple_settings"])
    if "n_samples" in value:
        return {key: value.get(key, default) for key, default in DEFAULT_SETTINGS.items()}
    sizes, knots = value.get("train_sizes"), value.get("k_values")
    steps = value.get("method_params", {}).get("soi", {}).get("training_steps")
    return dict(u_dim=value.get("u_dim", 1), v_dim=value.get("v_dim", 1),
                n_samples=sizes[0] if sizes else 10000, k=knots[0] if knots else 16,
                max_steps=steps if isinstance(steps, int) else 20000)


def build_experiment(settings, output_dir="results/experiment", resume=False):
    from .config import ROOT, read_config, validate
    settings = {key: settings.get(key, default) for key, default in DEFAULT_SETTINGS.items()}
    for key, value in settings.items():
        minimum = 0 if key == "v_dim" else 256 if key == "n_samples" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key}: требуется целое число не меньше {minimum}.")
    config = read_config(ROOT/"configs/base.json")
    config.update(u_dim=settings["u_dim"], v_dim=settings["v_dim"], train_sizes=[settings["n_samples"]],
                  k_values=[settings["k"]], num_blocks=10, benchmark="mixed", rho=RHO_VALUES[0],
                  rho_values=list(RHO_VALUES), rho_subsystem_indices=[0, 1],
                  mixed_subsystems=[dict(benchmark="redundancy", size=3, rho=RHO_VALUES[0]),
                                    dict(benchmark="redundancy", size=4, rho=RHO_VALUES[0]),
                                    dict(benchmark="synergy", size=3, rho=.7)],
                  distribution_seeds=[77], data_seeds=[77], training_seeds=[77],
                  control_v_dim_zero=False, budget_mode="paper", budget_multipliers=[1.],
                  n_val=2048, n_test=4096, eval_batch_size=256,
                  checkpoints_every=500, evaluation_every=0, log_every=500,
                  diagnostic_samples=0, output_dir=str(output_dir), resume=resume,
                  report_mode="curve", simple_settings=settings,
                  convergence=dict(every=500, min_steps=5000, patience=8, relative_tolerance=.001,
                                   smoothing=.5, n_samples=1024))
    for params in config["method_params"].values():
        params["training_steps"] = settings["max_steps"]
    config["method_params"]["soi"]["mc_iter"] = 10
    validate(config)
    return config
