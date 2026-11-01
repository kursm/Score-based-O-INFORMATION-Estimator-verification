"""Progress and approximate ETA. The UI imports this without importing torch."""
import json
from pathlib import Path
import time


def format_duration(seconds):
    if seconds is None:
        return "уточняется по ходу работы"
    seconds = max(0, int(seconds+.5))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days} д {hours} ч {minutes} мин"
    if hours:
        return f"{hours} ч {minutes} мин"
    if minutes:
        return f"{minutes} мин {seconds} с"
    return f"{seconds} с"


def build_tasks(config, plan, actual_device, root):
    """Read existing timings, preserving source results. No training is performed."""
    from .config import grid, method_steps
    from .runner import immutable_spec, signature_for
    root = Path(root)
    references = []
    for directory in [root/"results/ten_point_check", root/"results/timing_n6", root/"results/smoke"]:
        env_path = directory/"environment.json"
        if not env_path.exists():
            continue
        env = json.loads(env_path.read_text(encoding="utf-8"))
        if env["device"].split(":")[0] != actual_device.split(":")[0]:
            continue
        for path in sorted((directory/"runs").glob("*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["status"] != "ok" or not result["training_steps"]:
                continue
            resolved = json.loads((path.parent/"resolved_config.json").read_text(encoding="utf-8"))
            references.append((result, resolved["config"]))
    parameters = {(m["method"], m["v_dim"]): m["parameters"] for m in plan["models"]}
    tasks = []
    for job in grid(config):
        method = job["method"]
        params = config["method_params"][method]
        target_steps = method_steps(config, method, job["budget_multiplier"])
        units = (target_steps if config["budget_mode"] == "paper" else
                 config["equal_time_seconds"]*job["budget_multiplier"])
        run_id = method+"_"+signature_for(immutable_spec(config, job))[:16]
        previous_path = root/config["output_dir"]/"runs"/run_id/"result.json"
        initial, skipped = 0., False
        if config["resume"] and previous_path.exists():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            initial = min(units, previous["training_steps"] if config["budget_mode"] == "paper"
                          else previous["training_seconds"])
            skipped = (previous["status"] not in ["ok", "interrupted"] or
                       (previous["status"] == "ok" and (initial >= units or previous.get("stop_reason") == "converged")))
        candidates = [(r, c) for r, c in references if r["method"] == method]
        per_step, overhead = None, 0.
        if candidates:
            dimension = config["num_blocks"]*(config["u_dim"]+job["v_dim"])
            ref, old = min(candidates, key=lambda pair: abs(pair[0]["dimension"]-dimension))
            scale = parameters[(method, job["v_dim"])]/ref["parameter_count"]
            batch_ratio = params["batch_size"]/old["method_params"][method]["batch_size"]
            per_step = ref["training_seconds"]/ref["training_steps"]*scale*batch_ratio**(
                2 if method == "line_nwj" else 1)
            mc_ratio = (params["mc_iter"]/old["method_params"][method]["mc_iter"]
                        if method == "soi" else 1.)
            evaluation = ref["final_evaluation_seconds"]*scale*config["n_test"]/old["n_test"]*mc_ratio
            estimated_steps = target_steps if config["budget_mode"] == "paper" else units/per_step
            validations = int(estimated_steps/config["evaluation_every"]) if config["evaluation_every"] else 0
            overhead = evaluation*(1+validations*config["n_val"]/config["n_test"])
            if method == "soi" and config["diagnostic_samples"]:
                overhead += evaluation*5*min(config["diagnostic_samples"], config["n_val"])/(
                    config["n_test"]*params["mc_iter"])
        truth = next((p["omega_true"] for p in plan.get("curve_points", []) if p["rho"] == job.get("rho")), None)
        tasks.append(dict(job=job, run_id=run_id, units=units, initial=initial, skipped=skipped, omega_true=truth,
                          per_unit_seconds=per_step if config["budget_mode"] == "paper" else 1.,
                          overhead_seconds=overhead, budget_mode=config["budget_mode"]))
    return tasks


class ProgressState:
    def __init__(self, tasks, clock=time.monotonic):
        self.tasks, self.clock = tasks, clock
        self.completed, self.failed, self.rates = set(), set(), {}
        self.current = None
        self.unit = self.training_seconds = self.initial_unit = self.initial_training_seconds = 0.
        self.job_started_at = clock()
        self.phase = "preparing"

    @staticmethod
    def key(task):
        return task["job"]["method"], task["job"]["v_dim"]

    def update(self, event):
        name = event["event"]
        if name == "job_started":
            self.current = event["job_index"]
            self.unit = self.tasks[self.current]["initial"]
            self.training_seconds = self.initial_training_seconds = 0.
            self.initial_unit = self.unit
            self.job_started_at = self.clock()
        if "job_index" in event:
            self.current = event["job_index"]
        if self.current is not None:
            task = self.tasks[self.current]
            if "training_seconds" in event:
                self.training_seconds = event["training_seconds"]
                self.unit = event.get("step", self.unit) if task["budget_mode"] == "paper" else self.training_seconds
            if name == "training_started":
                self.initial_unit, self.initial_training_seconds = self.unit, self.training_seconds
            if name == "progress":
                delta_unit = self.unit-self.initial_unit
                delta_time = self.training_seconds-self.initial_training_seconds
                if delta_unit > 0 and delta_time > 0:
                    self.rates[self.key(task)] = delta_time/delta_unit
            if name == "job_finished" and event["status"] != "interrupted":
                self.completed.add(self.current)
                if event["status"] != "ok":
                    self.failed.add(self.current)
        self.phase = name

    def remaining(self):
        seconds = 0.
        for index, task in enumerate(self.tasks):
            if index in self.completed or task["skipped"]:
                continue
            current = index == self.current
            initial = self.unit if current else task["initial"]
            remaining_units = max(0., task["units"]-initial)
            rate = self.rates.get(self.key(task), task["per_unit_seconds"])
            if rate is None and remaining_units > 0:
                return None
            overhead = task["overhead_seconds"]
            if current:
                overhead_spent = max(0., self.clock()-self.job_started_at-
                                     (self.training_seconds-self.initial_training_seconds))
                overhead = max(0., overhead-overhead_spent)
            seconds += remaining_units*(rate or 0.)+overhead
        if self.phase in ["evaluation", "diagnostics", "report"] and seconds < .5:
            return None
        return seconds

    def snapshot(self):
        total = len(self.tasks)
        fraction = 0.
        if self.current is not None:
            fraction = min(1., max(0., self.unit/self.tasks[self.current]["units"]))
            if self.current in self.completed:
                fraction = 1.
        return dict(completed=len(self.completed), total=total, failed=len(self.failed),
                    overall_fraction=len(self.completed)/max(1, total),
                    current_fraction=fraction, remaining_seconds=self.remaining())
