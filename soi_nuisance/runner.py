"""Finite-data runner. No model selection on truth or test. Checkpoints are trusted local files."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
import traceback
import numpy as np
import torch
from .config import grid, method_steps
from .data import seed_for, get_dataset, case_config, covariance, gaussian_truth
from .estimators import make_estimator


class DivergenceError(RuntimeError):
    pass


class StopRequested(KeyboardInterrupt):
    """Cooperative interruption at an optimizer boundary, safe to checkpoint."""


def emit_progress(callback, event, **values):
    if callback is not None:
        callback(dict(event=event, **values))


def json_clean(value):
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(json_clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def configure_environment(cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(cfg["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use device=cpu")
    return torch.device(device)


def environment_info(device):
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                threads=torch.get_num_threads())


class BatchStream:
    """Shuffled epochs with drop_last, as upstream; stores exact traversal and RNG."""
    def __init__(self, n, batch_size, seed):
        self.n, self.batch_size = n, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(n, generator=self.generator)
        self.offset = 0
        self.seen = torch.zeros(n, dtype=torch.bool)

    def next(self):
        if self.offset+self.batch_size > self.n:
            self.order = torch.randperm(self.n, generator=self.generator)
            self.offset = 0
        ids = self.order[self.offset:self.offset+self.batch_size]
        self.offset += self.batch_size
        self.seen[ids] = True
        return ids

    def state_dict(self):
        return dict(order=self.order, offset=self.offset, seen=self.seen,
                    rng=self.generator.get_state(), n=self.n, batch_size=self.batch_size)

    def load_state_dict(self, state):
        if state["n"] != self.n or state["batch_size"] != self.batch_size:
            raise ValueError("Incompatible batch stream")
        self.order, self.offset, self.seen = state["order"], state["offset"], state["seen"]
        self.generator.set_state(state["rng"])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, stream, generator, state, signature):
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   stream=stream.state_dict(), train_rng=generator.get_state(),
                   global_rng=rng_state(), state=state, signature=signature, format_version=1)
    tmp = Path(str(path)+".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer, stream, generator, signature):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["signature"] != signature:
        raise ValueError("Resume signature mismatch")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    stream.load_state_dict(ckpt["stream"])
    generator.set_state(ckpt["train_rng"])
    restore_rng(ckpt["global_rng"])
    return ckpt["state"]


def replica_rng_seed(job, channel, step=None, method_specific=True):
    # Independent outer/nested replicas; paired streams across k, n and budgets.
    return seed_for(channel, job["distribution_seed"], job["data_seed"], job["training_seed"],
                    job["method"] if method_specific else "shared_batch_order", step)


def model_for(cfg, job, device):
    method = job["method"]
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(replica_rng_seed(job, "initialization"))
        model = make_estimator(method, cfg["num_blocks"], cfg["u_dim"]+job["v_dim"],
                               cfg["method_params"][method]).to(device)
    return model


def immutable_spec(cfg, job):
    cfg = case_config(cfg, job)
    params = copy.deepcopy(cfg["method_params"][job["method"]])
    params.pop("training_steps")
    spec = dict(job=job, params=params, benchmark=cfg["benchmark"], rho=cfg["rho"],
                mixed_subsystems=cfg["mixed_subsystems"], num_blocks=cfg["num_blocks"],
                u_dim=cfg["u_dim"], normalization=cfg["normalization"], precision=cfg["precision"],
                deterministic=cfg["deterministic"], eval_batch_size=cfg["eval_batch_size"],
                n_val=cfg["n_val"], n_test=cfg["n_test"], n_pool=max(cfg["train_sizes"]),
                budget_mode=cfg["budget_mode"], generator_version=1, training_rng_version=2)
    if cfg.get("convergence"):
        spec["convergence"] = cfg["convergence"]
    return spec


def signature_for(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def run_one(cfg, job, output, device, resume=False, retry_failed=False,
            progress_callback=None, stop_requested=None):
    cfg = case_config(cfg, job)
    spec = immutable_spec(cfg, job)
    signature = signature_for(spec)
    run_id = job["method"]+"_"+signature[:16]
    directory = Path(output)/"runs"/run_id
    directory.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint_path = directory/"result.json", directory/"last.pt"
    previous = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    target = method_steps(cfg, job["method"], job["budget_multiplier"])
    seconds_budget = cfg["equal_time_seconds"]*job["budget_multiplier"]
    emit_progress(progress_callback, "job_started", run_id=run_id, job=job,
                  target_steps=target, target_seconds=seconds_budget, budget_mode=cfg["budget_mode"])
    if previous:
        if not resume:
            raise FileExistsError("Existing run; use --resume or a new output_dir: "+run_id)
        if previous["status"] not in ["ok", "interrupted"] and not retry_failed:
            print("Preserved failed run", run_id, previous["status"], flush=True)
            emit_progress(progress_callback, "job_finished", status=previous["status"], skipped=True)
            return previous
        completed = (previous["training_steps"] >= target or previous.get("stop_reason") == "converged" if cfg["budget_mode"] == "paper"
                     else previous["training_seconds"] >= seconds_budget)
        if previous["status"] == "ok" and completed:
            print("Already complete", run_id, flush=True)
            emit_progress(progress_callback, "job_finished", status="ok", skipped=True)
            return previous
    data_begin = time.perf_counter()
    arrays, metadata, data_dir, reused = get_dataset(
        cfg, output, job["k"], job["distribution_seed"], job["data_seed"], job["v_dim"])
    data_access_seconds = time.perf_counter()-data_begin
    model = model_for(cfg, job, device)
    params = cfg["method_params"][job["method"]]
    optimizer_config = params["optimizer"].copy()
    optimizer_name = optimizer_config.pop("name")
    optimizer = getattr(torch.optim, optimizer_name)(model.trainable_parameters(), **optimizer_config)
    stream = BatchStream(job["n_train"], params["batch_size"], replica_rng_seed(job, "batch", method_specific=False))
    generator = torch.Generator(device=device).manual_seed(replica_rng_seed(job, "noise"))
    state = dict(step=0, training_seconds=0., diagnostics_seconds=0., checkpoint_seconds=0.,
                 x_presentations=0, last_loss=None)
    if resume and checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path, model, optimizer, stream, generator, signature)
    elif resume and previous and not (retry_failed and previous["status"] != "ok"
                                      and previous["training_steps"] == 0):
        raise FileNotFoundError("Cannot extend run without last.pt; enable save_checkpoints")
    elif not resume and checkpoint_path.exists():
        raise FileExistsError("Checkpoint exists; use --resume")
    monitor = None
    if cfg.get("convergence"):
        from .convergence import PlateauMonitor
        monitor = PlateauMonitor(cfg["convergence"], state.get("convergence"))
        state["convergence"] = monitor.state
    stop_reason = "step_limit" if cfg["budget_mode"] == "paper" else "time_budget"
    write_json(directory/"resolved_config.json", dict(config=cfg, job=job, immutable_signature=signature,
                                                     architecture=model.architecture))
    # Remove trajectory records beyond the restored checkpoint, without changing earlier records.
    history_path = directory/"history.jsonl"
    if resume:
        for log_path in [history_path, directory/"convergence.jsonl"]:
            if log_path.exists():
                retained = [line for line in log_path.read_text(encoding="utf-8").splitlines()
                            if json.loads(line)["step"] <= state["step"]]
                log_path.write_text("\n".join(retained)+("\n" if retained else ""), encoding="utf-8")
    train = torch.from_numpy(arrays["train"][:job["n_train"]])
    val, test = torch.from_numpy(arrays["val"]), torch.from_numpy(arrays["test"])
    status, failure = "ok", None
    evaluation, final_seconds, regression = {}, 0., None
    emit_progress(progress_callback, "training_started", step=state["step"],
                  training_seconds=state["training_seconds"])
    last_progress = 0.
    # Test is never consulted during the loop. Intermediate estimates use validation only.
    try:
        while True:
            if stop_requested is not None and stop_requested():
                raise StopRequested()
            if monitor is not None and monitor.state["converged"]:
                stop_reason = "converged"
                break
            if cfg["budget_mode"] == "paper" and state["step"] >= target:
                break
            if cfg["budget_mode"] == "equal_time" and state["training_seconds"] >= seconds_budget:
                break
            if cfg["max_run_seconds"] is not None and state["training_seconds"] >= cfg["max_run_seconds"]:
                raise TimeoutError("Training watchdog reached")
            sync(device)
            begin = time.perf_counter()
            model.train()
            ids = stream.next()
            batch = train[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch, generator)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            if cfg["divergence_threshold"] is not None and loss.abs().item() > cfg["divergence_threshold"]:
                raise DivergenceError("Configured absolute loss threshold exceeded")
            if job["method"] == "soi" and params["ema_timing"] == "before_step":
                model.update_ema()
            loss.backward()
            gradient_checks = [torch.isfinite(p.grad).all() for p in model.trainable_parameters()
                               if p.grad is not None]
            if gradient_checks and not torch.stack(gradient_checks).all():
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
            if job["method"] != "soi" or params["ema_timing"] == "after_step":
                model.update_ema()
            if not torch.stack([torch.isfinite(p).all() for p in model.trainable_parameters()]).all():
                raise FloatingPointError("Nonfinite parameters")
            sync(device)
            state["training_seconds"] += time.perf_counter()-begin
            state["step"] += 1
            state["x_presentations"] += len(batch)
            state["last_loss"] = loss.item()
            step = state["step"]
            if progress_callback is not None and (
                    time.monotonic()-last_progress >= .25 or step == target):
                emit_progress(progress_callback, "progress", step=step, loss=state["last_loss"],
                              training_seconds=state["training_seconds"])
                last_progress = time.monotonic()
            if cfg["log_every"] and step % cfg["log_every"] == 0:
                print(run_id, "step", step, "loss", round(loss.item(), 5),
                      "train_s", round(state["training_seconds"], 2), flush=True)
            if cfg["evaluation_every"] and step % cfg["evaluation_every"] == 0:
                emit_progress(progress_callback, "validation", step=step)
                sync(device)
                begin = time.perf_counter()
                ev_gen = torch.Generator(device=device).manual_seed(replica_rng_seed(job, "validation", step))
                measures = model.evaluate(val, cfg["eval_batch_size"], ev_gen, u_dim=cfg["u_dim"])
                sync(device)
                state["diagnostics_seconds"] += time.perf_counter()-begin
                append_jsonl(history_path, dict(step=step, training_seconds=state["training_seconds"],
                                               loss=state["last_loss"], split="validation", **measures))
                emit_progress(progress_callback, "training", step=step)
            if monitor is not None and step % cfg["convergence"]["every"] == 0:
                from .convergence import validation_objective
                emit_progress(progress_callback, "convergence_check", step=step)
                sync(device)
                begin = time.perf_counter()
                objective = validation_objective(model, val, params["batch_size"],
                    replica_rng_seed(job, "convergence_validation"), cfg["convergence"]["n_samples"])
                monitor.update(step, objective)
                sync(device)
                state["diagnostics_seconds"] += time.perf_counter()-begin
                append_jsonl(directory/"convergence.jsonl", dict(step=step, validation_loss=objective,
                             **monitor.state))
                emit_progress(progress_callback, "convergence", step=step, validation_loss=objective,
                              converged=monitor.state["converged"], stale=monitor.state["stale"])
            if cfg["save_checkpoints"] and cfg["checkpoints_every"] and step % cfg["checkpoints_every"] == 0:
                begin = time.perf_counter()
                save_checkpoint(checkpoint_path, model, optimizer, stream, generator, state.copy(), signature)
                state["checkpoint_seconds"] += time.perf_counter()-begin
        if cfg["save_checkpoints"]:
            begin = time.perf_counter()
            save_checkpoint(checkpoint_path, model, optimizer, stream, generator, state.copy(), signature)
            state["checkpoint_seconds"] += time.perf_counter()-begin
        if stop_requested is not None and stop_requested():
            raise StopRequested()
        emit_progress(progress_callback, "evaluation", step=state["step"],
                      training_seconds=state["training_seconds"])
        sync(device)
        begin = time.perf_counter()
        ev_gen = torch.Generator(device=device).manual_seed(replica_rng_seed(job, "final_test"))
        evaluation = model.evaluate(test, cfg["eval_batch_size"], ev_gen, u_dim=cfg["u_dim"])
        sync(device)
        final_seconds = time.perf_counter()-begin
        if not all(np.isfinite(evaluation[k]) for k in ["tc_hat", "dtc_hat", "omega_hat"]):
            raise FloatingPointError("Nonfinite final estimate")
        if job["method"] == "soi" and cfg["diagnostic_samples"] > 0:
            emit_progress(progress_callback, "diagnostics", step=state["step"])
            begin = time.perf_counter()
            dg = torch.Generator(device=device).manual_seed(replica_rng_seed(job, "regression"))
            regression = model.regression_diagnostics(
                val[:cfg["diagnostic_samples"]], cfg["eval_batch_size"], dg, cfg["u_dim"])
            sync(device)
            state["diagnostics_seconds"] += time.perf_counter()-begin
            write_json(directory/"regression_diagnostics.json", regression)
    except StopRequested:
        stop_reason = "user_stop"
        status, failure = "interrupted", "Stopped at an optimizer boundary"
        if cfg["save_checkpoints"]:
            save_checkpoint(checkpoint_path, model, optimizer, stream, generator, state.copy(), signature)
    except KeyboardInterrupt:
        stop_reason = "user_stop"
        status, failure = "interrupted", "Interrupted by user"
    except (FloatingPointError, TimeoutError, RuntimeError, MemoryError) as exc:
        status = ("nan" if isinstance(exc, FloatingPointError) else "divergence" if isinstance(exc, DivergenceError)
                  else "timeout" if isinstance(exc, TimeoutError)
                  else "oom" if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower() else "error")
        failure = "".join(traceback.format_exception(exc))
    if status != "ok":
        # Preserve last good checkpoint; diagnostic snapshot also records RNG after attempted update.
        try:
            save_checkpoint(directory/"diagnostic.pt", model, optimizer, stream, generator,
                            state.copy(), signature)
        except Exception as exc:
            failure += "\nDiagnostic checkpoint error: "+str(exc)
    evaluation = {key: value for key, value in evaluation.items()
                  if not isinstance(value, float) or np.isfinite(value)}
    evaluation.setdefault("evaluation_mode", "vp_"+params["sde"]["inference_weighting"]
                          if job["method"] == "soi" else params["eval_aggregation"])
    result = dict(run_id=run_id, **{k:v for k,v in job.items() if k != "rho"}, num_blocks=cfg["num_blocks"], u_dim=cfg["u_dim"],
                  benchmark=cfg["benchmark"], rho=cfg["rho"], normalization=cfg["normalization"],
                  dimension=cfg["num_blocks"]*(cfg["u_dim"]+job["v_dim"]),
                  budget_mode=cfg["budget_mode"], target_steps=target,
                  target_seconds=seconds_budget if cfg["budget_mode"] == "equal_time" else None,
                  status=status, stop_reason=stop_reason if status == "ok" or status == "interrupted" else status,
                  failure=failure, **metadata["truth"], **evaluation,
                  training_steps=state["step"], x_presentations=state["x_presentations"],
                  unique_train_available=job["n_train"], unique_train_seen=int(stream.seen.sum()),
                  critic_x_presentations=state["x_presentations"]*(2*cfg["num_blocks"]-1)
                       if job["method"] != "soi" else None,
                  new_diffusion_noise_draws=state["x_presentations"] if job["method"] == "soi" else 0,
                  training_seconds=state["training_seconds"], diagnostics_seconds=state["diagnostics_seconds"],
                  checkpoint_seconds=state["checkpoint_seconds"], final_evaluation_seconds=final_seconds,
                  data_generation_seconds=metadata["generation_seconds"], dataset_reused=reused,
                  data_access_seconds=data_access_seconds,
                  parameter_count=sum(p.numel() for p in model.trainable_parameters()),
                  architecture=model.architecture, data_directory=str(data_dir),
                  data_hashes=metadata["hashes"], final_loss=state["last_loss"])
    if status == "ok":
        error = result["omega_hat"]-result["omega_true"]
        result.update(signed_error=error, absolute_error=abs(error), squared_error=error*error)
    write_json(result_path, result)
    print(run_id, status, "Omega", evaluation.get("omega_hat"), "steps", state["step"], flush=True)
    emit_progress(progress_callback, "job_finished", status=status, step=state["step"],
                  training_seconds=state["training_seconds"], skipped=False, stop_reason=result["stop_reason"])
    if status == "interrupted":
        raise KeyboardInterrupt
    return result


def collect_results(output):
    results = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((Path(output)/"runs").glob("*/result.json"))]
    with (Path(output)/"results.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
    if results:
        fields = sorted(set().union(*(r.keys() for r in results)))
        with (Path(output)/"results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            for result in results:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                                 else value for key, value in result.items()})
    return results


def run(cfg, resume=False, retry_failed=False, progress_callback=None, stop_requested=None):
    device = configure_environment(cfg)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    old_cfg = output/"resolved_config.json"
    if old_cfg.exists():
        if not resume:
            raise FileExistsError("Output already exists; use --resume or change output_dir")
        previous_cfg = json.loads(old_cfg.read_text(encoding="utf-8"))
        checked = ["benchmark", "rho", "mixed_subsystems", "num_blocks", "u_dim", "v_dim",
                   "normalization", "transformation", "precision", "deterministic",
                   "eval_batch_size", "n_val", "n_test", "device", "budget_mode", "rho_values",
                   "rho_subsystem_indices", "convergence"]
        if any(previous_cfg.get(k, [] if k == "rho_subsystem_indices" else None) !=
               cfg.get(k, [] if k == "rho_subsystem_indices" else None) for k in checked) or (
                max(previous_cfg["train_sizes"]) != max(cfg["train_sizes"])):
            raise ValueError("Resume changes immutable data/evaluation settings; use a new output_dir")
        for method in set(cfg["methods"]) & set(previous_cfg["methods"]):
            before = {k: v for k, v in previous_cfg["method_params"][method].items() if k != "training_steps"}
            after = {k: v for k, v in cfg["method_params"][method].items() if k != "training_steps"}
            if before != after:
                raise ValueError("Resume changes method settings; use a new output_dir")
        # Preserve initial config and every subsequent invocation.
        write_json(output/"invocations"/(str(time.time_ns())+".json"), cfg)
    else:
        write_json(old_cfg, cfg)
    write_json(output/"environment.json", environment_info(device))
    try:
        jobs = list(grid(cfg))
        emit_progress(progress_callback, "experiment_started", total_jobs=len(jobs), device=str(device))
        for index, job in enumerate(jobs):
            if stop_requested is not None and stop_requested():
                raise StopRequested()
            def job_progress(event):
                if progress_callback is not None:
                    progress_callback(dict(event, job_index=index, total_jobs=len(jobs)))
            try:
                run_one(cfg, job, output, device, resume, retry_failed,
                        job_progress if progress_callback is not None else None, stop_requested)
            except (RuntimeError, MemoryError) as exc:
                resolved = case_config(cfg, job)
                signature = signature_for(immutable_spec(cfg, job))
                run_id = job["method"]+"_"+signature[:16]
                params = cfg["method_params"][job["method"]]
                failure_result = dict(
                    run_id=run_id, **{k:v for k,v in job.items() if k != "rho"}, num_blocks=cfg["num_blocks"], u_dim=cfg["u_dim"],
                    benchmark=cfg["benchmark"], rho=resolved["rho"], normalization=cfg["normalization"],
                    dimension=cfg["num_blocks"]*(cfg["u_dim"]+job["v_dim"]),
                    budget_mode=cfg["budget_mode"],
                    target_steps=method_steps(cfg, job["method"], job["budget_multiplier"]),
                    target_seconds=cfg["equal_time_seconds"]*job["budget_multiplier"]
                        if cfg["budget_mode"] == "equal_time" else None,
                    status="oom" if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower() else "error",
                    failure="".join(traceback.format_exception(exc)), phase="setup",
                    training_steps=0, training_seconds=0., final_evaluation_seconds=0.,
                    evaluation_mode="vp_"+params["sde"]["inference_weighting"]
                        if job["method"] == "soi" else params["eval_aggregation"],
                    **gaussian_truth(covariance(resolved), cfg["num_blocks"], cfg["u_dim"]))
                write_json(output/"runs"/run_id/"result.json", failure_result)
                job_progress(dict(event="job_finished", status=failure_result["status"], skipped=False))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        results = collect_results(output)
    return results


def plan(cfg):
    jobs = list(grid(cfg))
    device = torch.device("cpu")
    models = {}
    for job in jobs:
        key = (job["method"], job["v_dim"])
        if key not in models:
            model = model_for(cfg, job, device)
            models[key] = dict(method=key[0], v_dim=key[1], architecture=model.architecture,
                               parameters=sum(p.numel() for p in model.trainable_parameters()))
    datasets = {(j["k"], j["v_dim"], j["distribution_seed"], j["data_seed"], j.get("rho")) for j in jobs}
    data_bytes = sum((max(cfg["train_sizes"])+cfg["n_val"]+cfg["n_test"])*cfg["num_blocks"]*
                     (cfg["u_dim"]+vd)*4 for _, vd, _, _, _ in datasets)
    curve_points = [dict(rho=rho, **gaussian_truth(covariance(case_config(cfg, {"rho":rho})),
                    cfg["num_blocks"], cfg["u_dim"])) for rho in cfg.get("rho_values") or []]
    return dict(runs=len(jobs), datasets=len(datasets), uncompressed_data_bytes=data_bytes,
                curve_points=curve_points,
                models=list(models.values()),
                total_target_updates=sum(method_steps(cfg, j["method"], j["budget_multiplier"]) for j in jobs)
                     if cfg["budget_mode"] == "paper" else None,
                total_training_seconds=len(jobs)*cfg["equal_time_seconds"]*
                     sum(cfg["budget_multipliers"])/len(cfg["budget_multipliers"])
                     if cfg["budget_mode"] == "equal_time" else None,
                evaluation_batch_size=cfg["eval_batch_size"], resolved_config=cfg,
                fixed_data="same U across k; fixed pools; nested n prefixes",
                baseline_other_n_rule="N=6:40k, N=10:80k, otherwise 8000*N, explicit extrapolation",
                time_policy="Training includes data transfer, all critics, backward, finite checks and synchronization; excludes validation, diagnostics, checkpoint I/O and final evaluation.")
