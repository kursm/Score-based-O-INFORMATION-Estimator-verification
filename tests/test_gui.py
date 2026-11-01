from copy import deepcopy
import json
from pathlib import Path
import time

import pytest
import torch

from soi_nuisance.config import ROOT, grid, load_config
from soi_nuisance.gui_fields import field_help, iter_fields, parse_value, read_raw_config
from soi_nuisance.gui_progress import ProgressState, build_tasks
from soi_nuisance.runner import configure_environment, plan, run_one


@pytest.fixture(scope="module")
def tk_session():
    # One Tcl/Tk interpreter per process; independent test windows use Toplevel.
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip("Tk desktop unavailable: "+str(exc))
    root.withdraw()
    yield root
    root.destroy()


def test_every_exposed_parameter_has_specific_help():
    raw = read_raw_config(ROOT/"configs/base.json")
    for path, value in iter_fields(raw):
        label, description = field_help(path)
        assert label and len(description) > 25, path
    assert parse_value("null", [1, 2], "train_sizes") is None
    assert parse_value("paper", 100, "method_params.soi.training_steps") == "paper"
    for text, previous in [("NaN", 1.), ("true", 4), ("null", "results/run"), ("1", True)]:
        with pytest.raises(ValueError):
            parse_value(text, previous, "value")


def test_progress_and_safe_stop_save_latest_step(tiny, tmp_path):
    cfg = deepcopy(tiny)
    cfg["methods"] = ["line_mine"]
    cfg["method_params"]["line_mine"]["training_steps"] = 8
    events = []
    stopped = False
    def event_callback(event):
        nonlocal stopped
        events.append(event)
        if event["event"] == "progress":
            stopped = True
    job = next(grid(cfg))
    device = configure_environment(cfg)
    with pytest.raises(KeyboardInterrupt):
        run_one(cfg, job, tmp_path, device, progress_callback=event_callback, stop_requested=lambda: stopped)
    last_event = events[-1]
    assert last_event["event"] == "job_finished" and last_event["status"] == "interrupted"
    result_path = next((tmp_path/"runs").glob("*/result.json"))
    result = json.loads(result_path.read_text())
    checkpoint = torch.load(result_path.parent/"last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["state"]["step"] == result["training_steps"] == 1
    resumed = run_one(cfg, job, tmp_path, device, resume=True, progress_callback=events.append)
    assert resumed["status"] == "ok" and resumed["training_steps"] == 8
    assert any(e["event"] == "training_started" and e["step"] == 1 for e in events)


def test_eta_learns_speed_and_handles_equal_time_and_skips():
    now = [0.]
    task = dict(job=dict(method="soi", v_dim=1), units=100, initial=0., skipped=False,
                per_unit_seconds=None, overhead_seconds=2., budget_mode="paper")
    progress = ProgressState([task, deepcopy(task)], clock=lambda: now[0])
    assert progress.remaining() is None
    progress.update(dict(event="job_started", job_index=0))
    progress.update(dict(event="training_started", step=0, training_seconds=0.))
    now[0] = 2.
    progress.update(dict(event="progress", step=20, training_seconds=2.))
    assert progress.remaining() == pytest.approx(22.)  # 8+2 for this job; 10+2 for next.
    assert progress.snapshot()["current_fraction"] == .2
    progress.update(dict(event="job_finished", status="ok", step=100, training_seconds=10.))
    assert progress.snapshot()["overall_fraction"] == .5
    assert progress.remaining() == pytest.approx(12.)
    timed = task | dict(units=5., per_unit_seconds=1., budget_mode="equal_time", overhead_seconds=0.)
    p = ProgressState([timed], clock=lambda: 0.)
    p.update(dict(event="job_started", job_index=0))
    p.update(dict(event="training_started", step=0, training_seconds=0.))
    p.update(dict(event="progress", step=1000, training_seconds=2.))
    assert p.remaining() == pytest.approx(3.)
    assert p.snapshot()["current_fraction"] == .4
    assert ProgressState([task | dict(skipped=True)]).remaining() == 0


def test_preview_only_reads_existing_timings(tiny, tmp_path):
    cfg = deepcopy(tiny)
    cfg["methods"] = ["line_mine"]
    cfg["output_dir"] = "results/new"
    ref = tmp_path/"results/smoke"
    folder = ref/"runs/reference"
    folder.mkdir(parents=True)
    p = plan(cfg)
    result = dict(status="ok", training_steps=10, training_seconds=1., parameter_count=p["models"][0]["parameters"],
                  method="line_mine", dimension=3, final_evaluation_seconds=.2)
    (folder/"result.json").write_text(json.dumps(result))
    (folder/"resolved_config.json").write_text(json.dumps(dict(config=cfg)))
    (ref/"environment.json").write_text('{"device":"cpu"}')
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    tasks = build_tasks(cfg, p, "cpu", tmp_path)
    assert len(tasks) == 1 and tasks[0]["per_unit_seconds"] == pytest.approx(.1)
    after = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert build_tasks(cfg, p, "cuda", tmp_path)[0]["per_unit_seconds"] is None


def test_native_window_launches_worker_and_receives_final_progress(tiny, tmp_path, tk_session):
    import tkinter as tk
    from soi_nuisance.gui import ExperimentApp
    root = tk.Toplevel(tk_session)
    root.withdraw()
    cfg = deepcopy(tiny)
    cfg.update(output_dir=str(tmp_path/"run"), evaluation_every=0, checkpoints_every=0)
    for params in cfg["method_params"].values():
        params["training_steps"] = 2
    path = tmp_path/"gui.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    app = ExperimentApp(root, config_path=path, auto_preview=False)
    try:
        app.start()
        assert app.process is not None
        deadline = time.monotonic()+90
        while app.busy() and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert not app.busy(), "GUI worker did not terminate"
        assert app.terminal_event == "finished", app.log.get("1.0", "end")
        assert app.tracker.snapshot()["completed"] == 3
        assert app.overall_bar["value"] == 100
        assert "3 / 3" in app.job_var.get()
        assert (tmp_path/"run/report/REPORT.md").exists()
        assert (tmp_path/"run/report/figures.pdf").exists()
        result_rows = [json.loads(line) for line in (tmp_path/"run/results.jsonl").read_text().splitlines()]
        assert len(result_rows) == 3 and all(r["status"] == "ok" for r in result_rows)
        # A report-only rebuild must not train, even with invalid draft model settings.
        raw_paths = list((tmp_path/"run/runs").glob("*/result.json"))
        before = {p: p.read_bytes() for p in raw_paths}
        app.edit_var.set("2")  # selected num_blocks: invalid for training, irrelevant to saved figures.
        assert app.apply_edit()
        app.rebuild_report()
        deadline = time.monotonic()+60
        while app.busy() and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert not app.busy() and app.terminal_event == "finished", app.log.get("1.0", "end")
        assert all(p.read_bytes() == old for p, old in before.items())
    finally:
        if app.process is not None:
            app.request_stop()
            app.process.wait(timeout=30)
        app._destroy_window()


def test_report_repair_window_selects_folder_and_rebuilds_without_training(tmp_path, tk_session, monkeypatch):
    import tkinter as tk
    from soi_nuisance.repair_gui import ReportRepairApp
    folder = tmp_path/"existing experiment"
    report = folder/"report"
    report.mkdir(parents=True)
    (report/"figures.pdf").write_bytes(b"old PDF")
    rows = [dict(run_id=f"{method}_{i}", method=method, rho=rho, omega_true=true,
                 omega_hat=true+error, status="ok", distribution_seed=77, data_seed=77, training_seed=77,
                 benchmark="mixed", num_blocks=10, u_dim=1, v_dim=2, k=16, n_train=1000)
            for method, error in [("soi", 1.), ("line_mine", .1), ("line_nwj", -1e5)]
            for i, (rho, true) in enumerate([(.1, -.2), (.6, .3)])]
    raw = folder/"results.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    before = raw.read_bytes()
    monkeypatch.setattr("soi_nuisance.repair_gui.filedialog.askdirectory", lambda **kwargs: str(report))
    root = tk.Toplevel(tk_session)
    root.withdraw()
    app = ReportRepairApp(root)
    try:
        app.choose()
        assert app.folder_var.get() == str(report)
        app.start()
        assert app.process is not None
        deadline = time.monotonic()+60
        while app.process is not None and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert app.process is None, app.log.get("1.0", "end")
        assert app.result and app.result["page_count"] == 3
        assert "SΩI не лучший" in app.verdict_var.get()
        assert str(app.open_button["state"]) == "normal"
        assert app.progress["value"] == 4
        assert raw.read_bytes() == before and not (folder/"runs").exists()
        assert (Path(app.result["backup"])/"figures.pdf").read_bytes() == b"old PDF"
    finally:
        if app.process is not None:
            app.process.wait(timeout=60)
        app.destroy()


def test_window_stop_file_and_resume_are_end_to_end(tiny, tmp_path, tk_session):
    import tkinter as tk
    from soi_nuisance.gui import ExperimentApp
    root = tk.Toplevel(tk_session)
    root.withdraw()
    cfg = deepcopy(tiny)
    cfg.update(output_dir=str(tmp_path/"run"), methods=["line_mine"],
               evaluation_every=0, checkpoints_every=0)
    cfg["method_params"]["line_mine"]["training_steps"] = 100000
    path = tmp_path/"gui_stop.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    app = ExperimentApp(root, config_path=path, auto_preview=False)
    try:
        app.start()
        requested = False
        deadline = time.monotonic()+60
        while app.busy() and time.monotonic() < deadline:
            root.update()
            if app.tracker and app.tracker.unit > 0 and not requested:
                app.request_stop()
                requested = True
            time.sleep(.02)
        assert requested and not app.busy()
        assert app.terminal_event == "stopped", app.log.get("1.0", "end")
        result_path = next((tmp_path/"run/runs").glob("*/result.json"))
        result = json.loads(result_path.read_text())
        checkpoint = torch.load(result_path.parent/"last.pt", map_location="cpu", weights_only=False)
        assert 0 < result["training_steps"] < 100000
        assert checkpoint["state"]["step"] == result["training_steps"]
        app.config["method_params"]["line_mine"]["training_steps"] = result["training_steps"]+2
        app.config["resume"] = True
        app.resume_var.set(True)
        app.start()
        deadline = time.monotonic()+60
        while app.busy() and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert not app.busy() and app.terminal_event == "finished", app.log.get("1.0", "end")
        resumed = json.loads(result_path.read_text())
        assert resumed["training_steps"] == result["training_steps"]+2
    finally:
        if app.process is not None:
            app.request_stop()
            app.process.wait(timeout=30)
        app._destroy_window()


def test_simple_window_has_five_fields_and_generates_a_real_ten_point_curve(tmp_path, tk_session, monkeypatch):
    import tkinter as tk
    import soi_nuisance.simple_gui as simple_gui
    real_builder = simple_gui.build_experiment
    def quick_builder(*args, **kwargs):
        config = real_builder(*args, **kwargs)
        config.update(device="cpu", cpu_threads=1, n_val=32, n_test=33, checkpoints_every=0)
        config["method_params"]["soi"].update(width=16, time_dim=16, mc_iter=1)
        return config
    monkeypatch.setattr(simple_gui, "build_experiment", quick_builder)
    root = tk.Toplevel(tk_session)
    root.withdraw()
    app = simple_gui.SimpleExperimentApp(root, auto_preview=False)
    try:
        assert set(app.field_vars) == {"u_dim", "v_dim", "n_samples", "k", "max_steps"}
        assert len(app.field_entries) == 5
        for key, value in dict(u_dim=1, v_dim=0, n_samples=256, k=1, max_steps=2).items():
            app.field_vars[key].set(str(value))
        app.output_var.set(str(tmp_path/"simple"))
        app.start()
        deadline = time.monotonic()+120
        while app.busy() and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert not app.busy() and app.terminal_event == "finished", app.log.get("1.0", "end")
        rows = [json.loads(line) for line in (tmp_path/"simple/results.jsonl").read_text().splitlines()]
        assert len(rows) == 30 and all(r["status"] == "ok" for r in rows)
        assert len({r["omega_true"] for r in rows}) == 10
        for method in ["soi", "line_mine", "line_nwj"]:
            assert len([r for r in rows if r["method"] == method]) == 10
        manifest = json.loads((tmp_path/"simple/report/figures_manifest.json").read_text())
        assert manifest["page_count"] == 3 and manifest["confidence"] == "disabled"
        assert all(r["stop_reason"] == "step_limit" for r in rows)
        for point in sorted({r["rho"] for r in rows}):
            assert len({json.dumps(r["data_hashes"], sort_keys=True) for r in rows if r["rho"] == point}) == 1
    finally:
        if app.process is not None:
            app.request_stop()
            app.process.wait(timeout=30)
        app._destroy_window()
