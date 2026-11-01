from copy import deepcopy
import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from soi_nuisance.comparison import assess
from soi_nuisance.comparison_plots import comparison_specs, make_comparison_page
from soi_nuisance.repair_report import repair_report


def saved_rows():
    truth = [-.2, 0., .3]
    estimates = {"soi": [-.7, -.8, -1.], "line_mine": [-.3, .02, .31], "line_nwj": [-1e5, .0001, -2e4]}
    return [dict(run_id=f"{method}_{i}", method=method, rho=i*.1, omega_true=true, omega_hat=ys[i],
                 signed_error=999., distribution_seed=77, data_seed=77, training_seed=77,
                 status="ok", stop_reason="converged", benchmark="mixed", num_blocks=10, u_dim=1, v_dim=2,
                 k=16, n_train=1000, normalization="population", condition="main", budget_mode="paper",
                 budget_multiplier=1.) for method, ys in estimates.items() for i, true in enumerate(truth)]


def test_metrics_derive_real_errors_and_do_not_prefer_soi():
    summary = assess(saved_rows())
    metrics = {r["method"]: r for r in summary["metrics"]}
    assert metrics["line_mine"]["mae"] == pytest.approx(.13/3)
    assert metrics["line_mine"]["rmse"] == pytest.approx(math.sqrt(.0105/3))
    assert metrics["line_mine"]["wins"] == 2
    assert metrics["line_nwj"]["wins"] == 1
    assert metrics["soi"]["wins"] == 0
    assert summary["best_by"] == {"mae": ["Line-MINE"], "rmse": ["Line-MINE"]}
    assert "SΩI не лучший" in summary["verdict_ru"]


def test_failed_points_use_identical_support_and_cannot_win():
    rows = saved_rows()
    rows[-1].update(status="nonfinite", omega_hat=None)
    summary = assess(rows)
    assert summary["common_points"] == summary["common_replicates"] == 2
    assert summary["excluded_replicates"] == 1
    assert all(r["compared"] == 2 for r in summary["metrics"])
    assert next(r for r in summary["metrics"] if r["method"] == "line_nwj")["failed"] == 1
    assert sum(r["wins"] for r in summary["metrics"]) == 2


def test_exact_ties_are_reported_instead_of_arbitrary_winner():
    rows = saved_rows()
    for r in rows:
        r["omega_hat"] = r["omega_true"]
    summary = assess(rows)
    assert len(summary["best_by"]["rmse"]) == 3
    assert all(r["wins"] == 0 and r["ties"] == 3 and r["rmse"] == 0 for r in summary["metrics"])
    assert "делит первое место" in summary["verdict_ru"]


def test_outer_seeds_keep_equal_weight_with_uneven_inner_repeats():
    rows = []
    for method in ("soi", "line_mine"):
        for outer, count, error in [(1, 9, 0.), (2, 1, 10.)]:
            for inner in range(count):
                rows.append(dict(method=method, omega_true=0., omega_hat=error, rho=.5,
                                 distribution_seed=outer, training_seed=inner, status="ok"))
    summary = assess(rows)
    assert all(r["mae"] == 5. and r["rmse"] == pytest.approx(math.sqrt(50)) for r in summary["metrics"])


def test_zoom_marks_outliers_but_full_page_preserves_actual_coordinates():
    rows = saved_rows()
    before = deepcopy(rows)
    specs = comparison_specs([dict(records=rows)])
    focus, full, accuracy = [make_comparison_page(spec) for spec in specs]
    try:
        assert specs[0]["window"]["zoomed"]
        assert focus.axes[0].get_ylim()[0] > -10
        assert any(len(collection.get_offsets()) for collection in focus.axes[0].collections)
        ax = full.axes[0]
        assert ax.get_yscale() == "symlog"
        for line, method in zip(ax.lines[:3], ("soi", "line_mine", "line_nwj")):
            expected = [r for r in rows if r["method"] == method]
            np.testing.assert_array_equal(line.get_xdata(), [r["omega_true"] for r in expected])
            np.testing.assert_array_equal(line.get_ydata(), [r["omega_hat"] for r in expected])
            assert ax.get_ylim()[0] <= min(line.get_ydata()) <= max(line.get_ydata()) <= ax.get_ylim()[1]
        assert rows == before
        assert len(accuracy.axes[0].lines) == 3
    finally:
        for fig in (focus, full, accuracy):
            plt.close(fig)


def test_repair_recovers_broken_run_from_jsonl_and_backs_up_only_report(tmp_path):
    rows = saved_rows()
    raw = tmp_path/"results.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    run = tmp_path/"runs"/rows[0]["run_id"]
    run.mkdir(parents=True)
    (run/"result.json").write_text("broken JSON", encoding="utf-8")
    (run/"last.pt").write_bytes(b"original model")
    (tmp_path/"data").mkdir()
    (tmp_path/"data/train.npz").write_bytes(b"original data")
    report = tmp_path/"report"
    report.mkdir()
    (report/"figures.pdf").write_bytes(b"original PDF")
    (report/"notes.txt").write_text("personal notes")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and report not in p.parents}
    result = repair_report(report)
    assert result["page_count"] == 3
    content = Path(result["pdf"]).read_bytes()
    assert len(re.findall(rb"/Type\s*/Page\b", content)) == 3
    assert all(p.read_bytes() == original for p, original in before.items())
    backup = Path(result["backup"])
    assert (backup/"figures.pdf").read_bytes() == b"original PDF"
    assert (backup/"notes.txt").read_text() == (report/"notes.txt").read_text() == "personal notes"
    assert result["warnings"] and "SΩI не лучший" in (report/"REPORT.md").read_text(encoding="utf-8")
    receipt = json.loads((report/"repair_receipt.json").read_text(encoding="utf-8"))
    assert receipt["training_started"] is False
    assert receipt["saved_estimates_unchanged"]
    assert not (tmp_path/".report_repair.lock").exists()


def test_pdf_without_raw_measurements_is_not_fabricated(tmp_path):
    report = tmp_path/"report"
    report.mkdir()
    pdf = report/"figures.pdf"
    pdf.write_bytes(b"only a picture")
    with pytest.raises(ValueError, match="PDF/PNG"):
        repair_report(report)
    assert pdf.read_bytes() == b"only a picture"
    assert not (tmp_path/"report_backups").exists()
    assert not (tmp_path/".report_repair.lock").exists()


def test_locked_pdf_keeps_previous_report_intact(tmp_path, monkeypatch):
    rows = saved_rows()
    (tmp_path/"results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    report = tmp_path/"report"
    report.mkdir()
    pdf = report/"figures.pdf"
    pdf.write_bytes(b"old PDF")
    manifest = report/"figures_manifest.json"
    manifest.write_text("old manifest")
    original_replace = Path.replace
    def replace(source, target):
        if Path(target) == pdf:
            raise PermissionError("viewer locks PDF")
        return original_replace(source, target)
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="Закройте figures.pdf"):
        repair_report(tmp_path)
    assert pdf.read_bytes() == b"old PDF"
    assert manifest.read_text() == "old manifest"
    assert not (tmp_path/".report_repair.lock").exists()
