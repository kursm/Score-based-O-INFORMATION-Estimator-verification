from copy import deepcopy
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from soi_nuisance.paper_plots import cluster_summary, plot_comparison, plot_specs, STYLES
from soi_nuisance.report import make_report


def example(method="soi", k=1, outer=1, training=1, **values):
    row = dict(run_id=f"{method}_{k}_{outer}_{training}", method=method, k=k, distribution_seed=outer,
               data_seed=1, training_seed=training, benchmark="redundancy", rho=.5, num_blocks=3,
               u_dim=1, v_dim=1, dimension=6, normalization="population", condition="main",
               budget_mode="paper", budget_multiplier=1., target_steps=10, target_seconds=None,
               n_train=64, status="ok", training_steps=10, training_seconds=1.,
               omega_true=.2, omega_hat=.1, signed_error=-.1, absolute_error=.1, squared_error=.01,
               tc_true=.6, dtc_true=.4, tc_hat=.5, dtc_hat=.4,
               evaluation_mode="vp_upstream_constant" if method == "soi" else "paper_batch_mean")
    return row | values


def test_paper_comparison_keeps_methods_together_and_real_true_values():
    records = [example(method, rho=rho, omega_true=true, omega_hat=true-.1)
               for method in STYLES for rho, true in [(.7, .8), (.3, -.2)]]
    fig, ax = plt.subplots()
    try:
        plot_comparison(ax, records, "omega_true", "omega_hat", point_key="rho", identity=True)
        assert len(ax.lines) == 3
        assert [line.get_color() for line in ax.lines] == [STYLES[k]["color"] for k in STYLES]
        for line in ax.lines:
            np.testing.assert_allclose(line.get_xdata(), [-.2, .8])
            np.testing.assert_allclose(line.get_ydata(), [-.3, .7])
        assert not ax.collections  # One outer repetition: no invented confidence band.
    finally:
        plt.close(fig)


def test_plot_means_use_outer_clusters_and_confidence_is_disabled():
    records = [example(outer=1, training=i, value=0.) for i in range(9)]
    records += [example(outer=2, value=10.)]
    mean, interval = cluster_summary(records, "value")
    assert mean == 5. and interval is None
    rmse, _ = cluster_summary(records, "value", rmse=True)
    assert rmse == pytest.approx(np.sqrt(50))
    _, interval = cluster_summary(records[:9], "value")
    assert interval is None


def test_sparse_grid_does_not_generate_fake_n_sweeps_or_mix_control():
    records = [example(method, k) for method in STYLES for k in [1, 16]]
    records += [example(method, v_dim=0, condition="gaussian_control") for method in STYLES]
    specs = plot_specs(records, [], [])
    assert not any(s["kind"] in ["estimates_vs_n", "rmse_vs_n"] for s in specs)
    comparison = next(s for s in specs if s["kind"] == "estimated_vs_true")
    assert len(comparison["facets"]) == 3
    for facet in comparison["facets"]:
        assert len({r["k"] for r in facet["records"]}) == 1
        assert len({r["condition"] for r in facet["records"]}) == 1
        assert {r["method"] for r in facet["records"]} == set(STYLES)
    effect = next(s for s in specs if s["kind"] == "estimates_vs_k")
    assert {r["condition"] for f in effect["facets"] for r in f["records"]} == {"main"}


def test_one_pdf_contains_all_figures_and_preserves_raw_runs(tmp_path):
    saved = {}
    for method in STYLES:
        for k in [1, 16]:
            row = example(method, k)
            path = tmp_path/"runs"/row["run_id"]
            path.mkdir(parents=True)
            result_path = path/"result.json"
            result_path.write_text(json.dumps(row), encoding="utf-8")
            (path/"last.pt").write_bytes(b"untouched checkpoint")
            saved[result_path] = result_path.read_bytes()
            saved[path/"last.pt"] = (path/"last.pt").read_bytes()
    report = tmp_path/"report"
    report.mkdir()
    (report/"error_vs_n_0.pdf").write_bytes(b"legacy figure")
    (report/"personal_notes.pdf").write_bytes(b"user file")
    outcome = make_report(tmp_path)
    pdf = Path(outcome["pdf"])
    content = pdf.read_bytes()
    manifest = json.loads((report/"figures_manifest.json").read_text(encoding="utf-8"))
    assert content.startswith(b"%PDF") and content.rstrip().endswith(b"%%EOF")
    assert len(re.findall(rb"/Type\s*/Page\b", content)) == manifest["page_count"] == outcome["page_count"]
    assert outcome["page_count"] >= 4
    for entry in manifest["pages"]:
        if "png" in entry:
            with Image.open(report/entry["png"]) as image:
                image.verify()
    assert {p.name for p in report.glob("*.pdf")} == {"figures.pdf", "personal_notes.pdf"}
    assert next((report/"legacy_figures").rglob("error_vs_n_0.pdf")).read_bytes() == b"legacy figure"
    assert (report/"personal_notes.pdf").read_bytes() == b"user file"
    assert all(path.read_bytes() == contents for path, contents in saved.items())
    assert "[figures.pdf](figures.pdf)" in (report/"REPORT.md").read_text(encoding="utf-8")
