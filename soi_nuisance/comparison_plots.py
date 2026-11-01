"""Readable close-up, unabridged range and accuracy pages for saved estimates."""
import math
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .comparison import assess, curve_values, labels_for
from .paper_plots import STYLES

KINDS = ("estimated_vs_true", "comparison_full_range", "comparison_accuracy")


def _padded(values):
    lo, hi = min(values), max(values)
    pad = max((hi-lo)*.10, .025)
    return lo-pad, hi+pad


def display_window(records):
    """Keep two entire series and GT visible; mark a much larger third series explicitly."""
    truth = [r["omega_true"] for r in records]
    curves = [[y for _, y in curve_values(records, key) if math.isfinite(y)] for key in labels_for(records)]
    curves = [values for values in curves if values]
    all_values = truth+[y for values in curves for y in values]
    ordered = sorted(curves, key=lambda ys: max(abs(y) for y in ys))
    core = truth+[y for values in ordered[:2] for y in values]
    focus = len(ordered) > 2 and max(all_values)-min(all_values) > 8*max(max(core)-min(core), .05)
    return dict(ylim=_padded(core if focus else all_values), zoomed=focus,
                linthresh=max((max(core)-min(core))*.1, .001))


def comparison_specs(facets):
    specs = []
    for facet in facets:
        rows = facet["records"]
        summary = assess(rows)
        window = display_window(rows)
        for kind, title in zip(KINDS, ("Estimates vs true O-information: linear view",
                                      "Estimates vs true O-information: full range",
                                      "Accuracy against analytic ground truth")):
            specs.append(dict(kind=kind, title=title, caption=("Explicit off-scale markers; full data on next page."
                if kind == KINDS[0] and window["zoomed"] else "All saved estimates; no confidence intervals."),
                records=rows, summary=summary, window=window))
    return specs


def _context(rows):
    r = rows[0]
    return (f"{str(r.get('benchmark', 'unknown')).capitalize()} Gaussian, N={r.get('num_blocks', '?')}, "
            f"dim U={r.get('u_dim', '?')}, dim V={r.get('v_dim', '?')}, "
            f"n={r.get('n_train', '?')}, k={r.get('k', '?')}")


def _header(fig, spec):
    fig.suptitle(spec["title"], y=.98, fontsize=18)
    fig.text(.5, .915, _context(spec["records"]), ha="center", fontsize=12)


def _style(key, index):
    style = dict(STYLES[key[0]])
    style.pop("label", None)
    style.update(linewidth=1.8, zorder=3+index)
    return style


def _legend(fig, records, include_gt=True):
    labels = labels_for(records)
    handles = [Line2D([], [], label=label, **_style(key, i)) for i, (key, label) in enumerate(labels.items())]
    if include_gt:
        handles.append(Line2D([], [], color="black", linestyle=":", linewidth=1.8, label="GT"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .895), ncol=min(4, len(handles)),
               fontsize=11, frameon=True)


def _axes(ax, records):
    truths = [r["omega_true"] for r in records]
    ax.set_xlim(_padded(truths))
    ax.set_xlabel(r"True $\Omega(X)$", fontsize=13)
    ax.tick_params(direction="out", labelsize=11)
    ax.grid(True, alpha=.16, linewidth=.6)


def _readable_symlog_ticks(ax, threshold):
    bounds = ax.get_ylim()
    ax.set_yticks([v for v in ax.get_yticks() if bounds[0] <= v <= bounds[1] and
                  (v == 0 or abs(v) >= threshold)])
    ax.set_ylim(bounds)


def make_estimate_page(spec):
    rows, window = spec["records"], spec["window"]
    full = spec["kind"] == "comparison_full_range"
    fig = plt.figure(figsize=(10.5, 7.2))
    _header(fig, spec)
    _legend(fig, rows)
    ax = fig.add_axes([.14, .23, .82, .53])
    _axes(ax, rows)
    ax.set_ylabel(r"$\hat\Omega(X)$ estimate", fontsize=13)
    offscale = {}
    for i, (key, label) in enumerate(labels_for(rows).items()):
        values = curve_values(rows, key)
        xs, ys = np.array(values, dtype=float).T
        style = _style(key, i)
        if not full:
            low, high = window["ylim"]
            outside = np.isfinite(ys) & ((ys < low) | (ys > high))
            # Leave gaps instead of drawing invented connections to clipped estimates.
            ax.plot(xs, np.where(outside, np.nan, ys), **style)
            for mask, y, marker in ((ys < low, low+(high-low)*.018, "v"),
                                     (ys > high, high-(high-low)*.018, "^")):
                ax.scatter(xs[mask], np.full(mask.sum(), y), marker=marker, color=style["color"],
                           s=42, zorder=7, clip_on=True)
            if outside.any():
                offscale[label] = int(outside.sum())
        else:
            ax.plot(xs, ys, **style)
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, color="black", linestyle=":", linewidth=1.8, zorder=2)
    if full:
        ax.set_yscale("symlog", linthresh=window["linthresh"], linscale=1.)
        values = [r["omega_true"] for r in rows]+[y for key in labels_for(rows)
                  for _, y in curve_values(rows, key) if math.isfinite(y)]
        lo, hi = min(values), max(values)
        # Padding in transformed coordinates keeps the small central region readable.
        transform = ax.yaxis.get_transform()
        bounds = transform.transform(np.array([lo, hi]))
        pad = max(float(bounds[1]-bounds[0])*.10, window["linthresh"]*.1)
        ax.set_ylim(transform.inverted().transform(bounds+[-pad, pad]))
        _readable_symlog_ticks(ax, window["linthresh"])
        note = (f"Full range: all finite curve values are shown (repeated seeds averaged). Y uses symlog: linear for "
                f"|estimate| ≤ {window['linthresh']:.3g}, logarithmic outside; signs and zeros are preserved.")
    else:
        ax.set_ylim(window["ylim"])
        note = ("Linear close-up. Triangles at the boundary indicate off-scale estimates, not their values. "
                "The next page shows the full range. "+"; ".join(f"{label}: {count} off scale" for label, count in offscale.items())
                if offscale else "Linear view: all finite estimates are visible. Dotted line is the exact answer (GT).")
        if offscale:
            ax.text(.02, .97, "Off-scale: "+", ".join(f"{k} ({v})" for k, v in offscale.items()),
                    transform=ax.transAxes, va="top", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    fig.text(.5, .115, "\n".join(textwrap.wrap(note, 114)), ha="center", va="center", fontsize=10)
    fig.text(.5, .035, "Validation-loss plateau is a stopping rule, not proof of accurate O-information.",
             ha="center", fontsize=10)
    return fig


def make_accuracy_page(spec):
    rows, summary = spec["records"], spec["summary"]
    fig = plt.figure(figsize=(10.5, 8.6))
    _header(fig, spec)
    _legend(fig, rows, include_gt=False)
    fig.text(.5, .805, "\n".join(textwrap.wrap(summary["verdict_ru"], 96)), ha="center", fontsize=12)
    ax = fig.add_axes([.14, .435, .82, .30])
    _axes(ax, rows)
    ax.set_ylabel(r"$|\hat\Omega-\Omega_{GT}|$ (lower is better)", fontsize=12)
    for i, metric in enumerate(summary["metrics"]):
        xs = [p["omega_true"] for p in summary["points"]]
        ys = [p["absolute_errors"][metric["label"]] for p in summary["points"]]
        ax.plot(xs, ys, **_style((metric["method"], metric["evaluation_mode"]), i))
    truths = [r["omega_true"] for r in rows]
    threshold = max((max(truths)-min(truths))*1e-4, 1e-8)
    ax.set_yscale("symlog", linthresh=threshold)
    ax.set_ylim(bottom=0)
    _readable_symlog_ticks(ax, threshold)
    table_ax = fig.add_axes([.055, .20, .90, .15])
    table_ax.axis("off")
    number = lambda value: "—" if value is None else f"{value:.6g}"
    table = table_ax.table(cellText=[[r["label"], number(r["mae"]), number(r["rmse"]),
                                     number(r["max_absolute_error"]), f"{r['wins']} / {r['ties']}", r["failed"]]
                                    for r in summary["metrics"]],
                          colLabels=["Method", "MAE ↓", "RMSE ↓", "Max error ↓", "Wins / ties", "Failed"],
                          cellLoc="center", loc="center", colWidths=[.20, .16, .16, .18, .17, .13])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.8)
    for i, r in enumerate(summary["metrics"], 1):
        if r["label"] in summary["best_by"]["rmse"] and len(summary["metrics"]) > 1:
            table[(i, 2)].set_facecolor("#d9ead3")
    note = (f"Matched support: {summary['common_points']} true values, {summary['common_replicates']} point/seed combinations; "
            f"{summary['excluded_replicates']} incomplete combinations excluded. Wins use pointwise mean absolute error. "
            "Curves and metrics average absolute errors, not the absolute error of averaged estimates. "
            "Y uses symlog to retain both tiny errors and extreme failures.")
    fig.text(.5, .12, "\n".join(textwrap.wrap(note, 120)), ha="center", va="center", fontsize=9.5)
    fig.text(.5, .035, "Результат относится к этой серии. Универсальная оптимальность метода не доказана.",
             ha="center", fontsize=10)
    return fig


def make_comparison_page(spec):
    return make_accuracy_page(spec) if spec["kind"] == "comparison_accuracy" else make_estimate_page(spec)
