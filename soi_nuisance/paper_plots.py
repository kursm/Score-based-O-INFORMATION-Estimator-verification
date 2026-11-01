"""Paper-style comparison panels and one atomic, multipage PDF figure book."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

from .density import LinearDensity

STYLE_REFERENCE = "https://arxiv.org/html/2402.05667v3#S4.SS1"
METHODS = ("soi", "line_mine", "line_nwj")
STYLES = {
    "soi": dict(color="#1f77b4", marker="*", markersize=8, linestyle="-", label=r"S$\Omega$I"),
    "line_mine": dict(color="#ff7f0e", marker="o", markersize=5, linestyle="--",
                      markerfacecolor="white", label="Line-MINE"),
    "line_nwj": dict(color="#d62728", marker="x", markersize=6, linestyle="-.", label="Line-NWJ"),
}
PAPER_RC = {
    "font.family": "DejaVu Serif", "mathtext.fontset": "stix", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": .9,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.grid": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
}
SCIENTIFIC = ("benchmark", "rho", "num_blocks", "u_dim", "normalization", "budget_mode", "budget_multiplier")


def grouped(records, keys):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row.get(k) for k in keys)].append(row)
    def order(key):
        return tuple((0, float(v)) if isinstance(v, (int, float)) else (1, str(v)) for v in key)
    return [groups[key] for key in sorted(groups, key=order)]


def cluster_summary(records, key, rmse=False):
    """Equal weight per outer seed, with nested repeats averaged within it."""
    clusters = defaultdict(list)
    for row in records:
        value = row.get(key)
        if row.get("status", "ok") == "ok" and value is not None and np.isfinite(value):
            clusters[row["distribution_seed"]].append(float(value)**2 if rmse else float(value))
    if not clusters:
        return np.nan, None
    values = np.array([np.mean(clusters[s]) for s in sorted(clusters)])
    transform = np.sqrt if rmse else lambda x: x
    mean = float(transform(values.mean()))
    return mean, None


def series(records):
    keys = {(r["method"], r.get("evaluation_mode", "")) for r in records}
    return sorted(keys, key=lambda k: (METHODS.index(k[0]), k[1]))


def series_style(method, mode, records):
    result = dict(STYLES[method])
    modes = sorted({r.get("evaluation_mode", "") for r in records if r["method"] == method})
    if len(modes) > 1:
        names = {"paper_batch_mean": "batch mean", "global_moment": "global moments",
                 "vp_upstream_constant": "upstream weight", "vp_exact": "exact weight"}
        result["label"] += " ("+names.get(mode, mode)+")"
        result["linestyle"] = ["-", "--", "-.", ":"][modes.index(mode) % 4]
    return result


def plot_comparison(ax, records, x_key, y_key, point_key=None, truth_key=None, rmse=False, identity=False):
    """Never pool k/n/conditions here: the caller supplies a single scientific facet."""
    point_key = point_key or x_key
    points = grouped(records, [point_key])
    for method, mode in series(records):
        xs, ys, low, high = [], [], [], []
        for point in points:
            rows = [r for r in point if r["method"] == method and r.get("evaluation_mode", "") == mode]
            mean, ci = cluster_summary(rows, y_key, rmse=rmse)
            x, _ = cluster_summary(rows, x_key)
            if not np.isfinite(x):
                x, _ = cluster_summary(point, x_key)
            xs.append(x)
            ys.append(mean)
            low.append(ci[0] if ci is not None else np.nan)
            high.append(ci[1] if ci is not None else np.nan)
        order = np.argsort(xs)
        xs, ys, low, high = [np.asarray(v)[order] for v in [xs, ys, low, high]]
        style = series_style(method, mode, records)
        # A single true value is a point, not a fabricated dependence curve.
        ax.plot(xs, ys, **style)
    if truth_key is not None and not identity:
        xs, truth = [], []
        for point in points:
            x, _ = cluster_summary(point, x_key)
            true, _ = cluster_summary(point, truth_key)
            xs.append(x)
            truth.append(true)
        order = np.argsort(xs)
        xs, truth = np.asarray(xs)[order], np.asarray(truth)[order]
        if len(xs) == 1:
            ax.axhline(truth[0], color="black", linestyle=":", linewidth=1.6, label="GT")
        else:
            ax.plot(xs, truth, color="black", linestyle=":", linewidth=1.6, label="GT")
    ax.tick_params(direction="out", length=4, width=.8)
    ax.margins(x=.08, y=.14)


def context(rows, include_rho=True):
    row = rows[0]
    parts = [str(row.get("benchmark", "")), "N="+str(row.get("num_blocks", "?"))]
    if include_rho and len({r.get("rho") for r in rows}) == 1:
        parts.append(r"$\rho$="+str(row.get("rho")))
    parts.append(str(row.get("budget_mode", "paper"))+" budget x"+str(row.get("budget_multiplier", 1)))
    return ", ".join(parts)


def facet_caption(rows, show_k=True, show_n=True):
    row = rows[0]
    d = row["u_dim"]+row["v_dim"]
    parts = [f"d={d}, D={row['num_blocks']*d}"]
    if row.get("condition") == "gaussian_control" or row["v_dim"] == 0:
        parts.append("Gaussian control")
    elif show_k:
        parts.append("k="+str(row["k"]))
    if show_n:
        parts.append("n="+str(row["n_train"]))
    return ", ".join(parts)


def make_curve_page(spec):
    count = len(spec["facets"])
    width = max(6.8, 3.6*count+.6)
    fig, axes = plt.subplots(1, count, figsize=(width, 4.8), squeeze=False)
    fig.subplots_adjust(left=.13 if count == 1 else .08, right=.98,
                        bottom=.22 if spec.get("compact") else .32, top=.78, wspace=.32)
    for index, (ax, facet) in enumerate(zip(axes[0], spec["facets"])):
        records = facet["records"]
        plot_comparison(ax, records, spec["x"], facet.get("y", spec["y"]),
                        point_key=spec.get("point"), truth_key=facet.get("truth", spec.get("truth")),
                        rmse=spec.get("rmse", False), identity=spec.get("identity", False))
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(facet.get("ylabel", spec["ylabel"]))
        if spec.get("logx"):
            ax.set_xscale("log")
            ticks = sorted({r[spec["x"]] for r in records if r.get(spec["x"], 0) > 0})
            if len(ticks) <= 8:
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.minorticks_off()
        if spec.get("identity"):
            truths = [r["omega_true"] for f in spec["facets"] for r in f["records"]]
            estimates = [r["omega_hat"] for f in spec["facets"] for r in f["records"]
                         if r.get("omega_hat") is not None and np.isfinite(r["omega_hat"])]
            if truths:
                lo, hi = min(truths), max(truths)
                pad = max((hi-lo)*.08, .025)
                xlim = (lo-pad, hi+pad)
                values = truths+estimates
                ypad = max((max(values)-min(values))*.12, .05)
                ax.plot(xlim, xlim, ":", color="black", label="GT", linewidth=1.6)
                ax.set(xlim=xlim, ylim=(min(values)-ypad, max(values)+ypad))
        subcaption = "\n".join(textwrap.wrap(facet["caption"], 30 if count > 1 else 65))
        if subcaption:
            ax.text(.5, -.25, f"({chr(97+index)}) "+subcaption, ha="center", va="top",
                    transform=ax.transAxes, fontsize=11, wrap=True)
    handles = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    if handles:
        legend = fig.legend(handles.values(), handles.keys(), loc="upper center", bbox_to_anchor=(.5, .91),
                            ncol=min(4, len(handles)), frameon=True, edgecolor="#cccccc", handlelength=2.2)
        legend.get_frame().set_linewidth(.7)
    fig.suptitle(spec["title"], y=.99, fontsize=13)
    fig.text(.5, .025, "\n".join(textwrap.wrap(spec["caption"], max(90, int(width*13)))),
             ha="center", va="bottom", fontsize=8.5)
    return fig


def add_curve_specs(specs, records, kind, title, families, facets, x, y, xlabel, ylabel,
                    caption, require_variation=False, **options):
    for family in grouped(records, families):
        panels = []
        for rows in grouped(family, facets):
            if require_variation and len({r[options.get("point", x)] for r in rows}) < 2:
                continue
            panels.append(dict(records=rows, caption=facet_caption(rows, show_k=x != "k", show_n=x != "n_train")))
        for start in range(0, len(panels), 4):
            specs.append(dict(kind=kind, title=title+"\n"+context(family, include_rho=not options.get("identity")),
                              facets=panels[start:start+4], x=x, y=y, xlabel=xlabel, ylabel=ylabel,
                              caption=caption, **options))


def density_pages(metadata):
    candidates = [m for m in metadata if m["densities"]]
    if not candidates:
        return []
    first_seed = min(m["spec"]["distribution_seed"] for m in candidates)
    selected = {}
    for meta in candidates:
        if meta["spec"]["distribution_seed"] == first_seed:
            selected.setdefault(meta["spec"]["k"], meta["densities"][0])
    return [dict(kind="densities", title="Nuisance densities after population standardization",
                 densities=[selected[k] for k in sorted(selected)[i:i+4]],
                 caption=f"Exact population densities, first V coordinate, distribution seed={first_seed}. "
                         "k changes shape, not necessarily statistical difficulty.")
            for i in range(0, len(selected), 4)]


def make_density_page(spec):
    fig, axes = plt.subplots(1, len(spec["densities"]), squeeze=False,
                             figsize=(max(6.8, 3.6*len(spec["densities"])+.6), 4.5))
    fig.subplots_adjust(left=.08, right=.98, bottom=.27, top=.82, wspace=.32)
    for i, (ax, item) in enumerate(zip(axes[0], spec["densities"])):
        d = LinearDensity(np.asarray(item["heights"]))
        support = (d.knots-d.mean)/d.std
        ax.plot(support, d.pdf(support, standardized=True), color="#444444", linewidth=1.5)
        ax.set(xlabel="Standardized V", ylabel="Density")
        ax.set_ylim(bottom=0)
        ax.text(.5, -.25, f"({chr(97+i)}) k={d.k}, peaks={item['local_maxima']}\n"
                f"Knot spacing={item['standardized_knot_spacing']:.3g}",
                ha="center", va="top", transform=ax.transAxes, fontsize=11)
    fig.suptitle(spec["title"], y=.98, fontsize=13)
    fig.text(.5, .025, "\n".join(textwrap.wrap(spec["caption"], 115)), ha="center", fontsize=8.5)
    return fig


def plot_specs(results, trajectory, metadata):
    specs = []
    final_caption = ("Final test estimates; dotted black line: analytic ground truth (GT). "
                     "d=u_dim+v_dim is block dimension; D=N*d is total dimension. "
                     "Confidence intervals are disabled.")
    unique_truth = len({round(r["omega_true"], 12) for r in results})
    true_caption = final_caption
    if unique_truth == 1:
        true_caption += " Only one true Omega value is present: each method is a measured point, not a dependence curve."
    add_curve_specs(specs, results, "estimated_vs_true", "Estimated vs true O-information",
                    ["benchmark", "normalization", "budget_mode", "budget_multiplier"],
                    ["num_blocks", "u_dim", "v_dim", "condition", "k", "n_train"],
                    "omega_true", "omega_hat", r"True $\Omega(X)$", r"$\Omega(X)$ estimate",
                    true_caption, point="rho", identity=True)
    main = [r for r in results if r["v_dim"] > 0 and r["condition"] == "main"]
    # All three methods and GT share each panel, independent of their method-specific evaluation names.
    for family in grouped(main, SCIENTIFIC+("v_dim", "condition", "n_train")):
        if len({r["k"] for r in family}) < 2:
            continue
        panels = [dict(records=family, caption=label, y=key, truth=truth, ylabel=label+" estimate")
                  for label, key, truth in [(r"$\Omega(X)$", "omega_hat", "omega_true"),
                                            ("TC", "tc_hat", "tc_true"), ("DTC", "dtc_hat", "dtc_true")]]
        specs.append(dict(kind="estimates_vs_k", title="Effect of nuisance shape k\n"+context(family)+
                          ", n="+str(family[0]["n_train"]), facets=panels, x="k", y="omega_hat",
                          xlabel="Nuisance shape k", ylabel="Estimate (nats)", logx=True,
                          caption=final_caption+" Dimensions and Gaussian dependencies are fixed across k."))
    add_curve_specs(specs, main, "rmse_vs_k", "O-information estimation error vs k",
                    SCIENTIFIC+("v_dim", "condition"), ["n_train"], "k", "signed_error",
                    "Nuisance shape k", r"RMSE of $\hat\Omega$ (nats)",
                    "Final test error, equal weight per independent outer replication. "
                    "No confidence intervals.",
                    require_variation=True, rmse=True, logx=True)
    for kind, y, ylabel, rmse, truth in [
        ("estimates_vs_n", "omega_hat", r"$\Omega(X)$ estimate", False, "omega_true"),
        ("rmse_vs_n", "signed_error", r"RMSE of $\hat\Omega$ (nats)", True, None)]:
        add_curve_specs(specs, results, kind, "O-information error vs training sample size" if rmse else
                        "O-information vs training sample size",
                        SCIENTIFIC+("v_dim", "condition"), ["k"], "n_train", y,
                        "Unique training objects n", ylabel, final_caption if truth else
                        "Final test RMSE, with equal weight per independent outer replication. No confidence intervals.",
                        require_variation=True, rmse=rmse, logx=True, truth=truth)
    if trajectory:
        for kind, x, y, ylabel, rmse, truth in [
            ("validation_estimates_vs_steps", "step", "omega_hat", r"$\Omega(X)$ estimate", False, "omega_true"),
            ("validation_rmse_vs_steps", "step", "signed_error", r"RMSE of $\hat\Omega$ (nats)", True, None),
            ("validation_estimates_vs_time", "training_seconds", "omega_hat", r"$\Omega(X)$ estimate", False, "omega_true")]:
            caption = ("Validation checkpoints only; final test is separate. No checkpoint is selected by true Omega. "
                       "Curves join observed checkpoints. No confidence intervals.")
            if x == "training_seconds":
                caption += " Horizontal coordinates are mean measured training times at each checkpoint."
            title = ("O-information error vs training steps (validation)" if rmse else
                     "O-information vs training steps (validation)" if x == "step" else
                     "O-information vs training time (validation)")
            add_curve_specs(specs, trajectory, kind, title,
                            SCIENTIFIC, ["v_dim", "condition", "k", "n_train"], x, y,
                            "Optimizer updates" if x == "step" else "Training time (s)", ylabel,
                            caption, point="step", rmse=rmse, truth=truth, require_variation=True)
    for x, xlabel in [("training_seconds", "Training time (s)"), ("training_steps", "Optimizer updates")]:
        for family in grouped(results, tuple(k for k in SCIENTIFIC if k != "budget_multiplier")+
                              ("v_dim", "condition", "k", "n_train")):
            if len({r["budget_multiplier"] for r in family}) < 2:
                continue
            specs.append(dict(kind="test_rmse_vs_"+x, title="Final test error vs "+xlabel.lower()+"\n"+context(family),
                              facets=[dict(records=family, caption=facet_caption(family))],
                              x=x, y="signed_error", point="budget_multiplier", xlabel=xlabel,
                              ylabel=r"RMSE of $\hat\Omega$ (nats)", rmse=True,
                              caption="Each point is the final model for a prescribed budget; no interpolation or best-checkpoint selection."))
    # SΩI coordinate diagnostics retain the cancellation between independent U/V parts.
    soi = [r for r in main if r["method"] == "soi" and r["status"] == "ok" and "tc_v" in r]
    for family in grouped(soi, SCIENTIFIC+("v_dim", "condition", "n_train")):
        if len({r["k"] for r in family}) < 2:
            continue
        enriched = []
        for r in family:
            enriched += [dict(r, omega_v=r["tc_v"]-r["dtc_v"], zero_true=0.)]
        panels = [dict(records=enriched, caption=label, y=y, truth="zero_true", ylabel=label+" (nats)")
                  for label, y in [(r"TC$_V$", "tc_v"), (r"DTC$_V$", "dtc_v"), (r"$\Omega_V$", "omega_v")]]
        specs.append(dict(kind="soi_nuisance_contributions", title="SΩI: cancellation of independent nuisance contributions\n"+
                          context(family), facets=panels, x="k", y="omega_v", xlabel="Nuisance shape k",
                          ylabel="Nuisance contribution (nats)", logx=True,
                          caption="Analytic nuisance contribution is zero (dotted GT). Omega_V=TC_V-DTC_V; "
                                  "marginal noise regression loss is not an Omega error."))
    specs += density_pages(metadata)
    return specs


def cover_page(results, specs):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.07, .93, "SΩI / Line-MINE / Line-NWJ", fontsize=23, weight="bold")
    fig.text(.07, .885, "Сравнение оценок O-информации — единый отчёт", fontsize=15)
    statuses = dict(Counter(r["status"] for r in results))
    truths = sorted({r["omega_true"] for r in results})
    outer = sorted({r["distribution_seed"] for r in results})
    lines = [f"Сохранённых запусков: {len(results)}. Статусы: {json.dumps(statuses, ensure_ascii=False)}.",
             f"Независимых внешних seeds: {len(outer)}. Все значения информации — в nats.",
             "GT — аналитический ответ; синяя линия — SΩI, оранжевая — Line-MINE, красная — Line-NWJ.",
             "Одинаковые методы находятся на общих осях; условия и Gaussian-контроль подписаны отдельно."]
    lines += ["Настройки: N="+str(sorted({r['num_blocks'] for r in results}))+", n="+
              str(sorted({r['n_train'] for r in results}))+", k="+str(sorted({r['k'] for r in results}))+".",
              "Шагов обучения: "+"; ".join(method.replace("line_", "Line-")+" "+
                  str(sorted({r['training_steps'] for r in results if r['method'] == method}))
                  for method in METHODS if any(r['method'] == method for r in results))+"."]
    if len(truths) == 1:
        lines += [f"В этих данных одна истинная Ω = {truths[0]:.6f}.",
                  "На странице «Estimated vs true» это отдельные измеренные точки.",
                  "Для зависимости как в статье нужна серия с разными ρ и истинными Ω."]
    lines += ["Доверительные интервалы отключены."]
    lines += ["Smoke проверяет работоспособность; малый бюджет не доказывает сходимости.",
              "Размерность блока d=u_dim+v_dim; полная размерность D=N·d.",
              "Шаги на validation и окончательные оценки на test показаны на разных страницах."]
    y = .83
    for line in lines:
        for wrapped in textwrap.wrap(line, 112):
            fig.text(.07, y, wrapped, fontsize=10.5)
            y -= .028
    fig.text(.07, y-.02, "Содержание", fontsize=13, weight="bold")
    y -= .06
    entries = []
    for page, spec in enumerate(specs, 2):
        title = spec["title"].split("\n")[0]
        if entries and entries[-1][2] == title:
            entries[-1] = (entries[-1][0], page, title)
        else:
            entries.append((page, page, title))
    visible = max(1, min(11, int((y-.12)/.027)))
    for first, last, title in entries[:visible]:
        pages = str(first) if first == last else f"{first}–{last}"
        fig.text(.08, y, pages+".  "+title, fontsize=10)
        y -= .027
    if len(entries) > visible:
        fig.text(.08, y, "Полное содержание: figures_manifest.json / REPORT.md", fontsize=9)
    fig.text(.07, .055, "Оформление: Figure 1–3 статьи SΩI и официальный src/vis/curves.ipynb.", fontsize=9)
    return fig


def archive_legacy_figures(destination):
    """Move only the exact file families generated by the old reporter; preserve user files."""
    pattern = re.compile(r"(?:error_vs_[nk]_\d+|test_error_vs_training_(?:seconds|steps)|"
                         r"validation_error_vs_(?:step|training_seconds)|standardized_densities)\.(?:png|pdf)$")
    owned = [p for p in destination.iterdir() if p.is_file() and pattern.fullmatch(p.name)]
    if owned:
        archive = destination/"legacy_figures"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive.mkdir(parents=True)
        root = destination.resolve()
        for source in owned:
            target = archive/source.name
            if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
                raise ValueError("Legacy figure paths must remain inside report directory")
            source.replace(target)


def write_figure_book(destination, results, trajectory, metadata, mode="full"):
    from .comparison_plots import KINDS, comparison_specs, make_comparison_page
    destination = Path(destination)
    images = destination/"figures"
    images.mkdir(parents=True, exist_ok=True)
    if mode == "curve":
        from .comparison import FACET_KEYS
        specs = comparison_specs([dict(records=rows) for rows in grouped(results, FACET_KEYS)])
    else:
        expanded = []
        for spec in plot_specs(results, trajectory, metadata):
            expanded.extend(comparison_specs(spec["facets"]) if spec["kind"] == "estimated_vs_true" else [spec])
        specs = expanded
    pdf_path, temporary = destination/"figures.pdf", destination/".figures.tmp.pdf"
    manifest = []
    try:
        with matplotlib.rc_context(PAPER_RC), PdfPages(temporary, metadata={
                "Title": "SΩI experiment — comparison figures", "Author": "SOI nuisance experiment",
                "Subject": "Estimates of O-information against analytic ground truth"}) as pdf:
            if mode != "curve":
                cover = cover_page(results, specs)
                pdf.savefig(cover)
                plt.close(cover)
                manifest.append(dict(page=1, kind="cover", title="Summary and contents"))
            for index, spec in enumerate(specs, 1 if mode == "curve" else 2):
                fig = (make_comparison_page(spec) if spec["kind"] in KINDS else
                       make_density_page(spec) if spec["kind"] == "densities" else make_curve_page(spec))
                name = f"{index:02d}_{spec['kind']}"
                fig.text(.985, .01, str(index), ha="right", fontsize=8, color="#555555")
                try:
                    pdf.savefig(fig)
                    temp_image = images/(name+".tmp.png")
                    fig.savefig(temp_image, dpi=180)
                    temp_image.replace(images/(name+".png"))
                finally:
                    plt.close(fig)
                entry = dict(page=index, kind=spec["kind"], title=spec["title"],
                             caption=spec["caption"], png="figures/"+name+".png")
                if "window" in spec:
                    entry["linear_window"] = spec["window"]
                manifest.append(entry)
            count = pdf.get_pagecount()
        temporary.replace(pdf_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    archive_legacy_figures(destination)
    contents = dict(format_version=3, pdf="figures.pdf", page_count=count, pages=manifest,
                    style_reference=STYLE_REFERENCE, confidence="disabled",
                    unique_true_omega=len({round(r["omega_true"], 12) for r in results}))
    (destination/"figures_manifest.json").write_text(json.dumps(contents, indent=2, ensure_ascii=False), encoding="utf-8")
    return dict(pdf=str(pdf_path.resolve()), page_count=count, pages=manifest)
