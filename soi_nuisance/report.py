"""CSV/JSON summaries and one paper-style PDF figure book."""
import csv
from collections import defaultdict, Counter
import json
from pathlib import Path
import numpy as np
from .data import seed_for
from .runner import collect_results, write_json
from .config import grid, method_steps
from .estimators import make_estimator

GROUP = ["method", "benchmark", "rho", "num_blocks", "u_dim", "v_dim", "normalization",
         "condition", "budget_mode", "budget_multiplier", "target_steps", "target_seconds",
         "k", "n_train", "evaluation_mode"]


def csv_write(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def aggregates(results):
    groups = defaultdict(list)
    for r in results:
        # Evaluation mode is known even if a training run failed.
        groups[tuple(r.get(k) for k in GROUP)].append(r)
    rows = []
    for key, runs in groups.items():
        row = dict(zip(GROUP, key))
        good = [r for r in runs if r["status"] == "ok"]
        row.update(total_runs=len(runs), successful_runs=len(good),
                   failed_runs=len(runs)-len(good), status_counts=dict(Counter(r["status"] for r in runs)),
                   metric_population="conditional_on_success" if len(good) < len(runs) else "all_runs")
        if good:
            clusters = defaultdict(list)
            for r in good:
                clusters[r["distribution_seed"]].append(r)
            bias = np.array([np.mean([r["signed_error"] for r in cluster])
                             for cluster in clusters.values()])
            mse = np.array([np.mean([r["squared_error"] for r in cluster])
                            for cluster in clusters.values()])
            row.update(bias=float(bias.mean()), rmse=float(np.sqrt(mse.mean())),
                       outer_repetitions=len(clusters),
                       outer_bias_std=float(bias.std(ddof=1)) if len(bias) > 1 else None,
                       run_error_std=float(np.std([r["signed_error"] for r in good], ddof=1))
                           if len(good) > 1 else None,
                       mean_training_seconds=float(np.mean([r["training_seconds"] for r in good])),
                       mean_updates=float(np.mean([r["training_steps"] for r in good])))
        rows.append(row)
    return rows


def make_report(output, epsilon=None):
    from .paper_plots import write_figure_book
    from .comparison import comparison_text, write_comparison_outputs
    output = Path(output)
    results = collect_results(output)
    if not results:
        raise ValueError("В каталоге нет сохранённых запусков result.json.")
    destination = output/"report"
    destination.mkdir(parents=True, exist_ok=True)
    rows = aggregates(results)
    csv_write(destination/"aggregates.csv", rows)
    write_json(destination/"aggregates.json", rows)
    trajectory = []
    seed_keys = ["distribution_seed", "data_seed", "training_seed"]
    for result in results:
        path = output/"runs"/result["run_id"]/"history.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                point = json.loads(line)
                error = point["omega_hat"]-result["omega_true"]
                trajectory.append({k: result.get(k) for k in GROUP+seed_keys} | dict(
                    run_id=result["run_id"], status="ok", final_status=result["status"],
                    split="validation", step=point["step"], training_seconds=point["training_seconds"],
                    omega_hat=point["omega_hat"], omega_true=result["omega_true"],
                    signed_error=error, squared_error=error*error, absolute_error=abs(error),
                    loss=point["loss"]))
    csv_write(destination/"validation_trajectory.csv", trajectory)
    thresholds = []
    if epsilon is not None:
        by_curve = defaultdict(list)
        for row in rows:
            by_curve[tuple(row.get(k) for k in GROUP if k != "n_train")].append(row)
        for curve in by_curve.values():
            eligible = [r for r in curve if r["failed_runs"] == 0 and r.get("rmse", np.inf) <= epsilon]
            thresholds.append({k: curve[0].get(k) for k in GROUP if k != "n_train"} |
                              dict(epsilon=epsilon, minimum_tested_n=min(r["n_train"] for r in eligible)
                                   if eligible else None,
                                   interpretation="point RMSE on tested grid; no extrapolation; not a confidence guarantee"))
    csv_write(destination/"sample_thresholds.csv", thresholds)
    metadata, density_rows = [], []
    for path in sorted((output/"data").glob("*/metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        metadata.append(meta)
        for coord, item in enumerate(meta["densities"]):
            density_rows.append(dict(dataset=path.parent.name, coordinate=coord,
                distribution_seed=meta["spec"]["distribution_seed"], k=item["k"],
                local_maxima=item["local_maxima"],
                standardized_knot_spacing=item["standardized_knot_spacing"]))
    csv_write(destination/"density_shapes.csv", density_rows)
    contribution_rows, regression_rows = [], []
    for r in results:
        if r["method"] != "soi":
            continue
        contribution_rows.append({k: r.get(k) for k in ["run_id", "k", "v_dim", "condition", "n_train",
            "training_seed", "distribution_seed", "data_seed", "tc_u", "tc_v", "dtc_u", "dtc_v",
            "tc_coordinates", "dtc_coordinates", "omega_true", "omega_hat"]})
        path = output/"runs"/r["run_id"]/"regression_diagnostics.json"
        if path.exists():
            regression_rows += [dict(run_id=r["run_id"], **row)
                                for row in json.loads(path.read_text(encoding="utf-8"))]
    csv_write(destination/"soi_coordinate_contributions.csv", contribution_rows)
    csv_write(destination/"soi_regression.csv", regression_rows)
    resolved_path = output/"resolved_config.json"
    report_mode = json.loads(resolved_path.read_text(encoding="utf-8")).get("report_mode", "full") if resolved_path.exists() else "full"
    book = write_figure_book(destination, results, trajectory, metadata, mode=report_mode)
    comparisons = write_comparison_outputs(destination, results)
    unique_truth = sorted({r["omega_true"] for r in results})
    text = [
        "# Отчёт эксперимента", "",
        "**Все графики: [figures.pdf](figures.pdf)** — "+str(book["page_count"])+" страниц.",
        "Статусы: "+json.dumps(dict(Counter(r["status"] for r in results)), ensure_ascii=False), "",
        "Оформление следует [статье SΩI](https://arxiv.org/html/2402.05667v3#S4.SS1):",
        "общие панели для методов, единая легенда, SΩI — синий, Line-MINE — оранжевый,",
        "Line-NWJ — красный, аналитический ответ GT — чёрный пунктир.", "",
        "## Как читать графики", "",
        "Estimated vs true: по горизонтали аналитическая Ω, по вертикали оценка метода.",
        "Совпадение с диагональю GT означает точную оценку. Другие страницы показывают",
        "эффект k, размера выборки и числа шагов, когда для этого есть несколько измеренных точек.",
        "Панели validation и финальный test разделены; подписи не используют внутренние run_id.",
        "d=u_dim+v_dim — размерность одного блока, D=N·d — полная размерность.", ""]
    text += ["Основное сравнение содержит линейный вид, полный диапазон с осью symlog и страницу ошибок.",
             "Выход за границы увеличенного вида отмечен треугольником; сами оценки не изменены.", ""]
    text += comparison_text(comparisons)
    if len(unique_truth) == 1:
        text += [
            f"В этом запуске одна истинная Ω={unique_truth[0]:.12f} nats.",
            "Поэтому Estimated vs true содержит отдельные измеренные точки.",
            "Для кривых как в статье нужна серия с разными ρ и истинными Ω.",
            "Изменение k при фиксированном U не создаёт новых значений истинной Ω.", ""]
    text += [
        "Доверительные интервалы и полосы отключены. Показаны фактические оценки методов.",
        "RMSE одной оценки совпадает с абсолютной ошибкой.",
        "При сбоях метрики условны по успешным запускам; все статусы сохранены.",
        "Gaussian-контроль v_dim=0 меняет размерность и подписан отдельно.",
        "Рост k не гарантирует роста трудности. В идеальном денойзере V сокращается между режимами.",
        "Шумовая regression MSE не является непосредственно ошибкой Ω.",
        "Гиперпараметры и checkpoint не выбираются по test/ground truth.",
        "Короткий smoke не подтверждает и не опровергает исследовательскую гипотезу.", "",
        "## Страницы PDF", ""]
    for page in book["pages"]:
        text.append(str(page["page"])+". "+page["title"].replace("\n", " — "))
    text += ["", "PNG-превью страниц: figures/. Содержание: figures_manifest.json.",
             "Старые отдельные PDF/PNG сохранены в legacy_figures/.",
             "Численные таблицы CSV/JSON и диагностика сохранены рядом; обученные модели и данные не менялись.", ""]
    (destination/"REPORT.md").write_text("\n".join(text), encoding="utf-8")
    return dict(directory=str(destination), pdf=book["pdf"], page_count=book["page_count"],
                statuses=dict(Counter(r["status"] for r in results)), aggregate_groups=len(rows))


def estimate_duration(cfg, measured_output):
    results = collect_results(measured_output)
    successes = [r for r in results if r["status"] == "ok" and r["training_steps"] > 0]
    estimates, missing = [], set()
    jobs = list(grid(cfg))
    params_cache = {}
    for job in jobs:
        candidates = [r for r in successes if r["method"] == job["method"]]
        if not candidates:
            missing.add(job["method"])
            continue
        vd, n = job["v_dim"], cfg["num_blocks"]
        ref = min(candidates, key=lambda r: abs(r["dimension"]-n*(cfg["u_dim"]+vd)))
        cache_key = (job["method"], vd)
        if cache_key not in params_cache:
            model = make_estimator(job["method"], n, cfg["u_dim"]+vd,
                                   cfg["method_params"][job["method"]])
            params_cache[cache_key] = sum(p.numel() for p in model.trainable_parameters())
        scale = params_cache[cache_key]/ref["parameter_count"]
        with (Path(measured_output)/"runs"/ref["run_id"]/"resolved_config.json").open(encoding="utf-8") as f:
            old = json.load(f)
        old_batch = old["config"]["method_params"][job["method"]]["batch_size"]
        batch_ratio = cfg["method_params"][job["method"]]["batch_size"]/old_batch
        per_step = ref["training_seconds"]/ref["training_steps"]*scale*(
            batch_ratio**2 if job["method"] == "line_nwj" else batch_ratio)
        train_seconds = (method_steps(cfg, job["method"], job["budget_multiplier"])*per_step
                         if cfg["budget_mode"] == "paper" else
                         cfg["equal_time_seconds"]*job["budget_multiplier"])
        test_ratio = cfg["n_test"]/old["config"]["n_test"]
        mc_ratio = (cfg["method_params"]["soi"]["mc_iter"]/
                    old["config"]["method_params"]["soi"]["mc_iter"]) if job["method"] == "soi" else 1.
        final_seconds = ref["final_evaluation_seconds"]*scale*test_ratio*mc_ratio
        estimates.append(dict(method=job["method"], training_seconds=train_seconds,
                              final_evaluation_seconds=final_seconds))
    total = sum(r["training_seconds"]+r["final_evaluation_seconds"] for r in estimates)
    return dict(kind="EXTRAPOLATION, NOT MEASUREMENT", planned_runs=len(jobs),
                reference_runs=len(successes), missing_methods=sorted(missing),
                measured_rates=[dict(method=r["method"], dimension=r["dimension"],
                   seconds_per_update=r["training_seconds"]/r["training_steps"],
                   training_steps=r["training_steps"], run_id=r["run_id"]) for r in successes],
                approximate_training_and_final_eval_hours=total/3600,
                illustrative_range_hours=[.5*total/3600, 3*total/3600],
                excludes="dataset generation, compression, intermediate diagnostics and checkpoint I/O",
                limitations="Parameter-count and batch-size scaling is crude; GPU overhead, larger-N critics and different shapes may change throughput substantially. Range is a heuristic, not a confidence interval.")
