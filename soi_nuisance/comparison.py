"""Accuracy on matched saved runs; independent of training and torch."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

METHOD_LABELS = {"soi": "SΩI", "line_mine": "Line-MINE", "line_nwj": "Line-NWJ"}
FACET_KEYS = ("benchmark", "num_blocks", "u_dim", "v_dim", "normalization", "condition",
              "k", "n_train", "budget_mode", "budget_multiplier")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def valid(row):
    return row.get("status", "ok") == "ok" and finite(row.get("omega_hat")) and finite(row.get("omega_true"))


def series_key(row):
    return row["method"], row.get("evaluation_mode", "")


def point_key(row):
    return row.get("rho"), row["omega_true"]


def replicate_key(row):
    return point_key(row) + tuple(row.get(k) for k in ("distribution_seed", "data_seed", "training_seed"))


def labels_for(records):
    modes = defaultdict(set)
    for r in records:
        modes[r["method"]].add(r.get("evaluation_mode", ""))
    keys = sorted({series_key(r) for r in records}, key=lambda key: (list(METHOD_LABELS).index(key[0]), key[1]))
    return {key: METHOD_LABELS[key[0]]+(" ("+key[1]+")" if len(modes[key[0]]) > 1 else "") for key in keys}


def curve_values(records, key):
    """Plot means of estimates, with equal weights for outer distribution seeds."""
    points = defaultdict(list)
    for r in records:
        if series_key(r) == key:
            points[point_key(r)].append(r)
    output = []
    for point, rows in sorted(points.items(), key=lambda pair: pair[0][1]):
        clusters = defaultdict(list)
        for r in rows:
            if valid(r):
                clusters[r.get("distribution_seed")].append(r["omega_hat"])
        means = [sum(values)/len(values) for values in clusters.values()]
        output.append((point[1], sum(means)/len(means) if means else math.nan))
    return output


def _average_by_outer(values):
    clusters = defaultdict(list)
    for key, value in values:
        clusters[key[2]].append(value)
    return sum(sum(v)/len(v) for v in clusters.values())/len(clusters)


def assess(records):
    """Rank series on identical point/seed support; failed estimates never count as wins."""
    labels = labels_for(records)
    keys = sorted(labels, key=lambda key: (list(METHOD_LABELS).index(key[0]), key[1]))
    cells = {key: {} for key in keys}
    for r in records:
        key, rep = series_key(r), replicate_key(r)
        if rep in cells[key]:
            raise ValueError("Повтор одного метода/точки/seed: "+labels[key]+"; проверьте исходные результаты.")
        cells[key][rep] = r
    common = set.intersection(*[{k for k, r in cells[key].items() if valid(r)} for key in keys]) if keys else set()
    common_points = sorted({rep[:2] for rep in common}, key=lambda p: p[1])
    metrics = []
    for key in keys:
        rows = list(cells[key].values())
        errors = [(rep, cells[key][rep]["omega_hat"]-cells[key][rep]["omega_true"]) for rep in sorted(common, key=str)]
        mae = _average_by_outer([(rep, abs(e)) for rep, e in errors]) if errors else None
        rmse = math.sqrt(_average_by_outer([(rep, e*e) for rep, e in errors])) if errors else None
        metrics.append(dict(method=key[0], evaluation_mode=key[1], label=labels[key],
                            total=len(rows), valid=sum(valid(r) for r in rows),
                            failed=sum(not valid(r) for r in rows), compared=len(common),
                            mae=mae, rmse=rmse, max_absolute_error=max((abs(e) for _, e in errors), default=None),
                            bias=_average_by_outer(errors) if errors else None, wins=0, ties=0))
    points = []
    for point in common_points:
        means = []
        for key in keys:
            errors = [(rep, abs(cells[key][rep]["omega_hat"]-cells[key][rep]["omega_true"]))
                      for rep in sorted(common, key=str) if rep[:2] == point]
            means.append(_average_by_outer(errors))
        best = min(means)
        winners = [i for i, value in enumerate(means) if math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12)]
        for i in winners:
            metrics[i]["wins" if len(winners) == 1 else "ties"] += 1
        points.append(dict(rho=point[0], omega_true=point[1], winners=[metrics[i]["label"] for i in winners],
                           absolute_errors={labels[key]: means[i] for i, key in enumerate(keys)}))
    best_by = {}
    for metric in ("mae", "rmse"):
        minimum = min((r[metric] for r in metrics if r[metric] is not None), default=None)
        best_by[metric] = [r["label"] for r in metrics if minimum is not None and
                          math.isclose(r[metric], minimum, rel_tol=1e-9, abs_tol=1e-12)]
    if not common:
        verdict = "Недостаточно общих успешных точек для сравнения методов."
    elif len(keys) < 2:
        verdict = "Сохранён только один метод: сравнение с другими невозможно."
    else:
        verdict = "Минимальный RMSE на общих точках: "+", ".join(best_by["rmse"])+"."
        soi = [r for r in metrics if r["method"] == "soi"]
        if soi:
            winner = any(r["label"] in best_by["rmse"] for r in soi)
            verdict += " SΩI "+("делит первое место" if winner and len(best_by["rmse"]) > 1 else
                                    "лучший по RMSE" if winner else "не лучший по RMSE")+" в этой серии."
    all_reps = set.union(*(set(v) for v in cells.values())) if cells else set()
    return dict(metrics=metrics, points=points, common_replicates=len(common), common_points=len(common_points),
                excluded_replicates=len(all_reps-common), best_by=best_by, verdict_ru=verdict,
                interpretation="Сравнение только этих сохранённых запусков; универсальная оптимальность не доказана.",
                weighting="Same points and seeds for every series; equal weight per outer distribution seed.")


def assessments(records):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r.get(k) for k in FACET_KEYS)].append(r)
    return [dict(context=dict(zip(FACET_KEYS, key)), **assess(rows)) for key, rows in groups.items()]


def comparison_text(groups):
    text = ["## Какой метод точнее относительно GT", "",
            "Меньшая ошибка означает более точную оценку. RMSE сильнее штрафует большие промахи;",
            "MAE — средняя абсолютная ошибка. Метрики сравнивают одинаковые точки и seeds.", ""]
    for group in groups:
        text += [group["verdict_ru"], "", f"Общих точек Ω: {group['common_points']}; "
                 f"общих комбинаций точка/seed: {group['common_replicates']}; "
                 f"исключено неполных комбинаций: {group['excluded_replicates']}.", "",
                 "| Метод | MAE | RMSE | Макс. ошибка | Побед / ничьих | Сбоев |",
                 "|---|---:|---:|---:|---:|---:|"]
        for r in group["metrics"]:
            number = lambda key: "—" if r[key] is None else f"{r[key]:.8g}"
            text.append(f"| {r['label']} | {number('mae')} | {number('rmse')} | "
                        f"{number('max_absolute_error')} | {r['wins']} / {r['ties']} | {r['failed']} |")
        text += ["", "Победа в точке определяется по средней абсолютной ошибке на общих повторах.", ""]
    text += ["Это сравнение сохранённой серии, а не доказательство универсальной оптимальности метода.",
             "Статус `converged` означает срабатывание критерия плато validation loss;",
             "он не гарантирует точную Ω или устойчивость оценивателя.", "",
             "Числа: [comparison_metrics.json](comparison_metrics.json), "
             "[comparison_points.csv](comparison_points.csv).", ""]
    return text


def write_comparison_outputs(destination, records):
    destination = Path(destination)
    groups = assessments(records)
    (destination/"comparison_metrics.json").write_text(json.dumps(groups, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with (destination/"comparison_points.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["run_id", "method", "evaluation_mode", "rho", "omega_true", "omega_hat", "signed_error",
                  "absolute_error", "status", "distribution_seed", "data_seed", "training_seed", "stop_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fields}
            error = r["omega_hat"]-r["omega_true"] if valid(r) else None
            row.update(signed_error=error, absolute_error=abs(error) if error is not None else None)
            writer.writerow(row)
    return groups
