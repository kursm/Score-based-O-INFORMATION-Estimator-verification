"""Rebuild saved comparisons without importing or calling any training code."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .comparison import METHOD_LABELS, comparison_text, finite, write_comparison_outputs


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_folder(path):
    folder = Path(path).expanduser().resolve()
    if folder.is_file():
        folder = folder.parent
    if folder.name.lower() == "report":
        folder = folder.parent
    if not folder.is_dir():
        raise ValueError("Папка не найдена: "+str(folder))
    return folder


def load_saved_results(folder):
    """Prefer individual results; recover missing/broken ones from JSONL or CSV."""
    folder = Path(folder)
    rows, sources, warnings = {}, {}, []
    invalid_files = []

    def add(row, path, fallback=False, inferred_id=None):
        if not isinstance(row, dict) or row.get("method") not in METHOD_LABELS:
            raise ValueError("неизвестный или отсутствующий метод")
        if not finite(row.get("omega_true")):
            raise ValueError("нет конечного численного omega_true")
        row = dict(row)
        run_id = row.get("run_id") or inferred_id
        if not run_id:
            raise ValueError("нет run_id")
        row.update(run_id=run_id, status=row.get("status", "ok"), omega_hat=row.get("omega_hat"))
        if row["status"] == "ok" and row["omega_hat"] is None:
            raise ValueError("нет оценки omega_hat")
        if fallback and run_id in rows:
            return
        if run_id in rows:
            raise ValueError("повтор run_id: "+run_id)
        rows[run_id] = row
        sources[path] = _sha256(path)

    for path in sorted((folder/"runs").glob("*/result.json")):
        try:
            add(json.loads(path.read_text(encoding="utf-8-sig")), path, inferred_id=path.parent.name)
        except (OSError, ValueError, TypeError) as exc:
            invalid_files.append(path)
            warnings.append(str(path.relative_to(folder))+": "+str(exc))
    fallback = folder/"results.jsonl"
    if fallback.exists():
        for line_number, line in enumerate(fallback.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    add(json.loads(line), fallback, fallback=True)
                except (ValueError, TypeError) as exc:
                    warnings.append(f"results.jsonl:{line_number}: {exc}")
    elif (folder/"results.csv").exists():
        fallback = folder/"results.csv"
        with fallback.open(encoding="utf-8-sig", newline="") as stream:
            for line_number, raw in enumerate(csv.DictReader(stream), 2):
                row = {}
                for key, value in raw.items():
                    try:
                        row[key] = json.loads(value) if value else None
                    except (ValueError, TypeError):
                        row[key] = value
                try:
                    add(row, fallback, fallback=True)
                except (ValueError, TypeError) as exc:
                    warnings.append(f"results.csv:{line_number}: {exc}")
    if not rows:
        raise ValueError("Нет пригодных result.json, results.jsonl или results.csv. "
                         "Из одного PDF/PNG восстановить оценки методов невозможно.")
    for path in invalid_files:
        if path.parent.name not in rows:
            raise ValueError("Не удалось восстановить повреждённый запуск "+path.parent.name+
                             ". Исходные файлы сохранены; исправьте JSON или добавьте results.jsonl.")
    return list(rows.values()), sources, warnings


def repair_report(path, progress=None):
    emit = progress or (lambda event: None)
    folder = experiment_folder(path)
    lock_path = folder/".report_repair.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("В этой папке уже выполняется исправление отчёта (.report_repair.lock).") from None
    os.close(descriptor)
    try:
        emit(dict(event="progress", step=1, message="Чтение сохранённых оценок…"))
        records, sources, warnings = load_saved_results(folder)
        destination = folder/"report"
        backup = None
        with tempfile.TemporaryDirectory(prefix=".report_repair_", dir=folder) as temporary:
            staging = Path(temporary)
            if not staging.resolve().is_relative_to(folder):
                raise ValueError("Временный отчёт должен находиться внутри выбранной папки.")
            emit(dict(event="progress", step=2, message="Построение графиков и сравнение ошибок с GT…"))
            from .paper_plots import write_figure_book
            book = write_figure_book(staging, records, [], [], mode="curve")
            comparisons = write_comparison_outputs(staging, records)
            text = ["# Исправленный отчёт", "", "**[Все графики — figures.pdf](figures.pdf)**", "",
                    "1. Линейный вид: огромные значения отмечены треугольниками у границы.",
                    "2. Полный диапазон: symlog сохраняет знак, ноль и все конечные оценки.",
                    "3. Ошибки относительно GT: MAE, RMSE, максимальная ошибка и победы в точках.", "",
                    "Новые цвета/штрихи/маркеры помогают различать совпадающие кривые.",
                    "Обучение не запускалось. Оценки, данные и checkpoint не изменены.", ""]
            text += comparison_text(comparisons)
            text += ["## Предупреждения чтения", ""]+["- "+w for w in warnings] if warnings else []
            (staging/"REPORT.md").write_text("\n".join(text), encoding="utf-8")
            if not all(path.exists() and _sha256(path) == digest for path, digest in sources.items()):
                raise ValueError("Исходные оценки изменились во время построения. Повторите после завершения обучения.")
            emit(dict(event="progress", step=3, message="Сохранение копии прежнего отчёта…"))
            if destination.exists():
                backup = folder/"report_backups"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
                shutil.copytree(destination, backup)
            receipt = dict(format_version=1, input_folder=str(folder), backup=str(backup) if backup else None,
                           source_sha256={str(p.relative_to(folder)): digest for p, digest in sources.items()},
                           warnings=warnings, training_started=False, saved_estimates_unchanged=True)
            (staging/"repair_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
            emit(dict(event="progress", step=4, message="Сохранение исправленного PDF…"))
            destination.mkdir(exist_ok=True)
            # Publish PDF first: a PDF viewer can lock it on Windows. Other files stay intact on failure.
            files = sorted((p for p in staging.rglob("*") if p.is_file()), key=lambda p: (p.name != "figures.pdf", str(p)))
            published = []
            try:
                for source in files:
                    relative = source.relative_to(staging)
                    target = destination/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(target)
                    published.append(relative)
            except OSError as exc:
                for relative in reversed(published):
                    original = backup/relative if backup else None
                    if original and original.exists():
                        shutil.copy2(original, destination/relative)
                    else:
                        (destination/relative).unlink(missing_ok=True)
                raise OSError("Не удалось заменить отчёт. Закройте figures.pdf в просмотрщике и повторите. "
                              "Прежний отчёт сохранён"+(" в "+str(backup) if backup else "")+".") from exc
        result = dict(directory=str(destination), pdf=str(destination/"figures.pdf"), page_count=book["page_count"],
                      backup=str(backup) if backup else None, comparisons=comparisons, warnings=warnings)
        emit(dict(event="finished", result=result))
        return result
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Исправить графики по сохранённым оценкам, без обучения")
    parser.add_argument("folder", help="Папка эксперимента или её подпапка report")
    parser.add_argument("--json-events", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    def progress(event):
        if args.json_events:
            print("@repair "+json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
        elif event["event"] == "progress":
            print(f"[{event['step']}/4] {event['message']}", flush=True)
    try:
        result = repair_report(args.folder, progress)
        if not args.json_events:
            print("PDF: "+result["pdf"])
            print("Копия прежнего отчёта: "+str(result["backup"]))
            for group in result["comparisons"]:
                print(group["verdict_ru"])
        return 0
    except Exception as exc:
        if args.json_events:
            progress(dict(event="error", message=str(exc)))
        else:
            print("Ошибка: "+str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
