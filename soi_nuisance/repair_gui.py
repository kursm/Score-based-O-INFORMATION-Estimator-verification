"""Separate desktop tool for rebuilding a report from a selected results folder."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]


class ReportRepairApp:
    def __init__(self, root, folder=""):
        self.root, self.process, self.result = root, None, None
        self.messages = queue.Queue()
        self.close_after_finish = False
        root.title("Исправление графиков SΩI")
        root.geometry("900x650")
        root.minsize(780, 560)
        root.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Исправить сохранённый отчёт", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Выберите папку эксперимента или её подпапку report. Обучение не запускается.",
                  wraplength=830).pack(anchor="w", pady=(6, 12))
        picker = ttk.Frame(outer)
        picker.pack(fill="x")
        self.folder_var = tk.StringVar(value=str(folder))
        self.folder_entry = ttk.Entry(picker, textvariable=self.folder_var)
        self.folder_entry.pack(side="left", fill="x", expand=True)
        self.choose_button = ttk.Button(picker, text="Выбрать папку…", command=self.choose)
        self.choose_button.pack(side="left", padx=(8, 0))
        ttk.Label(outer, text="В одном PDF: увеличенный вид, полный диапазон и сравнение ошибок с точным GT.\n"
                  "Прежний отчёт автоматически сохраняется в report_backups.", wraplength=830).pack(anchor="w", pady=12)
        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        self.start_button = ttk.Button(actions, text="Исправить отчёт", command=self.start)
        self.start_button.pack(side="left")
        self.open_button = ttk.Button(actions, text="Открыть PDF", command=self.open_pdf, state="disabled")
        self.open_button.pack(side="right")
        self.status_var = tk.StringVar(value="Готово к выбору папки")
        ttk.Label(outer, textvariable=self.status_var, wraplength=830).pack(anchor="w", pady=(16, 5))
        self.progress = ttk.Progressbar(outer, maximum=4)
        self.progress.pack(fill="x")
        self.verdict_var = tk.StringVar(value="Здесь появятся MAE, RMSE и ответ, какой метод точнее в выбранной серии.")
        ttk.Label(outer, textvariable=self.verdict_var, wraplength=830, font=("Segoe UI", 11)).pack(anchor="w", pady=12)
        self.log = tk.Text(outer, wrap="word", height=12, font=("Consolas", 10), state="disabled")
        self.log.pack(fill="both", expand=True)
        self.poll_id = root.after(100, self._poll)

    def choose(self):
        folder = filedialog.askdirectory(parent=self.root, title="Папка с результатами эксперимента")
        if folder:
            self.folder_var.set(folder)

    def _append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line+"\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.process is not None:
            return
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showerror("Папка", "Выберите папку с результатами.", parent=self.root)
            return
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        try:
            process = subprocess.Popen([str(python), "-X", "utf8", "-m", "soi_nuisance.repair_report",
                                        folder, "--json-events"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace",
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            messagebox.showerror("Запуск", str(exc), parent=self.root)
            return
        self.process, self.result = process, None
        self.progress["value"] = 0
        self.status_var.set("Подготовка…")
        self.verdict_var.set("Читаем сохранённые оценки и сравниваем их с GT…")
        for widget in (self.start_button, self.choose_button, self.folder_entry, self.open_button):
            widget.configure(state="disabled")
        def read():
            for line in process.stdout:
                self.messages.put(("line", line.rstrip()))
            process.stdout.close()
            self.messages.put(("exit", process.wait()))
        threading.Thread(target=read, daemon=True).start()

    def _event(self, event):
        if event["event"] == "progress":
            self.status_var.set(event["message"])
            self.progress["value"] = event["step"]-1
            self._append(event["message"])
        elif event["event"] == "finished":
            self.result = event["result"]
            self.progress["value"] = 4
            self.status_var.set("Отчёт исправлен: "+str(self.result["page_count"])+" страниц в одном PDF.")
            self.verdict_var.set("\n".join(group["verdict_ru"] for group in self.result["comparisons"])+
                                 "\nЭто результат этой серии; универсальная оптимальность не доказана.")
            for group in self.result["comparisons"]:
                for row in group["metrics"]:
                    value = lambda key: "нет общих точек" if row[key] is None else f"{row[key]:.8g}"
                    self._append(f"{row['label']}: MAE={value('mae')}; RMSE={value('rmse')}; "
                                 f"побед={row['wins']}; ничьих={row['ties']}; сбоев={row['failed']}")
            self._append("PDF: "+self.result["pdf"])
            self._append("Прежний отчёт: "+str(self.result["backup"]))
            for warning in self.result["warnings"]:
                self._append("Предупреждение: "+warning)
        elif event["event"] == "error":
            self.status_var.set("Не удалось исправить отчёт")
            self.verdict_var.set(event["message"])
            self._append(event["message"])

    def _poll(self):
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                if value.startswith("@repair "):
                    try:
                        self._event(json.loads(value[len("@repair "):]))
                    except (ValueError, KeyError) as exc:
                        self._append("Некорректный ответ worker: "+str(exc))
                else:
                    self._append(value)
            elif kind == "exit":
                self.process = None
                for widget in (self.start_button, self.choose_button, self.folder_entry):
                    widget.configure(state="normal")
                self.open_button.configure(state="normal" if self.result else "disabled")
                if value and self.result is None:
                    self._append("Процесс завершился с кодом "+str(value))
                if self.close_after_finish:
                    self.destroy()
                    return
        self.poll_id = self.root.after(100, self._poll)

    def open_pdf(self):
        if self.result:
            try:
                os.startfile(self.result["pdf"])
            except OSError as exc:
                messagebox.showerror("PDF", str(exc), parent=self.root)

    def close(self):
        if self.process is not None:
            self.close_after_finish = True
            self.root.withdraw()
        else:
            self.destroy()

    def destroy(self):
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Окно исправления графиков")
    parser.add_argument("folder", nargs="?", default="")
    args = parser.parse_args(argv)
    root = tk.Tk()
    ReportRepairApp(root, args.folder)
    root.mainloop()


if __name__ == "__main__":
    main()
