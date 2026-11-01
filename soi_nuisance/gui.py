"""Small native desktop launcher. Training runs in an isolated child process."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from .gui_fields import (CHOICES, editor_text, field_help, get_value, iter_fields,
                         parse_value, read_raw_config, set_value)
from .gui_progress import ProgressState, format_duration

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("smoke", "gaussian_control", "timing_n6", "pilot", "paper", "full")
METHOD_NAMES = {"soi": "SΩI", "line_mine": "Line-MINE", "line_nwj": "Line-NWJ"}
PHASES = {
    "job_started": "Подготовка данных и модели", "training_started": "Обучение",
    "progress": "Обучение", "training": "Обучение", "validation": "Проверка на validation",
    "evaluation": "Финальное оценивание на test", "diagnostics": "Диагностика SΩI",
    "job_finished": "Задача завершена", "report": "Построение таблиц и графиков",
    "experiment_started": "Подготовка эксперимента",
    "convergence_check": "Проверка сходимости", "convergence": "Проверка сходимости",
}


def worker_python():
    executable = Path(sys.executable)
    console = executable.with_name("python.exe")
    return str(console if executable.name.lower() == "pythonw.exe" and console.exists() else executable)


def category(path):
    if path.startswith("method_params."):
        return METHOD_NAMES[path.split(".")[1]]
    if path.endswith("_seeds"):
        return "Повторения"
    if path in {"benchmark", "rho", "mixed_subsystems", "num_blocks", "u_dim", "v_dim", "k_values",
                "normalization", "transformation", "train_sizes", "n_start", "n_multiplier", "n_levels",
                "n_val", "n_test", "control_v_dim_zero"}:
        return "Данные"
    if path in {"budget_mode", "budget_multipliers", "equal_time_seconds", "wall_time_seconds",
                "max_run_seconds", "methods"}:
        return "Бюджет и методы"
    return "Выполнение и сохранение"


class ExperimentApp:
    def __init__(self, root, preset="smoke", config_path=None, auto_preview=True):
        self.root = root
        self.config = {}
        self.selected = None
        self.messages = queue.Queue()
        self.process = None
        self.mode = None
        self.stop_path = None
        self.tracker = None
        self.plan = None
        self.terminal_event = None
        self.started_at = None
        self.close_after_stop = False
        self._token = 0
        self._poll_id = self._preview_id = None
        self.root.title("SΩI — параметры и запуск эксперимента")
        self.root.geometry("1180x850")
        self.root.minsize(940, 720)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=25)
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<F5>", lambda event: self.start())
        self.root.bind("<Control-s>", lambda event: self.save_config())
        if config_path:
            self.load_file(config_path)
        else:
            self.load_preset(preset)
        self._poll_id = self.root.after(100, self._poll)
        if auto_preview:
            self._preview_id = self.root.after(200, self.preview)

    def _build(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="Сравнение оценивателей O-информации", font=("Segoe UI", 14, "bold")).pack(side="left")
        self.preset_var = tk.StringVar(value="smoke")
        self.preset_combo = ttk.Combobox(top, textvariable=self.preset_var, values=PRESETS, width=19, state="readonly")
        self.preset_combo.pack(side="right")
        self.preset_combo.bind("<<ComboboxSelected>>", lambda event: self.load_preset(self.preset_var.get()))
        ttk.Label(top, text="Пресет: ").pack(side="right")
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 10))
        self.load_button = ttk.Button(toolbar, text="Загрузить JSON…", command=self.choose_config)
        self.load_button.pack(side="left")
        self.save_button = ttk.Button(toolbar, text="Сохранить JSON…", command=self.save_config)
        self.save_button.pack(side="left", padx=8)
        self.source_var = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.source_var).pack(side="left", padx=8)
        self.resume_var = tk.BooleanVar()
        self.resume_check = ttk.Checkbutton(toolbar, text="Продолжить (resume)", variable=self.resume_var,
                                           command=self._toggle_resume)
        self.resume_check.pack(side="right")

        pane = ttk.Panedwindow(outer, orient="horizontal")
        pane.pack(fill="both", expand=True)
        parameters = ttk.Frame(pane)
        details = ttk.Frame(pane, padding=(12, 0, 0, 0))
        pane.add(parameters, weight=3)
        pane.add(details, weight=2)
        search_row = ttk.Frame(parameters)
        search_row.pack(fill="x", pady=(0, 6))
        ttk.Label(search_row, text="Поиск параметра: ").pack(side="left")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_var.trace_add("write", lambda *args: self._render_fields())
        tree_frame = ttk.Frame(parameters)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=("value",), selectmode="browse")
        self.tree.heading("#0", text="Параметр")
        self.tree.heading("value", text="Установленное значение")
        self.tree.column("#0", width=315, minwidth=190)
        self.tree.column("value", width=200, minwidth=100)
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._select_field)
        self.tree.bind("<Double-1>", lambda event: self.editor.focus_set())

        self.field_title = tk.StringVar(value="Выберите параметр слева")
        self.field_path = tk.StringVar()
        self.description = tk.StringVar()
        ttk.Label(details, textvariable=self.field_title, font=("Segoe UI", 11, "bold"), wraplength=380).pack(anchor="w")
        ttk.Label(details, textvariable=self.field_path, foreground="#5a6470", wraplength=380).pack(anchor="w", pady=(3, 8))
        self.help_label = ttk.Label(details, textvariable=self.description, wraplength=380, justify="left")
        self.help_label.pack(fill="x", anchor="w")
        details.bind("<Configure>", lambda event: self.help_label.configure(wraplength=max(230, event.width-20)))
        edit_row = ttk.Frame(details)
        edit_row.pack(fill="x", pady=(12, 3))
        self.edit_var = tk.StringVar()
        self.editor = ttk.Combobox(edit_row, textvariable=self.edit_var)
        self.editor.pack(side="left", fill="x", expand=True)
        self.editor.bind("<Return>", lambda event: self.apply_edit())
        self.apply_button = ttk.Button(edit_row, text="Применить", command=self.apply_edit)
        self.apply_button.pack(side="left", padx=(6, 0))
        ttk.Label(details, text="Списки: [1, 16]  ·  флаги: true/false  ·  авто: null", foreground="#5a6470", wraplength=380).pack(anchor="w")
        ttk.Separator(details).pack(fill="x", pady=12)
        ttk.Label(details, text="План и ожидаемое время", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.plan_text = tk.Text(details, height=11, wrap="word", font=("Segoe UI", 9), state="disabled", relief="flat")
        self.plan_text.pack(fill="both", expand=True, pady=(6, 0))
        self._set_plan_text("Нажмите «Оценить время».\n\nПрогноз использует сохранённые замеры и уточняется по скорости текущего обучения.")

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(10, 8))
        self.preview_button = ttk.Button(actions, text="Оценить время", command=self.preview)
        self.preview_button.pack(side="left")
        self.start_button = ttk.Button(actions, text="Запустить работу  (F5)", command=self.start)
        self.start_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(actions, text="Остановить", command=self.request_stop, state="disabled")
        self.stop_button.pack(side="left")
        self.results_button = ttk.Button(actions, text="Открыть результаты", command=self.open_results)
        self.results_button.pack(side="right")
        self.pdf_button = ttk.Button(actions, text="Открыть PDF", command=self.open_pdf)
        self.pdf_button.pack(side="right", padx=6)
        self.rebuild_button = ttk.Button(actions, text="Пересобрать PDF", command=self.rebuild_report)
        self.rebuild_button.pack(side="right")
        self.directory_button = ttk.Button(actions, text="Каталог результатов…", command=self.choose_output)
        self.directory_button.pack(side="right", padx=8)

        self.status_var = tk.StringVar(value="Готово к запуску")
        ttk.Label(outer, textvariable=self.status_var, wraplength=1100).pack(anchor="w")
        self.job_var = tk.StringVar(value="Задачи: 0 / 0")
        ttk.Label(outer, textvariable=self.job_var).pack(anchor="w", pady=(5, 0))
        self.overall_bar = ttk.Progressbar(outer, maximum=100)
        self.overall_bar.pack(fill="x", pady=(2, 3))
        self.step_var = tk.StringVar(value="Обучение ещё не началось")
        ttk.Label(outer, textvariable=self.step_var).pack(anchor="w")
        self.step_bar = ttk.Progressbar(outer, maximum=100)
        self.step_bar.pack(fill="x", pady=(2, 4))
        self.time_var = tk.StringVar(value="Осталось: уточняется по ходу работы")
        ttk.Label(outer, textvariable=self.time_var).pack(anchor="w", pady=(0, 6))
        self.log = tk.Text(outer, height=7, wrap="word", font=("Consolas", 9), state="disabled")
        self.log.pack(fill="x")

    def _set_plan_text(self, text):
        self.plan_text.configure(state="normal")
        self.plan_text.delete("1.0", "end")
        self.plan_text.insert("1.0", text)
        self.plan_text.configure(state="disabled")

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip()+"\n")
        if int(self.log.index("end-1c").split(".")[0]) > 1500:
            self.log.delete("1.0", "501.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def busy(self):
        return self.process is not None

    def _controls(self, busy):
        state = "disabled" if busy else "normal"
        for widget in [self.load_button, self.save_button, self.resume_check, self.apply_button,
                       self.editor, self.preview_button, self.start_button, self.directory_button, self.rebuild_button]:
            widget.configure(state=state)
        self.preset_combo.configure(state="disabled" if busy else "readonly")
        self.stop_button.configure(state="normal" if busy and self.mode == "run" else "disabled")

    def load_preset(self, preset):
        if self.busy():
            return
        self.preset_var.set(preset)
        self.load_file(ROOT/"configs"/(preset+".json"), fresh_output=True)

    def load_file(self, path, fresh_output=False):
        try:
            config = read_raw_config(path)
            for key, _ in iter_fields(config):
                field_help(key)
            if fresh_output:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                config["output_dir"] = "results/gui_"+Path(path).stem+"_"+stamp+"_"+uuid.uuid4().hex[:4]
                config["resume"] = False
            self.config, self.plan, self.tracker = config, None, None
            self.resume_var.set(config["resume"])
            self.source_var.set(Path(path).name)
            self.search_var.set("")
            self.selected = None
            self._render_fields()
            self.tree.selection_set("num_blocks")
            self.tree.see("num_blocks")
            self._select_field()
            self.status_var.set("Параметры загружены. "+(
                "Для пресета выбран новый каталог результатов." if fresh_output else str(path)))
            self._set_plan_text("Оценка ещё не рассчитана для этих параметров.")
        except Exception as exc:
            messagebox.showerror("Не удалось загрузить конфигурацию", str(exc), parent=self.root)

    def _render_fields(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        query = self.search_var.get().casefold().strip()
        for path, value in iter_fields(self.config):
            label, help_text = field_help(path)
            if query and query not in (path+" "+label+" "+help_text).casefold():
                continue
            group = "group:"+category(path)
            if not self.tree.exists(group):
                self.tree.insert("", "end", iid=group, text=category(path), open=True)
            parent = group
            for part, title in [(".optimizer.", "Оптимизатор"), (".sde.", "VP-SDE")]:
                if part in path:
                    parent = group+part
                    if not self.tree.exists(parent):
                        self.tree.insert(group, "end", iid=parent, text=title, open=bool(query))
            shown = "авто / отключено (null)" if value is None else editor_text(value)
            self.tree.insert(parent, "end", iid=path, text=label, values=(shown,))
        if self.selected and self.tree.exists(self.selected):
            self.tree.selection_set(self.selected)

    def _select_field(self, event=None):
        selection = self.tree.selection()
        if not selection or selection[0].startswith("group:"):
            return
        self.selected = selection[0]
        label, description = field_help(self.selected)
        value = get_value(self.config, self.selected)
        self.field_title.set(label)
        self.field_path.set(self.selected)
        self.description.set(description)
        self.edit_var.set(editor_text(value))
        choices = ("true", "false") if isinstance(value, bool) else CHOICES.get(self.selected.split(".")[-1], ())
        self.editor.configure(values=choices)

    def apply_edit(self, show_error=True):
        if self.busy() or not self.selected:
            return True
        try:
            previous = get_value(self.config, self.selected)
            value = parse_value(self.edit_var.get(), previous, self.selected)
            if value != previous or type(value) != type(previous):
                set_value(self.config, self.selected, value)
                self.resume_var.set(self.config["resume"])
                self.plan = self.tracker = None
                self._render_fields()
                self._set_plan_text("Параметры изменены. Нажмите «Оценить время» для нового прогноза.")
            return True
        except Exception as exc:
            if show_error:
                messagebox.showerror("Некорректное значение", str(exc), parent=self.root)
            return False

    def _toggle_resume(self):
        self.config["resume"] = self.resume_var.get()
        if self.selected == "resume":
            self.edit_var.set(editor_text(self.config["resume"]))
        self._render_fields()
        self.plan = self.tracker = None

    def choose_config(self):
        path = filedialog.askopenfilename(parent=self.root, title="Открыть конфигурацию",
                                          initialdir=ROOT/"configs", filetypes=[("JSON", "*.json")])
        if path:
            self.load_file(path)

    def save_config(self):
        if self.busy() or not self.apply_edit():
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Сохранить параметры",
                                            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")
                self.status_var.set("Конфигурация сохранена: "+path)
            except OSError as exc:
                messagebox.showerror("Не удалось сохранить", str(exc), parent=self.root)

    def choose_output(self):
        path = filedialog.askdirectory(parent=self.root, title="Каталог результатов", initialdir=ROOT/"results")
        if path:
            self.config["output_dir"] = path
            if self.selected == "output_dir":
                self.edit_var.set(path)
            self.plan = self.tracker = None
            self._render_fields()

    def open_results(self):
        directory = (ROOT/self.config["output_dir"]).resolve()
        if not directory.is_dir():
            messagebox.showinfo("Результаты", "Каталог ещё не создан. Он появится после запуска.", parent=self.root)
            return
        if os.name == "nt":
            os.startfile(directory)
        else:
            import webbrowser
            webbrowser.open(directory.as_uri())

    def open_pdf(self):
        path = (ROOT/self.config["output_dir"]/"report/figures.pdf").resolve()
        if not path.is_file():
            messagebox.showinfo("Единый PDF", "PDF ещё не создан. Используйте «Пересобрать PDF» для сохранённых результатов.",
                                parent=self.root)
            return
        if os.name == "nt":
            os.startfile(path)
        else:
            import webbrowser
            webbrowser.open(path.as_uri())

    def rebuild_report(self):
        if self.busy() or not self.apply_edit():
            return
        directory = ROOT/self.config["output_dir"]
        if not any((directory/"runs").glob("*/result.json")):
            messagebox.showinfo("Нет сохранённых результатов", "Выберите каталог завершённого эксперимента.", parent=self.root)
            return
        self._launch("report")

    def preview(self):
        if not self.busy() and self.apply_edit():
            self._launch("preview")

    def start(self):
        if self.busy() or not self.apply_edit():
            return
        output = ROOT/self.config["output_dir"]
        if (output/"resolved_config.json").exists() and not self.config["resume"] and not self.config["dry_run"]:
            messagebox.showerror("Каталог уже содержит результаты",
                                 "Включите «Продолжить (resume)» или выберите новый каталог результатов.", parent=self.root)
            return
        self._launch("run")

    def _launch(self, mode):
        session = ROOT/"results/gui_sessions"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
        session.mkdir(parents=True)
        config_path = session/"config.json"
        config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")
        self.stop_path = session/"stop.request"
        self.mode, self.terminal_event = mode, None
        self.started_at = time.monotonic()
        self.tracker = None
        self._token += 1
        token = self._token
        command = [worker_python(), "-u", "-X", "utf8", "-m", "soi_nuisance.gui_worker",
                   "--config", str(config_path), "--mode", mode, "--stop-file", str(self.stop_path)]
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                            bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            self.process = None
            self.status_var.set("Не удалось запустить: "+str(exc))
            return
        self._controls(True)
        self.status_var.set("Расчёт плана и загрузка библиотек…")
        self._append_log("Проверка параметров…" if mode == "preview" else "Запуск эксперимента…")
        process = self.process
        def reader():
            with process.stdout:
                for line in process.stdout:
                    if line.startswith("@soi-event "):
                        try:
                            self.messages.put((token, "event", json.loads(line[len("@soi-event "):])))
                        except json.JSONDecodeError:
                            self.messages.put((token, "log", line))
                    else:
                        self.messages.put((token, "log", line))
            self.messages.put((token, "exit", process.wait()))
        threading.Thread(target=reader, daemon=True).start()

    def _event(self, event):
        kind = event["event"]
        if kind == "plan":
            self.plan = event["plan"]
            self.tracker = ProgressState(event["tasks"])
            duration = format_duration(self.tracker.remaining())
            text = [f"Задач: {self.plan['runs']}", f"Устройство: {event['device']}",
                    f"Размерность D: {self.config['num_blocks']*(self.config['u_dim']+self.config['v_dim'])}",
                    f"Данные X: {self.plan['uncompressed_data_bytes']/2**20:.2f} МиБ",
                    "Предварительно ≈ "+duration, "Результаты: "+self.config["output_dir"], ""]
            for model in self.plan["models"]:
                architecture = model["architecture"]
                width = architecture.get("width", architecture.get("base_width"))
                text.append(f"{METHOD_NAMES[model['method']]}, dv={model['v_dim']}: "
                            f"{model['parameters']:,} параметров, ширина {width}")
            text += ["", "Прогноз приблизителен: загрузка библиотек, запись checkpoint и отчёта могут увеличить время.",
                     "Оценка уточняется по фактическим шагам. Другие устройство, размерность или batch меняют скорость."]
            if self.config["dry_run"]:
                text.insert(0, "Только план: dry_run=true")
            self._set_plan_text("\n".join(text))
            self.job_var.set(f"Задачи: 0 / {self.plan['runs']}")
        elif kind in ["preview_finished", "finished", "stopped", "error"]:
            self.terminal_event = kind
            if self.tracker:
                snapshot = self.tracker.snapshot()
                self.job_var.set(f"Завершено задач: {snapshot['completed']} / {snapshot['total']} · со сбоями: {snapshot['failed']}")
                self.overall_bar["value"] = 100*snapshot["overall_fraction"]
                self.step_bar["value"] = 100*snapshot["current_fraction"]
            if kind == "preview_finished":
                self.status_var.set("План готов. Обучение не запускалось.")
            elif kind == "finished":
                self.overall_bar["value"] = self.step_bar["value"] = 100
                self.status_var.set(f"Работа завершена. Успешных запусков в каталоге: {event['successful']} из {event['total']}.")
                self.step_var.set("Единый PDF готов: report/figures.pdf")
                self.time_var.set("Завершено за "+format_duration(time.monotonic()-self.started_at))
            elif kind == "stopped":
                self.status_var.set("Остановлено. Для продолжения включите resume." if self.config["save_checkpoints"]
                                    else "Остановлено. Сохранение checkpoint было отключено.")
                self.time_var.set("Остановлено через "+format_duration(time.monotonic()-self.started_at))
            else:
                self.status_var.set("Ошибка: "+event["message"])
                self.time_var.set("Работа не выполняется. Исправьте параметры и повторите запуск.")
                self._append_log("Ошибка: "+event["message"])
        else:
            if self.tracker:
                self.tracker.update(event)
            if kind == "job_started":
                job = event["job"]
                self._target_steps = event["target_steps"]
                self._target_seconds = event["target_seconds"]
                self._job_title = (f"{METHOD_NAMES[job['method']]} · k={job['k']} · n={job['n_train']} · "
                                   f"dv={job['v_dim']} · seed={job['distribution_seed']}/{job['data_seed']}/{job['training_seed']}")
            if not (self.stop_path and self.stop_path.exists()):
                self.status_var.set(PHASES.get(kind, kind)+" — "+getattr(self, "_job_title", ""))
            if kind == "progress":
                counter = (f"Шаг {event['step']:,} / {self._target_steps:,}" if self.config["budget_mode"] == "paper"
                           else f"Обучение {event['training_seconds']:.1f} / {self._target_seconds:g} с · шаг {event['step']:,}")
                self.step_var.set(counter+f" · loss={event['loss']:.5g}")
            if kind == "report":
                self.step_var.set("Собираются таблицы и единый PDF. Обучение не выполняется.")

    def _poll(self):
        for _ in range(150):
            try:
                token, kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if token != self._token:
                continue
            if kind == "event":
                self._event(value)
            elif kind == "log":
                self._append_log(value)
            elif kind == "exit":
                self.process = None
                self._controls(False)
                if self.terminal_event is None:
                    self.status_var.set(f"Процесс завершился с кодом {value}. Подробности в журнале.")
                if self.close_after_stop:
                    self._destroy_window()
                    return
        if self.tracker and self.terminal_event not in ["finished", "stopped", "error"]:
            snapshot = self.tracker.snapshot()
            self.overall_bar["value"] = 100*snapshot["overall_fraction"]
            self.step_bar["value"] = 100*snapshot["current_fraction"]
            self.job_var.set(f"Завершено задач: {snapshot['completed']} / {snapshot['total']} · со сбоями: {snapshot['failed']}")
            elapsed = format_duration(time.monotonic()-self.started_at) if self.started_at else "0 с"
            label = "Ожидаемое время ≈ " if self.mode == "preview" else "Осталось ≈ "
            self.time_var.set(label+format_duration(snapshot["remaining_seconds"])+
                              ("   ·   Прошло: "+elapsed if self.mode == "run" else ""))
        self._poll_id = self.root.after(100, self._poll)

    def request_stop(self):
        if self.busy() and self.mode == "run":
            self.stop_path.touch()
            self.status_var.set("Остановка после безопасного шага. Ждём завершения текущей операции…")
            self.stop_button.configure(state="disabled")

    def close(self):
        if self.busy() and self.mode in ["run", "report"]:
            self.close_after_stop = True
            if self.mode == "run":
                self.request_stop()
            self.root.withdraw()
        else:
            if self.process is not None:
                self.process.terminate()  # Preview performs no training and can be cancelled immediately.
                self.process.wait(timeout=5)
            self._destroy_window()

    def _destroy_window(self):
        for callback in [self._poll_id, self._preview_id]:
            if callback:
                try:
                    self.root.after_cancel(callback)
                except tk.TclError:
                    pass
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Графическое окно эксперимента SΩI")
    parser.add_argument("--preset", choices=("experiment",)+PRESETS, default="experiment")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    root = tk.Tk()
    from .simple_gui import SimpleExperimentApp
    SimpleExperimentApp(root, preset=args.preset, config_path=args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
