"""Default interface: five fields and a fixed ten-point author-compatible series."""
from datetime import datetime
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from .gui import ExperimentApp, METHOD_NAMES, PHASES, ROOT
from .gui_fields import read_raw_config
from .gui_progress import format_duration
from .simple_experiment import DEFAULT_SETTINGS, build_experiment, settings_from_config

FIELDS = (
    ("u_dim", "dim U", "Число Gaussian-признаков в каждом блоке."),
    ("v_dim", "dim V", "Число независимых V-признаков в каждом блоке; 0 — без V."),
    ("n_samples", "Размер выборки", "Уникальные обучающие объекты; не менее 256."),
    ("k", "k плотности V", "Число случайных высот плотности; k=1 — треугольник."),
    ("max_steps", "Максимум шагов", "Обучение до плато validation loss или этого предела."),
)


class SimpleExperimentApp(ExperimentApp):
    def __init__(self, root, preset="experiment", config_path=None, auto_preview=True):
        super().__init__(root, preset=preset, config_path=config_path, auto_preview=auto_preview)
        root.title("SΩI — простой эксперимент")
        root.geometry("940x780")
        root.minsize(880, 720)

    def _build(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="SΩI · Line-MINE · Line-NWJ", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(outer, text="10 значений истинной информации → три кривые оценок и пунктир GT.",
                  font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 12))
        body = ttk.Frame(outer)
        body.pack(fill="x")
        inputs = ttk.Frame(body)
        inputs.pack(side="left", fill="x", expand=True)
        self.field_vars, self.field_entries = {}, []
        for index, (key, label, help_text) in enumerate(FIELDS):
            ttk.Label(inputs, text=label, font=("Segoe UI", 10, "bold")).grid(row=index*2, column=0, sticky="w", pady=(6, 0))
            var = tk.StringVar(value=str(DEFAULT_SETTINGS[key]))
            entry = ttk.Entry(inputs, textvariable=var, width=14)
            entry.grid(row=index*2, column=1, sticky="ew", padx=(10, 12), pady=(6, 0))
            ttk.Label(inputs, text=help_text, wraplength=420, foreground="#58616b").grid(
                row=index*2+1, column=0, columnspan=2, sticky="w", pady=(2, 6))
            self.field_vars[key] = var
            self.field_entries.append(entry)
        inputs.columnconfigure(1, weight=1)
        summary = ttk.LabelFrame(body, text="Программа делает автоматически", padding=12)
        summary.pack(side="left", fill="both", expand=True, padx=(14, 0))
        self.plan_text = tk.Text(summary, width=41, height=17, wrap="word", font=("Segoe UI", 10),
                                 state="disabled", relief="flat", background=self.root.cget("background"))
        self.plan_text.pack(fill="both", expand=True)
        tools = ttk.Frame(outer)
        tools.pack(fill="x", pady=(10, 5))
        self.load_button = ttk.Button(tools, text="Загрузить настройки…", command=self.choose_config)
        self.load_button.pack(side="left")
        self.save_button = ttk.Button(tools, text="Сохранить настройки…", command=self.save_config)
        self.save_button.pack(side="left", padx=7)
        self.resume_var = tk.BooleanVar()
        self.resume_check = ttk.Checkbutton(tools, text="Продолжить", variable=self.resume_var, command=self._toggle_resume)
        self.resume_check.pack(side="right")
        destination = ttk.Frame(outer)
        destination.pack(fill="x", pady=(3, 10))
        self.output_var = tk.StringVar()
        self.directory_button = ttk.Button(destination, text="Папка результатов…", command=self.choose_output)
        self.directory_button.pack(side="left")
        ttk.Label(destination, textvariable=self.output_var, wraplength=650, foreground="#58616b").pack(side="left", padx=8)
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(2, 10))
        self.start_button = ttk.Button(actions, text="Запустить  (F5)", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Остановить", command=self.request_stop, state="disabled")
        self.stop_button.pack(side="left", padx=7)
        self.preview_button = ttk.Button(actions, text="Оценить время", command=self.preview)
        self.preview_button.pack(side="left")
        self.pdf_button = ttk.Button(actions, text="Открыть PDF", command=self.open_pdf)
        self.pdf_button.pack(side="right")
        self.rebuild_button = ttk.Button(actions, text="Обновить PDF", command=self.rebuild_report)
        self.rebuild_button.pack(side="right", padx=7)
        self.results_button = ttk.Button(actions, text="Результаты", command=self.open_results)
        self.results_button.pack(side="right")
        self.status_var = tk.StringVar(value="Готово к запуску")
        ttk.Label(outer, textvariable=self.status_var, wraplength=900).pack(anchor="w")
        self.job_var = tk.StringVar(value="30 обучений: 10 точек × 3 метода")
        ttk.Label(outer, textvariable=self.job_var).pack(anchor="w", pady=(6, 0))
        self.overall_bar = ttk.Progressbar(outer, maximum=100)
        self.overall_bar.pack(fill="x", pady=(3, 5))
        self.step_var = tk.StringVar(value="Остановка: сходимость validation loss или максимум шагов")
        ttk.Label(outer, textvariable=self.step_var).pack(anchor="w")
        self.step_bar = ttk.Progressbar(outer, maximum=100)
        self.step_bar.pack(fill="x", pady=(3, 6))
        self.time_var = tk.StringVar(value="Ожидаемое время уточняется")
        ttk.Label(outer, textvariable=self.time_var).pack(anchor="w", pady=(0, 7))
        self.log = tk.Text(outer, height=7, wrap="word", font=("Consolas", 9), state="disabled")
        self.log.pack(fill="both", expand=True)

    def _controls(self, busy):
        for widget in self.field_entries+[self.start_button, self.preview_button, self.load_button,
                self.save_button, self.directory_button, self.resume_check, self.rebuild_button]:
            widget.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy and self.mode == "run" else "disabled")

    def _settings(self):
        try:
            return {key: int(var.get().strip()) for key, var in self.field_vars.items()}
        except ValueError:
            raise ValueError("Все пять значений должны быть целыми числами.") from None

    def _new_output(self):
        return "results/experiment_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:4]

    def load_preset(self, preset):
        self.load_file(ROOT/"configs"/(preset+".json"), fresh_output=True)

    def load_file(self, path, fresh_output=False):
        try:
            raw = read_raw_config(path)
            settings = settings_from_config(raw)
            output = self._new_output() if fresh_output or "output_dir" not in raw else raw["output_dir"]
            config = build_experiment(settings, output, raw.get("resume", False) if not fresh_output else False)
            for key, value in settings.items():
                self.field_vars[key].set(str(value))
            self.config = config
            self.output_var.set(output)
            self.resume_var.set(config["resume"])
            self.plan = self.tracker = None
            self._set_plan_text("Авторский mixed benchmark: 10 блоков.\n\n"
                "10 связей ρ выбираются автоматически; истинная Ω вычисляется аналитически.\n\n"
                "SΩI: Adam, lr=0.01, batch=256.\nLine-MINE / NWJ: Adam, lr=0.001, batch=64.\n\n"
                "Архитектуры и остальные настройки фиксированы по исходникам. Доверительных интервалов нет.")
            self.status_var.set("Готово: измените пять значений и нажмите «Запустить».")
        except Exception as exc:
            messagebox.showerror("Настройки", str(exc), parent=self.root)

    def apply_edit(self, show_error=True):
        if self.busy():
            return True
        try:
            self.config = build_experiment(self._settings(), self.output_var.get(), self.resume_var.get())
            return True
        except Exception as exc:
            if show_error:
                messagebox.showerror("Проверьте настройки", str(exc), parent=self.root)
            return False

    def save_config(self):
        if self.busy() or not self.apply_edit():
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Сохранить пять настроек",
                                           defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                from pathlib import Path
                Path(path).write_text(json.dumps(self._settings(), indent=2), encoding="utf-8")
                self.status_var.set("Сохранены только пять основных настроек.")
            except OSError as exc:
                messagebox.showerror("Сохранение", str(exc), parent=self.root)

    def _render_fields(self):
        self.output_var.set(self.config["output_dir"])

    def _toggle_resume(self):
        self.config["resume"] = self.resume_var.get()
        self.tracker = None

    def _event(self, event):
        super()._event(event)
        kind = event["event"]
        if kind == "plan":
            points = event["plan"]["curve_points"]
            truths = [p["omega_true"] for p in points]
            self._set_plan_text(f"10 значений истинной Ω:\nот {min(truths):.4g} до {max(truths):.4g} nats.\n\n"
                f"30 обучений: 10 точек × 3 метода.\nУстройство: {event['device']}.\n\n"
                "По полному лимиту ≈ "+format_duration(self.tracker.remaining())+".\n"
                "При сходимости завершится раньше.\n\n"
                "Один PDF: линейный вид, полный диапазон и сравнение ошибок методов с GT.")
        elif "job_index" in event and self.tracker:
            task = self.tracker.tasks[event["job_index"]]
            job = task["job"]
            self._job_title = f"Точка {job.get('point_index', 0)+1}/10, Ω={task['omega_true']:.4g} · {METHOD_NAMES[job['method']]}"
            if kind not in ["job_finished"] and not (self.stop_path and self.stop_path.exists()):
                self.status_var.set(PHASES.get(kind, "Обучение")+" — "+self._job_title)
        if kind == "convergence":
            self._append_log(f"Validation loss: {event['validation_loss']:.6g}; плато {event['stale']}/8"+
                             (" — сходимость" if event["converged"] else ""))
