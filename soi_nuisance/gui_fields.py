"""Russian parameter help and a JSON editor, independent of Tk and PyTorch."""
import copy
import json
import math
from pathlib import Path


HELP = {
    "rho_values": ("Внутренняя сетка связей", "Автоматическая серия значений rho для построения кривой по истинной O-информации."),
    "rho_subsystem_indices": ("Подсистемы серии", "Внутренние индексы redundancy-подсистем, у которых автоматически меняется rho."),
    "convergence": ("Правило сходимости", "Внутреннее правило остановки по плато validation loss; не использует test и истинную Omega."),
    "report_mode": ("Формат отчёта", "curve сохраняет основной график оценённой информации против истинной; full оставляет дополнительную диагностику."),
    "simple_settings": ("Основные настройки", "Пять пользовательских параметров простого эксперимента; остальные настройки фиксированы по исходникам статьи."),
    "methods": ("Методы", 'Оцениватели для сравнения: ["soi", "line_mine", "line_nwj"]. Каждый получает одни и те же данные.'),
    "benchmark": ("Тип зависимости U", "redundancy — избыточность; synergy — синергия; mixed — независимые подсистемы из mixed_subsystems."),
    "rho": ("Сила связи ρ", "Параметр исходного Gaussian-генератора. Его смысл различается у redundancy и synergy. Неположительно определённая ковариация отклоняется."),
    "mixed_subsystems": ("Подсистемы mixed", 'Список объектов {"benchmark":"redundancy","size":3,"rho":0.6}. Сумма size должна равняться N. Используется при benchmark=mixed.'),
    "num_blocks": ("Число блоков N", "Не менее 3. Один блок Xi содержит U и V; координаты V не считаются новыми блоками O-информации."),
    "u_dim": ("Размерность U в блоке", "Число Gaussian-координат в каждом блоке. Именно U задаёт зависимости и аналитическую O-информацию."),
    "v_dim": ("Размерность V в блоке", "Число независимых nuisance-координат. 0 отключает V. Полная размерность D=N·(u_dim+v_dim)."),
    "k_values": ("Значения k", "Список чисел узлов случайной кусочно-линейной плотности V. k=1 даёт треугольную плотность. Рост k не гарантирует роста трудности."),
    "normalization": ("Нормировка V", "population — стандартизация точными средним и дисперсией плотности; none — исходный масштаб носителя [0,k+1]."),
    "transformation": ("Преобразование U", 'Должно быть пустой строкой "": этот эксперимент не применяет нелинейных преобразований U.'),
    "train_sizes": ("Размеры обучающей выборки n", "Список чисел уникальных объектов. Выборки вложены: префиксы одного train pool. null включает геометрическую сетку ниже."),
    "n_start": ("Начальное n", "Первый размер геометрической сетки. Используется только при train_sizes=null."),
    "n_multiplier": ("Множитель сетки n", "Следующий размер выборки равен предыдущему, умноженному на это число. Используется при train_sizes=null."),
    "n_levels": ("Число уровней n", "Число размеров геометрической сетки. Используется при train_sizes=null."),
    "n_val": ("Объекты validation", "Размер независимой проверочной выборки для промежуточной диагностики. Она не выбирает лучший checkpoint."),
    "n_test": ("Объекты test", "Размер фиксированной независимой выборки для окончательного оценивания после обучения."),
    "distribution_seeds": ("Seeds внешних реализаций", "Список seeds независимых реализаций распределения. Доверительные интервалы строятся по этим внешним повторам."),
    "data_seeds": ("Seeds выборок", "Список seeds наборов объектов внутри каждой внешней реализации. Не меняет фиксированные высоты её плотностей V."),
    "training_seeds": ("Seeds обучения", "Список seeds инициализации сетей, порядка батчей и обучающих шумов. Это вложенные повторы, а не дополнительные внешние реализации."),
    "control_v_dim_zero": ("Gaussian-контроль без V", "true добавляет отдельные запуски v_dim=0. Этот контроль меняет размерность и отделён от сравнения по k."),
    "budget_mode": ("Режим бюджета", "paper — заданное число optimizer updates; equal_time — время обучения всего оценивателя, включая все его MI-критики."),
    "budget_multipliers": ("Множители бюджета", "Список множителей числа шагов или времени. Для каждого бюджета запускается отдельная модель."),
    "equal_time_seconds": ("Время equal_time, секунд", "Обучение одного полного оценивателя, до умножения на budget_multipliers. Validation, checkpoint и final test считаются отдельно."),
    "wall_time_seconds": ("Приоритетный бюджет времени", "Альтернативное имя equal_time_seconds. Если не null, имеет приоритет над equal_time_seconds. Используется в equal_time."),
    "max_run_seconds": ("Ограничитель обучения", "Максимум секунд обучения одного запуска; null отключает ограничитель. Достижение даёт статус timeout, а не успешное завершение."),
    "eval_batch_size": ("Батч оценивания", "Число объектов в evaluation-батче. Задаётся независимо от train batch и не меняется автоматически при нехватке памяти."),
    "precision": ("Точность сетей", "Поддерживается float32. Ковариации, моменты плотностей и чувствительные суммы вычисляются в float64."),
    "deterministic": ("Детерминированные операции", "true включает детерминированные алгоритмы PyTorch. Побитный повтор требует тех же версий и оборудования."),
    "device": ("Устройство", "auto выбирает CUDA при доступности; cpu выполняет вычисления на процессоре; cuda или cuda:0 — на видеокарте."),
    "cpu_threads": ("Потоки CPU", "Число потоков PyTorch на CPU. Влияет на скорость; не является числом одновременно обучаемых моделей."),
    "checkpoints_every": ("Период checkpoint, шагов", "Как часто сохранять last.pt с моделью, EMA, optimizer и RNG. 0 отключает промежуточные записи; финальная зависит от save_checkpoints."),
    "evaluation_every": ("Период validation, шагов", "Частота промежуточных оценок на validation. 0 отключает их. Test во время обучения не используется."),
    "log_every": ("Период текстового лога", "Как часто печатать loss и время. 0 отключает периодические строки; графический прогресс обновляется независимо."),
    "diagnostic_samples": ("Объекты диагностики SΩI", "Число validation-объектов для noise MSE по U/V, режимам и времени. 0 отключает эту диагностику."),
    "epsilon": ("Порог ошибки ε", "null отключает порог. Иначе отчёт ищет минимальное проверенное n с RMSE≤ε; это не доверительная гарантия."),
    "save_checkpoints": ("Сохранять checkpoint", "true сохраняет состояния для продолжения. false экономит место, но не позволяет продолжить незавершённое обучение."),
    "output_dir": ("Каталог результатов", "Путь для данных, checkpoint, CSV/JSONL и графиков. Относительный путь считается от корня проекта. Для существующих результатов включите resume."),
    "resume": ("Продолжить сохранённую работу", "true загружает последний checkpoint и пропускает готовые запуски. Параметры данных и моделей должны оставаться совместимыми."),
    "dry_run": ("Только план", "true рассчитывает число задач и размеры моделей без обучения. В окне для этого также есть кнопка «Оценить время»."),
    "divergence_threshold": ("Порог расходимости loss", "null отключает порог. Иначе |loss| выше порога останавливает запуск со статусом divergence. Clipping не применяется."),
    "name": ("Оптимизатор", "Adam — исходный оптимизатор эксперимента. Также поддерживается явно выбранный AdamW."),
    "lr": ("Скорость обучения", "Learning rate оптимизатора. В исходных настройках SΩI — 0.01, baseline — 0.001."),
    "betas": ("Коэффициенты Adam β", "Список [β1,β2] для скользящих моментов градиентов; исходные значения [0.9,0.999]."),
    "eps": ("Численный ε оптимизатора", "Добавка в знаменателе Adam для устойчивости. Это не порог ошибки O-информации epsilon."),
    "weight_decay": ("Регуляризация весов", "Коэффициент weight decay. В исходных настройках равен 0."),
    "batch_size": ("Обучающий батч", "Число X в одном optimizer update. Должно помещаться в каждую train_sizes; новые X сверх фиксированной выборки не генерируются."),
    "training_steps": ("Шаги обучения", 'Положительное число optimizer updates либо "paper": 195000 для SΩI; 40000/80000 baseline при N=6/10, иначе явное правило 8000·N.'),
    "width": ("Ширина сети", "null выбирает исходное правило по размерности. Для baseline это базовая ширина, затем умноженная на число блоков пары и ограниченная cap."),
    "time_dim": ("Ширина time embedding", "Размер временного представления SΩI. null выбирает ту же ширину, что у основной сети."),
    "dim_mults": ("Внутренние уровни MLP", "Исходный вызов SΩI использует []. Непустой список единиц добавляет уровни и меняет архитектуру."),
    "groups": ("Группы GroupNorm", "Число групп нормализации MLP. Ширина должна делиться на это число; исходное значение 8."),
    "weight_s_functions": ("Веса выбора режимов", "true чаще выбирает joint и conditional согласно исходным весам режимов; все режимы обучает одна сеть."),
    "use_ema": ("Использовать EMA сети", "true применяет экспоненциально усреднённые веса сети при оценивании SΩI."),
    "ema_decay": ("Коэффициент EMA", "Доля старых весов в скользящем среднем. Исходное значение 0.999."),
    "ema_timing": ("Момент обновления EMA", "before_step сохраняет поведение upstream; after_step — явно изменённый режим обновления после optimizer step."),
    "mc_iter": ("Monte Carlo повторения SΩI", "Сколько раз усреднить интеграл после обучения. Больше повторений увеличивает время оценивания; это не независимые повторы эксперимента."),
    "beta_min": ("Начальная β VP-SDE", "Начальная интенсивность линейного расписания β(t). Исходное значение 0.1."),
    "beta_max": ("Конечная β VP-SDE", "Конечная интенсивность β(t) при T=1. Исходное значение 20."),
    "t_epsilon": ("Порог importance sampling", "Ниже этого t плотность importance sampling постоянна. Исходное значение 0.001."),
    "importance_sampling": ("Importance sampling времени", "true выбирает времена зашумления по исходной importance-плотности; false — равномерно."),
    "inference_weighting": ("Вес интеграла SΩI", "upstream_constant сохраняет исходное приближение; exact корректирует область t<t_epsilon и может увеличить дисперсию."),
    "uniform_epsilon": ("Минимальное равномерное t", "Нижняя граница времени при выключенном importance sampling. Исходное значение 0.00001."),
    "width_cap": ("Предел ширины критика", "null выбирает исходный cap: 32 у MINE, 256 у NWJ. Ограничивает ширину каждого отдельного MI-критика."),
    "hidden_layers": ("Скрытые слои критика", "null сохраняет исходную архитектуру: один скрытый ReLU-слой у MINE, два у NWJ."),
    "pair_chunk": ("Пар за один чанк NWJ", "Максимум отрицательных пар в одном чанке. 0 использует все пары сразу. Чанки сохраняют исходный функционал."),
    "negative_sampling": ("Отрицательные пары MINE", "with_replacement сохраняет upstream torch.randint; permutation — отдельный режим с перестановкой. NWJ всегда использует все пары, включая диагональ."),
    "eval_aggregation": ("Агрегация MI на test", "paper_batch_mean — среднее bounds по батчам; global_moment — объединение математических ожиданий до логарифма. Это разные варианты оценивания."),
}

CHOICES = {
    "benchmark": ("redundancy", "synergy", "mixed"),
    "normalization": ("population", "none"), "budget_mode": ("paper", "equal_time"),
    "device": ("auto", "cpu", "cuda", "cuda:0"), "precision": ("float32",),
    "name": ("Adam", "AdamW"), "ema_timing": ("before_step", "after_step"),
    "inference_weighting": ("upstream_constant", "exact"),
    "negative_sampling": ("with_replacement", "permutation"),
    "eval_aggregation": ("paper_batch_mean", "global_moment"),
}


def read_raw_config(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if "config" in value and isinstance(value["config"], dict):
        value = value["config"]
    parent = value.pop("extends", None)
    if parent:
        base = read_raw_config(path.parent/parent)
        def merge(a, b):
            result = copy.deepcopy(a)
            for key, item in b.items():
                result[key] = merge(result[key], item) if isinstance(item, dict) and key in result else item
            return result
        return merge(base, value)
    return value


def iter_fields(config, prefix=""):
    for key, value in config.items():
        path = prefix+key
        if isinstance(value, dict):
            yield from iter_fields(value, path+".")
        else:
            yield path, value


def field_help(path):
    label, description = HELP[path.split(".")[-1]]
    if path == "method_params.soi.width":
        description = "null выбирает H=128 при D≤50, 192 при D≤100, иначе 256. По k ширина остаётся постоянной."
    return label, description


def get_value(config, path):
    value = config
    for part in path.split("."):
        value = value[part]
    return value


def set_value(config, path, value):
    parts = path.split(".")
    obj = config
    for part in parts[:-1]:
        obj = obj[part]
    obj[parts[-1]] = value


def editor_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def parse_value(text, previous, path):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        if isinstance(previous, str) or (path.endswith("training_steps") and text == "paper"):
            value = text
        else:
            raise ValueError("Введите JSON: число, [список], true/false или null.") from None
    if isinstance(previous, bool) and not isinstance(value, bool):
        raise ValueError("Для флага используйте true или false.")
    if isinstance(previous, int) and not isinstance(previous, bool):
        if type(value) is not int and not (path.endswith("training_steps") and value == "paper"):
            raise ValueError("Введите целое число.")
    if isinstance(previous, float) and (type(value) not in [int, float]):
        raise ValueError("Введите число.")
    if isinstance(previous, str) and not isinstance(value, str):
        raise ValueError("Введите строку.")
    if isinstance(previous, list) and not isinstance(value, list) and not (path == "train_sizes" and value is None):
        raise ValueError("Введите список в квадратных скобках.")
    def check(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("NaN и Infinity не являются допустимыми параметрами.")
        if isinstance(item, (list, dict)):
            for child in (item.values() if isinstance(item, dict) else item):
                check(child)
    check(value)
    return value
