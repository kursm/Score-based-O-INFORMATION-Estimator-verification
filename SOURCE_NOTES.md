# Источники и решения реализации

Основное окно и configs/experiment.json содержат пять настроек: dim U, dim V,
размер обучающей выборки, k, предел шагов. Конфигурация разворачивается автоматически
в simple_experiment.py. Архитектуры и параметры Adam сохранены по исходникам;
критерий плато validation loss добавлен в проекте и не выдаётся за авторский протокол.

По умолчанию выбран mixed benchmark N=10 из synthetic_dataset.py:
две redundancy-подсистемы размеров 3 и 4 и одна synergy размера 3.
У redundancy одновременно меняется ρ, у synergy остаётся авторское ρ=0.7.
Десять значений ρ выбраны из vendor/soi/jobs/o_inf.sh:
0.1, 0.2, 0.25, 0.3, 0.35, 0.375, 0.4, 0.45, 0.5, 0.6.
Истинная Ω вычисляется из ковариации каждой точки. У dim U=1 она проходит
от −0.1921960787 до 0.3202105188 nats. Это десять разных зависимостей,
а не повторы оценки одного значения. Автоматического поиска проигрыша SΩI нет.

Используется один фиксированный seed=77 из авторского job script.
В каждой точке все методы получают побитово одинаковые train/val/test.
При смене ρ V и её плотности остаются неизменными, U меняется по covariance.
Число блоков N фиксировано; изменяемый dim U — размерность внутри каждого блока.
Общий предел updates задаётся пользователем одинаково для трёх методов;
исходные индивидуальные бюджеты статьи этим режимом не воспроизводятся.

График использует стиль Figure 1–3 и vendor/soi/src/vis/curves.ipynb:
синий/оранжевый/красный цвета и пунктирный GT. Для различимости наложений
используются звезда / пустой круг / крестик и разные штрихи SΩI/MINE/NWJ.
С 1 ноября 2026 один figures.pdf содержит линейный вид Estimated vs true,
полный диапазон symlog и страницу ошибок с MAE/RMSE и победами в точках.
Выбросы не изменяются: в линейном виде они помечены на границе, а полный вид
показывает все конечные значения кривых. Повторы seeds усредняются с равными
весами внешних распределений. Метрики используют одинаковые точки и seeds
всех сравниваемых методов; пропуски и сбои не превращаются в победы.
Новый инструмент repair_report.py восстанавливает только оформление и сравнение
по сохранённым числам и сохраняет резервную копию прежнего отчёта.
Доверительные интервалы и bootstrap отключены во всех режимах отчёта.
Ранее созданные PDF не меняются автоматически. Обозначения d (размерность блока)
и D (полная) разделены явно в подробных отчётах.

Проверены статья SΩI, Appendix C.1–C.3, Table 1 и официальный код,
а также исходная реализация TC-Line. SHA архивов и коммитов находятся в
vendor/SOURCES.json, SHA256 отдельных использованных файлов — vendor/FILE_HASHES.json.
Снимки источников сохранены локально и используются в parity-тестах.

- SΩI: Mustapha Bounoua, Giulio Franzese, Pietro Michiardi,
  [статья v3](https://arxiv.org/html/2402.05667v3),
  [PDF](https://arxiv.org/pdf/2402.05667).
- [soi](https://github.com/MustaphaBounoua/soi/tree/12952f58136454f813dc2887db220876823e053a),
  commit 12952f58136454f813dc2887db220876823e053a.
- Ke Bai, Pengyu Cheng, Weituo Hao, Ricardo Henao, Larry Carin:
  [Estimating Total Correlation with Mutual Information Estimators](https://proceedings.mlr.press/v206/bai23a.html).
- [TC-estimation](https://github.com/Linear95/TC-estimation/tree/8ebe62e46ff2d72326320ec2840cd677ee624de7),
  commit 8ebe62e46ff2d72326320ec2840cd677ee624de7. Это реально скачанный SHA master,
  а не предположение о состоянии ветки.

## Данные и ground truth

vendor/soi/src/benchmark/synthetic_dataset.py:
Task_redundant.build_cov_pure_redundancy_sigma_normalized — единицы на диагонали,
rho на остальных местах; rho=1/(1+sigma²).
Task_synergy.build_cov_pure_synergy_sigma_normalized — диагональ 1,
cov[0,1]=1/sqrt(N-1), cov[1,j]=rho/sqrt(N-1) для j>=2, остальные связи 0;
rho=1/sqrt(1+sigma²). Это исходный смысл rho, без замены на equicorrelation.
Task_combination и combined_cov соединяют независимые подсистемы.
eye и sample_cov разворачивают скалярную N×N covariance как kron(cov,I_du)
в порядке «блок, координата». entropy/tc/dtc/o_inf умножают скалярный ответ на du.

Наш data.py сохраняет эти covariance без скрытого jitter и без nonlinear transformation.
Ground truth — Cholesky logdet полной блочной U-covariance, nats.
Тесты сверяют covariance и три информационные меры непосредственно с извлечёнными
AST функциями/классами upstream, используя SciPy для исходного entropy.
Блочный truth также проверяется при добавлении независимой Gaussian nuisance-covariance.

Стандартные settings upstream — [3], [3,3], [3,3,4]. Простой режим использует
последнее разбиение с указанными выше типами подсистем. Внутренние режимы
redundancy/synergy создают одну подсистему размера N; mixed_subsystems позволяет
явно задавать разбиение и ρ. Особый gradient-settings из upstream не подставляется.

Новый генератор V реализован по пользовательскому промпту: фиксированные абсциссы,
случайные высоты, точный непрерывный sampler как смесь треугольников, аналитические
моменты полиномов и population standardization. Это новый эксперимент, которого нет
в опубликованных источниках.

## SΩI: точный вызов сети и режимы

vendor/soi/src/libs/soi.py: SOI.__init__ вызывает
UnetMLP_simple(dim=sum(sizes), init_dim=hidden_dim, dim_mults=[],
time_dim=hidden_dim, nb_var=N).
В soi_nuisance/_upstream_mlp.py находится побайтовая копия исходного mlp.py.
Основной preset сохраняет пустой dim_mults. Следовательно, downs/ups пусты,
в forward остаются входная проекция, time MLP, final_res_block со skip-конкатенацией
и финальная GroupNorm/SiLU/проекция. mid_block1 зарегистрирован, но его вызов
закомментирован upstream; его параметры сохранены и учтены. Это важное отличие
от предположения, что default dim_mults=(1,1) действительно используется.
Time conditioning — Linear(N,H), GELU, Linear(H,H), а не sinusoidal embedding.
Выходная проекция инициализируется нулями.

calculate_hidden_dim: H=128 при D<=50, 192 при 50<D<=100, 256 при D>100;
time_dim=H. При D=100 выбирается 192, следуя коду. Числа параметров в Table 1
не гарантированно совпадают с фактическим вызовом этого коммита. Runner записывает
реально зарегистрированные параметры, не копирует цифры из таблицы.
Непустой dim_mults можно задать явно; это уже изменённая архитектура.

score_forward строит tau: diffused=t, conditioned=0, отсутствующие=1.
Режимы обрабатывает одна сеть; маски охватывают Xi целиком.
SDE.get_masks_training и weight_masks: joint повторяется N раз;
marginal каждого i — один раз; conditional каждого i — 1+floor((N-1)/2).
Мы строим только 2N+1 нужных масок, без перебора 3^N; вероятности сохранены.
Порядок масок детерминирован вместо np.random.shuffle; это не меняет распределение.
train_step использует независимый режим для каждой строки батча. Неактивные
координаты не входят в loss. Вес regression loss = 2-(число активных координат)/D;
сумма по координатам и среднее по строкам, как в исходнике.

VP-SDE: beta(t)=.1+19.9t, alpha(t)=exp(-.5∫beta),
variance(t)=1-exp(-∫beta), T=1. Аддитивный heat noise из теоретической части
не используется. Все режимы inference используют один x_t и одни шумы целевых
координат; другие marginal-блоки заменяются независимым Gaussian noise.

importance.py задаёт q(t) пропорционально beta/variance при t>.001 и постоянную
плотность ниже .001; реализована та же inverse CDF. Для точности использованы
expm1, softplus, float64 и рационализированная квадратная формула.
Основной inference_weighting=upstream_constant использует Z/2 умножить на
квадраты разностей предсказанного noise. Это соответствует get_tc/get_dtc.
Из-за floor q при t<.001 постоянный множитель — приближение к точному VP integral;
оно сохранено явно, а не объявлено математически точным.
Опциональный inference_weighting=exact добавляет beta/variance/r(.001) при t<.001;
это меняет оцениватель и может увеличивать дисперсию.
В uniform режиме учтена длина интервала 1-t_epsilon; upstream этот множитель
опускает. Настраиваемые beta используются и в Z; upstream get_normalizing_constant
hardcodes .1/20/.001. Значения по умолчанию сохранены.

TC и DTC оцениваются исходными квадратами разностей joint/marginal и
joint/conditional, с накоплением по координатам в float64. Чанки взвешены по числу
объектов; upstream compute_o_inf_batch усреднял батчи одинаково.
Различия noise/score знака не меняют квадратов; сеть предсказывает positive noise,
как SDE.train_step. Diagnostics не поступают на вход сети.

SOI.configure_optimizers — torch.optim.Adam, lr=.01 из jobs/o_inf.sh;
betas=(.9,.999), eps=1e-8, weight_decay=0 — defaults Adam.
EMA decay=.999; util.EMA, SOI.on_before_backward обновляют EMA ДО текущего
optimizer step. Основной ema_timing=before_step сохраняет это запаздывание.
after_step — отдельная явная опция.
Paper budget 195000 соответствует 500*floor(100000/256) из job script.
Он фиксирован по updates при sweep n.

## Baseline: архитектуры и функционалы

Главный источник сборки — vendor/soi/src/baseline/baseline.py O_Estimator.__init__.
Размер блока d=du+dv; base hidden=8 при d=1, 16 при 1<d<=10, иначе20.
S-критики: min(base*(N-1), cap); TC-критик для prefix длины p:
min(base*p, cap). Cap=32 для MINE, 256 для NWJ.
Сначала инициализируются N S-критиков, затем N-1 TC-критиков.
Одинаковый математический MI у крайних термов всё равно имеет независимые веса.
Никакой переориентации пар или weight sharing.
TC-пары — prefix X[:i+1] и X[i+1]; S-пары — X[-i] и X[i].
S=TC+DTC; Omega=2TC-S. Каждая MI-оценка сохраняется отдельно.

vendor/soi/src/baseline/TC_estimation/mi_estimators.py:
MINE — Linear, ReLU, Linear (ОДИН скрытый слой).
NWJ — Linear, ReLU, Linear, ReLU, Linear (ДВА скрытых слоя).
В исходном vendor/tc_estimation/mi_estimators.py NWJ имеет ОДИН скрытый слой.
Для сравнения SΩI используется именно fork SΩI. Общая фраза статьи
про три linear layers не заменяет различающиеся фактические реализации.

MINE: E_joint[f]-log E_negative[exp(f)]. Исходный комментарий говорит shuffle,
но реально вызывается torch.randint(B,(B,)), то есть индексы С ВОЗВРАЩЕНИЕМ.
Это не случайная перестановка! Совпадения с собственным индексом разрешены,
повторение одного y несколько раз тоже. Основной negative_sampling=with_replacement
сохраняет исходный закон. permutation — отдельная опция, тоже допускающая fixed points.
Не добавлены EMA знаменателя, clipping critic/gradient и другие цели.

NWJ: E_joint[f]-E_all_pairs[exp(f-1)], ровно B² упорядоченных пар.
Включена диагональ; это конечновыборочный эффект исходной реализации.
logsumexp аккумулируется в float64. Pair chunks и recomputation checkpointing
сохраняют тот же функционал и его градиент; тестируются и параметры, и входы.
Реальное exp всё равно может переполниться при divergence; это сохраняется как сбой.
MINE также переведён с log(exp().mean()) на logsumexp без изменения функционала.
Очистка CUDA cache на каждом forward upstream удалена: она не входит в формулу
и резко замедляет малые задачи.

baseline.training_step усредняет 2N-1 отрицательных bounds и делает один Adam step
для полного оценивателя. lr=.001, betas/eps/decay — те же defaults Adam.
jobs/baseline.sh: batch64, 50 эпох по100k дают ~78100 updates независимо от N.
Appendix C.3 задаёт 40k для N6 и80k для N10: paper preset следует этим явно указанным
индивидуальным бюджетам, а не округляет job script. Для иных N правило8000*N —
наше явно подписанное расширение, не настройка статьи.

baseline.forward усредняет оценки по evaluation-батчам без весов.
paper_batch_mean воспроизводит это, включая маленький последний батч.
global_moment — отдельный вариант, описанный в README. Eval batch независим от
train batch; разбиение не меняется автоматически при OOM.

## Исправления протокола и ограничения

Lightning и его logger не нужны; используется компактный PyTorch loop.
RNG для evaluation отдельны, test не используется в обучении. Статья отбрасывала
diverged baseline; здесь сбои остаются в таблицах, как требует пользователь.
Данные сохраняются целиком и больше не генерируются онлайн. Finite-проверки
останавливают запуск, но не меняют objective и не клипуют значения.
При NaN сохраняется диагностическое состояние и last good checkpoint.
Число параметров baseline — сумма всех критиков; EMA не считается второй
обучаемой сетью. Результаты малых бюджетов не доказывают сходимость.

Сохранены локальные PDF обеих статей. HTML/PDF и исходные snapshots дают возможность
повторной проверки без доступа к сети. Тесты извлекают необходимые определения из
оригиналов, не устанавливая устаревшие Lightning/sklearn зависимости.
TC notebook просмотрен для проверки демонстрационного обучения и line path;
настройки основной сборки взяты из SΩI fork, не из notebook.

## Атрибуция и лицензирование

Исходники принадлежат авторам соответствующих репозиториев. Оригинальные заголовки
и комментарии в snapshots и копии mlp.py сохранены. В двух скачанных деревьях
нет файлов LICENSE/COPYING и явной лицензии в README; наличие permissive-лицензии
не утверждается. Архивы сохранены для локальной исследовательской проверки;
перед публичным распространением перенесённого кода следует установить права у
авторов. Эта заметка не назначает новую лицензию исходным файлам.
MLP целиком взят у Mustapha Bounoua и соавторов; importance sampling в upstream
атрибутирован CW-Huang/sdeflow-light. Новый runner и генератор не меняют авторство
исходных компонентов.

## Точные координаты исходных определений

- [vendor/soi/src/benchmark/synthetic_dataset.py:20](vendor/soi/src/benchmark/synthetic_dataset.py#L20) — get_dataloader.
- [vendor/soi/src/benchmark/synthetic_dataset.py:36](vendor/soi/src/benchmark/synthetic_dataset.py#L36) — get_task.
- [vendor/soi/src/benchmark/synthetic_dataset.py:89](vendor/soi/src/benchmark/synthetic_dataset.py#L89) — eye.
- [vendor/soi/src/benchmark/synthetic_dataset.py:103](vendor/soi/src/benchmark/synthetic_dataset.py#L103) — entropy.
- [vendor/soi/src/benchmark/synthetic_dataset.py:118](vendor/soi/src/benchmark/synthetic_dataset.py#L118) — get_cov_minus_i.
- [vendor/soi/src/benchmark/synthetic_dataset.py:142](vendor/soi/src/benchmark/synthetic_dataset.py#L142) — tc.
- [vendor/soi/src/benchmark/synthetic_dataset.py:150](vendor/soi/src/benchmark/synthetic_dataset.py#L150) — o_inf.
- [vendor/soi/src/benchmark/synthetic_dataset.py:161](vendor/soi/src/benchmark/synthetic_dataset.py#L161) — dtc.
- [vendor/soi/src/benchmark/synthetic_dataset.py:172](vendor/soi/src/benchmark/synthetic_dataset.py#L172) — s_inf.
- [vendor/soi/src/benchmark/synthetic_dataset.py:183](vendor/soi/src/benchmark/synthetic_dataset.py#L183) — fill_cov.
- [vendor/soi/src/benchmark/synthetic_dataset.py:193](vendor/soi/src/benchmark/synthetic_dataset.py#L193) — combined_cov.
- [vendor/soi/src/benchmark/synthetic_dataset.py:206](vendor/soi/src/benchmark/synthetic_dataset.py#L206) — Task.
- [vendor/soi/src/benchmark/synthetic_dataset.py:348](vendor/soi/src/benchmark/synthetic_dataset.py#L348) — Task_redundant.
- [vendor/soi/src/benchmark/synthetic_dataset.py:380](vendor/soi/src/benchmark/synthetic_dataset.py#L380) — Task_synergy.
- [vendor/soi/src/benchmark/synthetic_dataset.py:419](vendor/soi/src/benchmark/synthetic_dataset.py#L419) — SynthetitcDataset.
- [vendor/soi/src/benchmark/synthetic_dataset.py:437](vendor/soi/src/benchmark/synthetic_dataset.py#L437) — Task_combination.
- [vendor/soi/src/libs/soi.py:12](vendor/soi/src/libs/soi.py#L12) — SOI.
- [vendor/soi/src/libs/SDE.py:9](vendor/soi/src/libs/SDE.py#L9) — VP_SDE.
- [vendor/soi/src/libs/info_measures.py:9](vendor/soi/src/libs/info_measures.py#L9) — compute_entropy.
- [vendor/soi/src/libs/info_measures.py:49](vendor/soi/src/libs/info_measures.py#L49) — get_tc.
- [vendor/soi/src/libs/info_measures.py:81](vendor/soi/src/libs/info_measures.py#L81) — get_dtc.
- [vendor/soi/src/libs/info_measures.py:113](vendor/soi/src/libs/info_measures.py#L113) — compute_all_measures.
- [vendor/soi/src/libs/info_measures.py:126](vendor/soi/src/libs/info_measures.py#L126) — compute_grad_o_inf.
- [vendor/soi/src/libs/importance.py:11](vendor/soi/src/libs/importance.py#L11) — sample_vp_truncated_q.
- [vendor/soi/src/libs/importance.py:20](vendor/soi/src/libs/importance.py#L20) — get_normalizing_constant.
- [vendor/soi/src/libs/importance.py:30](vendor/soi/src/libs/importance.py#L30) — VariancePreservingTruncatedSampling.
- [vendor/soi/src/libs/importance.py:116](vendor/soi/src/libs/importance.py#L116) — log_normal.
- [vendor/soi/src/libs/importance.py:120](vendor/soi/src/libs/importance.py#L120) — sample_rademacher.
- [vendor/soi/src/libs/importance.py:123](vendor/soi/src/libs/importance.py#L123) — sample_gaussian.
- [vendor/soi/src/libs/importance.py:126](vendor/soi/src/libs/importance.py#L126) — sample_v.
- [vendor/soi/src/libs/util.py:7](vendor/soi/src/libs/util.py#L7) — EMA.
- [vendor/soi/src/libs/util.py:27](vendor/soi/src/libs/util.py#L27) — pop_elem_i.
- [vendor/soi/src/libs/util.py:33](vendor/soi/src/libs/util.py#L33) — deconcat.
- [vendor/soi/src/libs/util.py:39](vendor/soi/src/libs/util.py#L39) — concat_vect.
- [vendor/soi/src/libs/util.py:44](vendor/soi/src/libs/util.py#L44) — unsequeeze_dict.
- [vendor/soi/src/libs/util.py:51](vendor/soi/src/libs/util.py#L51) — cond_x_data.
- [vendor/soi/src/libs/util.py:60](vendor/soi/src/libs/util.py#L60) — marginalize_data.
- [vendor/soi/src/libs/util.py:71](vendor/soi/src/libs/util.py#L71) — marginalize_one_var.
- [vendor/soi/src/libs/util.py:82](vendor/soi/src/libs/util.py#L82) — minus_x_data.
- [vendor/soi/src/libs/util.py:93](vendor/soi/src/libs/util.py#L93) — expand_mask.
- [vendor/soi/src/libs/util.py:99](vendor/soi/src/libs/util.py#L99) — get_samples.
- [vendor/soi/src/models/mlp.py:10](vendor/soi/src/models/mlp.py#L10) — exists.
- [vendor/soi/src/models/mlp.py:14](vendor/soi/src/models/mlp.py#L14) — default.
- [vendor/soi/src/models/mlp.py:20](vendor/soi/src/models/mlp.py#L20) — identity.
- [vendor/soi/src/models/mlp.py:24](vendor/soi/src/models/mlp.py#L24) — num_to_groups.
- [vendor/soi/src/models/mlp.py:35](vendor/soi/src/models/mlp.py#L35) — Residual.
- [vendor/soi/src/models/mlp.py:44](vendor/soi/src/models/mlp.py#L44) — Upsample.
- [vendor/soi/src/models/mlp.py:51](vendor/soi/src/models/mlp.py#L51) — Downsample.
- [vendor/soi/src/models/mlp.py:55](vendor/soi/src/models/mlp.py#L55) — Residual.
- [vendor/soi/src/models/mlp.py:64](vendor/soi/src/models/mlp.py#L64) — Block.
- [vendor/soi/src/models/mlp.py:90](vendor/soi/src/models/mlp.py#L90) — ResnetBlock.
- [vendor/soi/src/models/mlp.py:123](vendor/soi/src/models/mlp.py#L123) — UnetMLP_simple.
- [vendor/soi/src/baseline/baseline.py:29](vendor/soi/src/baseline/baseline.py#L29) — O_Estimator.
- [vendor/soi/src/baseline/TC_estimation/mi_estimators.py:56](vendor/soi/src/baseline/TC_estimation/mi_estimators.py#L56) — MINE.
- [vendor/soi/src/baseline/TC_estimation/mi_estimators.py:87](vendor/soi/src/baseline/TC_estimation/mi_estimators.py#L87) — NWJ.
- [vendor/soi/experiments/config.py:5](vendor/soi/experiments/config.py#L5) — get_default_config.
- [vendor/soi/experiments/config.py:9](vendor/soi/experiments/config.py#L9) — get_config.
- [vendor/soi/experiments/config.py:73](vendor/soi/experiments/config.py#L73) — get_config_baseline.
- [vendor/tc_estimation/mi_estimators.py:53](vendor/tc_estimation/mi_estimators.py#L53) — MINE.
- [vendor/tc_estimation/mi_estimators.py:82](vendor/tc_estimation/mi_estimators.py#L82) — NWJ.
- [vendor/tc_estimation/tc_estimators.py:16](vendor/tc_estimation/tc_estimators.py#L16) — TCLineEstimator.
- [vendor/tc_estimation/tc_estimators.py:67](vendor/tc_estimation/tc_estimators.py#L67) — TCTreeEstimator.
- [vendor/soi/src/libs/soi.py:37](vendor/soi/src/libs/soi.py#L37) — self.score = UnetMLP_simple(dim=np.sum(self.sizes), init_dim=hidden_dim, dim_mults=[],.
- [vendor/soi/src/libs/soi.py:88](vendor/soi/src/libs/soi.py#L88) — def on_before_backward(self, loss: torch.Tensor) -> None:.
- [vendor/soi/src/libs/soi.py:145](vendor/soi/src/libs/soi.py#L145) — optimizer = torch.optim.Adam(self.score.parameters(), lr=self.args.lr).
- [vendor/soi/src/baseline/baseline.py:45](vendor/soi/src/baseline/baseline.py#L45) — hidden_size = 8.
- [vendor/soi/src/baseline/baseline.py:47](vendor/soi/src/baseline/baseline.py#L47) — hidden_size = 16.
- [vendor/soi/src/baseline/baseline.py:49](vendor/soi/src/baseline/baseline.py#L49) — hidden_size = 20.
- [vendor/soi/src/baseline/baseline.py:60](vendor/soi/src/baseline/baseline.py#L60) — max_dim= 32.
- [vendor/soi/src/baseline/baseline.py:135](vendor/soi/src/baseline/baseline.py#L135) — optimizer = torch.optim.Adam(self.parameters(), lr= self.args.lr  ).

## Дополнительные численные детали

Uniform epsilon задаётся отдельно: uniform_epsilon=1e-5, как в upstream train_step
и compute_o_inf, а importance floor t_epsilon=.001.
Равномерное число для времени генерируется в float64 и ограничивается снизу
machine epsilon float64, чтобы исключить точную границу t=0 и деление на ноль
в опциональном exact importance correction. Плотность времени и формулы inverse CDF
сохранены; побитное совпадение последовательности случайных чисел upstream не обещается.
Full-baseline parity test дополнительно сверяет сборку, общий loss и все градиенты;
training negatives выбираются в исходном порядке TC, затем S.
divergence_threshold по умолчанию отключён; если задан, это явное правило остановки
по абсолютному loss, без clipping и изменения цели.

EMA также сохраняет обход state_dict из util.EMA: выходная Linear зарегистрирована
как proj и final_lin.2, поэтому исходник обновляет её EMA дважды за вызов.
Это поведение намеренно сохранено и проверяется отдельным побитным parity-тестом;
остальные параметры получают одно EMA-обновление.

Inverse CDF проверена также при вычислении исходной формулы в float64: это отделяет
совпадение алгебры от cancellation исходного float32 при вычислении r(.001).
На контрольных u отличие от float32 upstream не превосходит 3e-5 по t;
реализация не имитирует эту численную погрешность.
Resume-тест параметризован CPU/CUDA и сравнивает модель, optimizer, EMA, train RNG,
порядок батчей и итоговую оценку с непрерывным обучением.

RNG обучения использует namespace (distribution_seed, data_seed, training_seed, method).
Внешние и вложенные реализации имеют независимые инициализации и шумы;
по k, n и бюджетам потоки остаются парными. Порядок train-объектов общий между методами.
