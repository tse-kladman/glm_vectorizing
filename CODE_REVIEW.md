# Код-ревью `glmbench` (2026-10-18)

## Итог

Ядро вычислений корректно: kNN исключает «себя» по индексу (не по позиции), группы
leave-genome-out и RC-пары выровнены, слой ищется в метрических таблицах по позиции,
сиды детерминированы, k-medoids, Wilcoxon, Holm и BH реализованы верно. Ошибок, которые
меняли бы посчитанные на сервере числа (`results/metrics/**`), **не найдено**, и
**пересчёт на сервере не требуется**.

Найдены и исправлены ошибки уровня фигур и устойчивости:

1. **Номер слоя ViraLM расходился на единицу между фигурами.** В Fig. 1/2, в
   `layerwise_reference` и `box_fig1` показывалась позиция в бенчмарке (0..11), а в
   Fig. 4–6 — метка блока (1..12). Один и тот же слой выглядел как «L2» на Fig. 1 и
   как «3» на Fig. 4, а кривая ViraLM на Fig. 2 была сдвинута на один блок влево
   относительно DNABERT.
2. **На доп. фигурах `newvirus_p5/viralm_*` пунктир «выбранного слоя» стоял на соседнем
   слое**: позицию из `selection.json` рисовали на оси меток.
3. **Holm/BH давали NaN для всего семейства**, если хотя бы одно сырое p было NaN.
4. **Этап `figures` строил Fig. 4–6 раньше, чем пересчитывал таблицу тестов.** Звёздочки
   на них могли браться из устаревшего `pairwise_model_tests.csv`.
5. Мелочи: `np.argmax` по слоям с NaN, падение фигур на частично посчитанных моделях
   реестра, мёртвый и дублированный код, устаревшие docstring и README.

## Таблица находок

| # | Важность | Файл:строка | Проблема | Статус |
|---|---|---|---|---|
| 1 | высокая (неверная подпись в статье) | `glmbench/metrics/reference.py:53` (`layer_label`), `figures/main.py:71,147`, `figures/statistics.py:84`, `figures/supplementary.py:243` | В Fig. 1/2, `layerwise_reference`, `box_fig1` и `evo2_early_layer` ViraLM показывался позицией (0..11), а в Fig. 4–6 — меткой (1..12). | исправлено |
| 2 | средняя | `figures/supplementary.py:37` (`_chosen_label`) | На графиках `newvirus_p5` ViraLM пунктир стоял на позиции 2 по оси меток, то есть на блоке 2 вместо блока 3. | исправлено |
| 3 | средняя | `glmbench/statistics.py:35` (`_finite_only`), `:103` | Через `np.minimum/maximum.accumulate` NaN распространялся на всё семейство Holm/BH и входил в размер семейства. При отсутствии общих образцов `compare_pair` возвращал p = 1 («нет различия»). | исправлено |
| 4 | средняя | `glmbench/cli.py:323` | Этап `figures` строил main (звёзды Fig. 4–6 из таблицы тестов) до `statistics.build`, поэтому звёзды могли быть устаревшими. | исправлено |
| 5 | низкая | `glmbench/selection.py:72` | `choose_layer` использовал `np.argmax`, который возвращает индекс первого NaN; `reference.summary` использует `nanargmax`. | исправлено |
| 6 | низкая | `figures/main.py:51` (`_has_table`) | Модель реестра (minhash1k, kmer_canon) без части таблиц роняла `build_fig5/6` с `FileNotFoundError`. | исправлено (модель пропускается и называется в логе, как в статистике) |
| 7 | низкая (защита) | `representations/transformers.py:181`; `envs_for_glms/ViraLM/viralm_hook.py:361` | Хук пишет `np.mean(states, axis=0)` на батч под именем первого окна. При `batch_size > 1` разные фрагменты молча усреднились бы. Сейчас `batch_size = 1`, поэтому числа верны. | исправлено (guard в `extract_viralm`) |
| 8 | низкая (структура) | `figures/significance.py:95` | `best_and_rest` был мёртвым кодом (только в тесте), а Fig. 2/S1 дублировали его логику. | исправлено: `against_best` возвращает сырые p, Fig. 2 и S1 его используют |
| 9 | низкая | `figures/supplementary.py:243` | `build_early_layer` не сверял число слоёв таблицы и бенчмарка (`iloc` без проверки), использовал `argmax`, жёстко зашитое 5 и дублировал группы. | исправлено |
| 10 | низкая | `cli.py` (`stage_select`) | k-sweep писался в `config.METRICS`, игнорируя `--metrics-dir`. | исправлено |
| 11 | косметика | `metrics/tables.py`, `figures/main.py`, `selection.py` | Неиспользуемые `PROTOCOL_PREFIX/PROTOCOLS`, импорт `registry`, импорт `store`. | исправлено |
| 12 | косметика | docstring: `figures/main.py`, `figures/supplementary.py`, `config.py` (`FIG_MAIN`), `store.model_dir`, `statistics.py` («ten pairs»); заголовок «(a) 4-mer profile» на Fig. S1 (ось — k от 1 до 10) | Устаревшие и вводящие в заблуждение описания. | исправлено |
| 13 | косметика | `README.md`, `results/README.md` | Таблица фигур: «Fig. 3» вместо S1, нет Fig. 8, у Fig. 1 указан силуэт, нет колонок `p_*_bh`. | исправлено |
| 14 | методика | `figures/main.py:71,147` | Звёзды Fig. 1 и Fig. 2 считаются in-sample: лучший слой выбран на тех же 50 образцах, по которым идёт тест. Это winner's curse: тесты «трансформер vs 4-mer» и «слой vs лучший» антиконсервативны. В `box_fig1` уже используется правило `split`. | предложение: указать в подписи или брать p из `pairwise_model_tests.csv` (`selection == "split"`) |
| 15 | методика | `cli.py:414` (`--select-by`), `selection.py`, `metrics/runner.py:212` | `select --select-by ARI` записывает в `selection.json` слой по ARI, и `metrics --layers selected` считает его. Фигуры же всегда выбирают слой заново по `REFERENCE_METRIC = P5` и не читают `selection.json`. При значении по умолчанию (P5) всё согласовано. | предложение: сделать единый источник (читать `selection.json` в `layer_of`) или убрать `--select-by` |
| 16 | структура | `figures/supplementary.py` | Fig. 8 (основная фигура) строится в модуле supplementary. | предложение: вынести scaling/distance и early-layer в `figures/sensitivity.py` |
| 17 | структура | `models.py` (`tunes_layer`), `representations/minhash.py` (`jaccard`) | Не используются в пакете (только в тестах или нигде). | предложение (безвредно) |
| 18 | информационно | `transformers.extract_evo2` | `model.revision` передаётся как `--model_id`; семантика неочевидна. | предложение: переименовать поле или аргумент |

Проверено и признано корректным: самоисключение и исключение группы в
`neighbours.knn_excluding`; `precision_at_ks` с паддингом; JSD-ранжирование без корня
и √JSD в `distance_matrix`; MinHash (каноника, fmix64, универсальные хеши); k-medoids
(++-инициализация, alternate, выбор лучшего рестарта); `P5_lgo` (выравнивание по
манифесту, отказ при несовпадении); усреднение окон ViraLM и порядок фрагментов; RC-пары
в `own_rc` (assert выравнивания); единица усреднения (геном → образец);
трансдуктивный `StandardScaler` (задокументирован); сиды (`seed + i` для выборок,
`genome_rng` по crc32); Wilcoxon (`zero_method="wilcox"`, при нулевых разностях p = 1);
Holm/BH; выравнивание пар по индексу образца во всех тестах фигур (`compare_pair` делает
`concat` по индексу); семейства BH (per-figure для Fig. 1, 2, S1, 8 и early-layer;
study-wide `p_bh_all` при `selection == reference` для Fig. 4–6); разбор имён файлов
(`P@k_matrix_*`, `sample_i`, `REVERSECOMPLEMENT*`); обработка семи моделей статьи;
`kmer_canon` и `minhash1k` не ломают сборку фигур; ссылок на `distance_protocols`,
`PROTOCOLS_DIR`, `fig3_kmer_sweep`, `protocol_comparison` в коде и README не осталось
(`--protocol genome|order` сохранён).

## Что изменено и чем покрыто

| Исправление | Изменения | Тесты |
|---|---|---|
| 1. Метки слоёв | `reference.layer_label` и `reference.best_position` — единственная реализация (`selection.layer_label` делегирует). `summary` и `layerwise` пишут метку. `build_fig1` берёт значения по позиции, а не по `best_layer`. `build_fig2` индексирует строки метками. `_reference_panel` (box_fig1) и `build_early_layer` пишут метки. | `tests/test_significance.py::test_fig1_and_fig2_show_viralm_layers_by_label_and_test_by_position`, `test_regression.py` (CSV перегенерированы) |
| 2. Пунктир выбранного слоя | `supplementary._chosen_label` переводит позицию из `selection.json` в метку. | `test_supplementary_marks_the_chosen_layer_by_label` |
| 3. NaN в поправках | Декоратор `statistics._finite_only` для `holm` и `benjamini_hochberg`; `compare_pair` при n = 0 возвращает NaN; `stars(NaN)` даёт `""`. | `test_primitives.py::test_corrections_leave_missing_p_values_out_of_the_family`, `::test_compare_pair_without_common_samples_has_no_p_value`, `test_significance.py::test_adjust_ignores_comparisons_without_a_p_value` |
| 4. Порядок в `figures` | `statistics_figures.build` вызывается до `main_figures.build_all`; при `--plot-only` ничего не пересчитывается. | `test_figures_stage_builds_the_study_tests_before_the_main_figures` |
| 5. `nanargmax` | `selection.choose_layer`. | `test_primitives.py::test_a_layer_that_is_nan_on_every_sample_is_never_chosen` |
| 6. Частичные модели | `main._has_table`; пропуск в `_unseen_table` и `build_fig6`. | `test_a_registry_model_without_some_tables_is_left_out_not_fatal` |
| 7. Батч ViraLM | `extract_viralm` выбрасывает `ValueError` при `batch_size != 1`. | `test_extraction.py::test_the_viralm_hook_refuses_batches_it_would_average_together` (и проверка `--batch_size 1`) |
| 8. `against_best` | Общий хелпер для Fig. 2 и S1, BH один раз на фигуру. | `test_against_best_compares_every_row_with_the_best`, `test_paired_p_aligns_samples_by_index_not_by_position` |
| 9–13 | Проверка длины, `nanargmax`, `config.PK_FOR_FIGURES` в early-layer; `--metrics-dir` для k-sweep; удалён мёртвый код; docstring; README. | общий прогон |

## Перегенерированные артефакты (только из закоммиченных таблиц, без нового счёта)

Перегенерированы, потому что исправление 1 меняет то, как строятся эти файлы:

- `results/figure_data/fig1_reference_benchmark.csv`: `best_layer` ViraLM 4→5 (ARI) и 2→3 (P5).
- `results/figure_data/fig2_layerwise.csv` и `supplementary/layerwise_reference.csv`:
  `layer` ViraLM 0..11→1..12. Средние, std и p не изменились (разница ≤ 1e-16).
- `results/statistics/pairwise_model_tests.csv` и `panel_samples.csv`: изменились только
  `layer`, `layer_a`, `layer_b` ViraLM в панелях fig1. p-значения те же.
- Фигуры: `figures/main/fig1_reference_benchmark`, `fig2_layerwise` и `figS1_kmer_sweep`
  (заголовок (a)); `fig4`, `fig5`, `fig6` и `fig8` перерисованы с теми же данными;
  `figures/statistics/box_fig1`, `box_fig4`, `box_fig5`; все `figures/supplementary/*/viralm*`.
  `fig7_pk_vs_cross` **не трогался**. Графики остальных моделей и прочие статистические
  фигуры, у которых поменялись только байты, восстановлены через `git checkout`.
- `results/metrics/**` и `results/run_manifest.json` не изменены.

**Для рукописи:** выбранный слой ViraLM — блок 3 (в Fig. 1 и Fig. 4 теперь одинаково).
Лучший по ARI — блок 5. Если в тексте статьи стоит «ViraLM L2», это нужно исправить на L3.

## Команды для сервера

Пересчёт не требуется: все исправления касаются чтения таблиц, подписей, статистики и
устойчивости. Локально перерисовать всё (кроме ручной правки Fig. 7) можно так:

```
python run_pipeline.py figures --which all
```

Этот вызов также перерисует `fig7_pk_vs_cross.png`, поэтому при необходимости сохраните
ручную версию. По пересчётам, уже перечисленным в README («What requires re-running»),
изменений нет.

## Тесты

`python -m pytest -q --ignore=tests/test_end_to_end.py`

- До ревью: **126 passed**.
- После: **136 passed** (+10 новых тестов, `test_best_and_rest…` переименован в
  `test_against_best…`). Три теста `test_regression` после исправления 1 падали на
  колонке `layer` ViraLM и снова проходят после перегенерации CSV.
- `tests/test_end_to_end.py` локально не запускался (ограничение среды Windows/scipy DLL).
