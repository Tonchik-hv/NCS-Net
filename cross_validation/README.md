# Кросс-валидация NCS-Net

Отдельный проект для **стратифицированной K-fold** оценки `DirectComptonModel` без изменения файлов в корне репозитория.

## Зависимости

Используются исходные модули из корня: `compton_averaging/*`, `src/calc_spectrum.py`, кеш датасета  
`compton_dataset_cache_angles_pi_2_ns800000_ts2500_seed42/`.

Локальные копии/обёртки только в этой папке:

| Файл | Назначение |
|------|------------|
| `config.py` | пути `cv_results/`, гиперпараметры CV |
| `kfold.py` | стратифицированное разбиение на K фолдов |
| `training_cv.py` | копия цикла обучения с возвратом метрик |
| `metrics.py` | MSE/MAE (log), ошибка по площади на val |
| `run_cross_validate.py` | главный скрипт |
| `evaluate_trained_on_folds.py` | **одна** обученная модель на val каждого fold + пересечение с hold-out |

## Запуск

Из **корня репозитория** (чтобы импорт `compton_averaging` работал):

```bash
# venv создаётся один раз: python3 -m venv cross_validation/.venv && cross_validation/.venv/bin/pip install torch numpy scipy matplotlib

# Честная 5-fold CV: на каждом фолде новая модель, 300 эпох (как train.py)
bash cross_validation/run_cv_300epochs.sh
# эквивалент:
cross_validation/.venv/bin/python -u cross_validation/run_cross_validate.py --epochs 300 --resume

# Краткие выводы в Markdown (после завершения всех фолдов)
cross_validation/.venv/bin/python cross_validation/write_conclusions.py
```

Параметры по умолчанию в `config.py`: **5 фолдов**, **300 эпох**, `lr=1e-3`, `batch_size=128` (как в `train.py`).  
Флаги: `--epochs N`, `--fold K`, `--resume` (пропуск только если есть `metrics.json` + `model_best.pth` **и** в метриках `epochs == N`; иначе фолд переобучается; старый прогон на 30 эпохах не мешает `--epochs 300 --resume`). Прерванный фолд дозапускается с checkpoint.

Результаты: `cross_validation/cv_results/fold_*/`, сводка `cv_results/cv_summary.json`. Лог: `... | tee cross_validation/cv_results/run.log`.

**Заметка:** в `CONCLUSIONS.md` зафиксирован старый прогон **30** эпох; после полного прогона на **300** перегенерируйте выводы командой `write_conclusions.py`.

Выводы: **`CONCLUSIONS.md`** (генерируется `write_conclusions.py`).

### Одна готовая модель (300 эпох) на 5 fold val

Используются те же индексы, что и при K-fold (`cv_results/cv_folds_k5_*.pkl`), и ваш нормализатор/веса из корня.

```bash
cross_validation/.venv/bin/python cross_validation/evaluate_trained_on_folds.py
# быстрый smoke-тест (первые N примеров в каждом подсчёте):
# cross_validation/.venv/bin/python cross_validation/evaluate_trained_on_folds.py --only-fold 0 --max-samples 512
# только «чистый» hold-out внутри каждого val фолда — см. поле aggregate_holdout_only в JSON
```

Результат: `cross_validation/cv_results/trained_model_on_folds.json` (для **каждого** фолда: метрики на полном val фолда и на `holdout ∩ fold_val`).
