# NCS-Net

Нейросеть **NCS-Net** для прямой генерации усреднённых спектров нелинейного комптоновского рассеяния по параметрам `(a₀, τ, θ)`. Репозиторий содержит код обучения, оценки, визуализации и инференса для произвольных параметров и для пучка электронов.

Исходный монолитный скрипт `NN_for_averaging_full_param_angles_pi_2_inference_time.py` сохранён без изменений; для работы из репозитория используйте модульную структуру ниже.

## Структура репозитория

```
.
├── train.py                          # обучение NCS-Net (главная точка входа)
├── physrev.mplstyle                  # стиль графиков matplotlib
│
├── compton_averaging/                # Python-пакет
│   ├── config.py                     # гиперпараметры и пути к артефактам
│   ├── dataset.py                    # генерация/загрузка датасета
│   ├── model.py                      # архитектура NCS-Net (класс DirectComptonModel)
│   ├── normalizer.py                 # нормализация параметров и спектров
│   ├── loss.py                       # физический лосс
│   ├── training.py                   # цикл обучения и валидации
│   ├── physics.py                    # инициализация snlcs-калькулятора
│   ├── plotting.py                   # утилиты оформления графиков
│   ├── io.py                         # загрузка NCS-Net, нормализатора, val-выборки
│   └── viz_2d_theta_maps.py          # 2D-карты I(θ, ω) и метрики
│
├── scripts/                          # самостоятельные скрипты
│   ├── evaluate_val_dataset.py       # метрики на валидационном датасете
│   ├── benchmark_cpu_inference.py    # замер времени инференса 1D на CPU
│   ├── visualize_2d_theta_maps.py    # 2D-карты и 2D-метрики
│   ├── visualize_random_predictions.py  # N случайных 1D-графиков
│   ├── predict_spectrum_from_params.py  # инференс для одной тройки (a₀, τ, θ)
│   ├── predict_beam_spectrum_corrected.py  # 2D-спектр пучка (NCS-Net / аналитика / quad)
│   └── benchmark_beam_spectrum_cpu.py  # бенчмарк 2D-спектра пучка на CPU
│
├── src/
│   └── calc_spectrum.py              # аналитика и численный расчёт 1D-спектров
│
└── config/
    ├── ExtFieldsParams.yaml      # параметры лазерного поля
    ├── eBeamParams.yaml          # параметры электронного пучка
    └── OtherParams.yaml          # прочие параметры snlcs
```

Корневые файлы `predict_spectrum_from_params.py`, `predict_beam_spectrum_corrected.py`, `benchmark_beam_spectrum_cpu.py` — **исходники**, оставлены без изменений. Для публикации и запуска из репозитория используйте их копии в `scripts/`.

## Установка

```bash
# из корня репозитория
pip install torch numpy scipy matplotlib pyyaml

# библиотека snlcs (нужна для quad_interp и части визуализаций)
pip install -e snlcs/
```

Все команды ниже запускаются **из корня репозитория**.

## Подготовка: обучение NCS-Net

Перед оценкой и инференсом нужны обученная NCS-Net и нормализатор. При первом запуске также генерируется (или загружается из кеша) датасет.

```bash
python train.py
```

Скрипт:
- создаёт/загружает кеш датасета в `compton_dataset_cache_angles_pi_2_ns800000_ts2500_seed42/`;
- формирует stratified train/val и сохраняет индексы в `val_indices_angles_pi_2_ns800000_ts2500.pkl`;
- обучает нормализатор → `normalizer_angles_pi_2_ns800000_ts2500.pkl`;
- обучает NCS-Net (если файл ещё не существует) → `model_with_averaging_angles_pi_2_ns800000_ts2500.pth`.

Если NCS-Net уже обучена, `train.py` только загружает веса и не переобучает.

---

## Скрипты: что делает и как запускать

### Обучение

| Файл | Назначение | Запуск |
|------|-----------|--------|
| `train.py` | Генерация датасета, обучение нормализатора и NCS-Net, сохранение весов | `python train.py` |

### Оценка обученной NCS-Net (валидационный датасет)

Требуют: `model_with_averaging_angles_pi_2_ns800000_ts2500.pth`, `normalizer_angles_pi_2_ns800000_ts2500.pkl`, кеш датасета.

| Файл | Назначение | Запуск |
|------|-----------|--------|
| `scripts/evaluate_val_dataset.py` | MSE/MAE в log-масштабе, ошибка по площади, время инференса на GPU/CPU | `python scripts/evaluate_val_dataset.py` |
| `scripts/benchmark_cpu_inference.py` | Среднее время одного forward-pass на CPU | `python scripts/benchmark_cpu_inference.py` |
| `scripts/visualize_2d_theta_maps.py` | 2D-карты I(θ, ω): предсказание, аналитика, Lab-frame; 2D-метрики по val | `python scripts/visualize_2d_theta_maps.py` |
| `scripts/visualize_random_predictions.py` | N случайных 1D-графиков True vs Predicted vs snlcs | `python scripts/visualize_random_predictions.py` |

### Инференс для произвольных параметров

| Файл | Назначение | Запуск |
|------|-----------|--------|
| `scripts/predict_spectrum_from_params.py` | Предсказание 1D-спектра по `(a₀, τ, θ)` с графиком True vs Predicted | `python scripts/predict_spectrum_from_params.py --a0 0.8 --tau 50 --theta 2.5` |

Полезные опции:

```bash
python scripts/predict_spectrum_from_params.py --a0 1.0 --tau 40 --theta 3.0 -o output.png
python scripts/predict_spectrum_from_params.py --a0 0.5 --tau 30 --theta 2.8 --skip-snlcs
python scripts/predict_spectrum_from_params.py --help
```

Параметры:
- `--a0` — амплитуда поля;
- `--tau` — длительность импульса в единицах π (`dur = tau * π`);
- `--theta` — угол в радианах (диапазон обучения: π/2 … π);
- `--model`, `--normalizer` — пути к весам (по умолчанию из `compton_averaging/config.py`);
- `--output` / `-o` — путь к PNG;
- `--skip-snlcs` — не вызывать snlcs (быстрее, если пакет недоступен).

### Инференс для электронного пучка

Используют конфиги `config/ExtFieldsParams.yaml` и `config/eBeamParams.yaml`.

| Файл | Назначение | Запуск |
|------|-----------|--------|
| `scripts/predict_beam_spectrum_corrected.py` | Три 2D-карты спектра пучка: NCS-Net (с преобразованием Лоренца), аналитика D2ECalculator_angles, quad_interp snlcs; метрики интеграла | `python scripts/predict_beam_spectrum_corrected.py` |
| `scripts/benchmark_beam_spectrum_cpu.py` | Бенчмарк CPU: NCS-Net vs D2ECalculator_angles для 2D-спектра пучка (без построения графиков) | `python scripts/benchmark_beam_spectrum_cpu.py` |

Выходные PNG (для `predict_beam_spectrum_corrected.py`):
- `predicted_beam_spectrum_map.png` — NCS-Net + преобразование Лоренца
- `analytical_beam_spectrum_map.png` — аналитика D2ECalculator_angles
- `analytical_beam_spectrum_theta_frequency_map_lab.png` — quad_interp snlcs (третья карта)

---

## Зависимости между модулями

```
src/calc_spectrum.py
  └── используется в: dataset, viz_2d, predict_spectrum_from_params, predict_beam_spectrum_corrected

config/*.yaml
  └── используется в: physics.py, predict_beam_spectrum_corrected, benchmark_beam_spectrum_cpu, visualize_random_predictions

snlcs (pip install -e snlcs/)
  └── нужен для: quad_interp-спектров, init_spectrum_sqinrp, predict_beam_spectrum_corrected
```

## Конфигурационные файлы snlcs

| Файл | Содержимое |
|------|-----------|
| `config/ExtFieldsParams.yaml` | Тип поля, `a₀`, длина волны, длительность импульса `T` |
| `config/eBeamParams.yaml` | Параметры гауссова пучка: энергия, размеры, число частиц |
| `config/OtherParams.yaml` | Метод интегрирования, сетка частот/углов, `random_seed` |

Перед запуском скриптов для пучка при необходимости отредактируйте `a₀`, `T` и параметры пучка в этих YAML-файлах.

## Артефакты (создаются при работе)

| Файл / каталог | Описание |
|----------------|----------|
| `model_with_averaging_angles_pi_2_ns800000_ts2500.pth` | Веса обученной NCS-Net |
| `normalizer_angles_pi_2_ns800000_ts2500.pkl` | Параметры нормализации |
| `val_indices_angles_pi_2_ns800000_ts2500.pkl` | Индексы валидационной выборки |
| `compton_dataset_cache_angles_pi_2_ns.../` | Кеш сгенерированного датасета |
| `training_history_with_averaging.png` | Кривые train/val loss |
| `val_2d_metrics_maps/` | Случайные 2D-карты из валидационных метрик |
| `averaged_prediction_result_angles_pi_2_*.png` | Случайные 1D-графики из визуализации |
| `predicted_spectrum_theta_frequency_map*.png` | 2D-карты из `visualize_2d_theta_maps.py` |

## Типичный порядок работы

```bash
# 1. Установка зависимостей
pip install torch numpy scipy matplotlib pyyaml
pip install -e snlcs/

# 2. Обучение (или загрузка готовых весов)
python train.py

# 3. Оценка на валидации
python scripts/evaluate_val_dataset.py
python scripts/visualize_random_predictions.py

# 4. Инференс для конкретных параметров
python scripts/predict_spectrum_from_params.py --a0 0.8 --tau 50 --theta 2.8

# 5. Спектр пучка (после настройки config/*.yaml)
python scripts/predict_beam_spectrum_corrected.py
python scripts/benchmark_beam_spectrum_cpu.py
```

## Пакет `compton_averaging`

Основные импорты для собственных скриптов (класс `DirectComptonModel` — реализация NCS-Net):

```python
from compton_averaging import DirectComptonModel, ComptonDataset, train_model  # DirectComptonModel = NCS-Net
from compton_averaging.io import load_for_inference
from compton_averaging.config import model_file, normalizer_file, DATASET_NUM_SAMPLES
```
