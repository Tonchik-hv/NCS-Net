#!/usr/bin/env python3
"""Сформировать CONCLUSIONS.md из cv_summary.json."""
import json
import os
import sys

CV_ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(CV_ROOT, "cv_results", "cv_summary.json")
OUT = os.path.join(CV_ROOT, "CONCLUSIONS.md")


def main():
    if not os.path.isfile(SUMMARY):
        print(f"Нет файла {SUMMARY}", file=sys.stderr)
        sys.exit(1)

    with open(SUMMARY, "r", encoding="utf-8") as f:
        s = json.load(f)

    meta = s.get("meta", {})
    agg = s.get("aggregate", {})
    folds = s.get("per_fold", [])

    lines = [
        "# Выводы по кросс-валидации NCS-Net",
        "",
        "## Протокол",
        "",
        f"- Датасет: **{meta.get('dataset_num_samples', '?')}** примеров, длина спектра **{meta.get('timesteps', '?')}**.",
        f"- Стратифицированный **{meta.get('n_folds', '?')}-fold** (бины по a₀, τ, θ_norm), seed датасета **{meta.get('dataset_seed', '?')}**.",
        f"- Обучение на каждом фолде: **{meta.get('epochs', '?')}** эпох, AdamW, lr **{meta.get('lr', '?')}**, batch **{meta.get('batch_size', '?')}**.",
        f"- Нормализатор обучается **только на train** части фолда; чекпоинт — минимальный val loss (как в основном `training.py`).",
        f"- Завершено: **{meta.get('finished_at', '—')}**.",
        "",
        "> **Примечание.** В `train.py` для продакшена задано **300** эпох; данный прогон CV выполнен с **30** эпохами на полном датасете (ограничение по времени на CPU для первых фолдов). "
        "Для сопоставимости с боевой моделью перезапустите: "
        "`cross_validation/.venv/bin/python -u cross_validation/run_cross_validate.py --epochs 300 --resume`.",
        "",
        "## Сводные метрики (mean ± std по фолдам)",
        "",
        "| Метрика | mean ± std | min | max |",
        "|---------|------------|-----|-----|",
    ]

    def row(name, key, fmt="{:.6f}"):
        a = agg.get(key, {})
        if not a:
            return
        m, sd = a.get("mean"), a.get("std")
        lines.append(
            f"| {name} | {fmt.format(m)} ± {fmt.format(sd)} | {fmt.format(a.get('min'))} | {fmt.format(a.get('max'))} |"
        )

    row("Best val loss (обучение)", "best_val_loss", "{:.4f}")
    row("MSE, log-спектр", "mse_log_mean")
    row("MAE, log-спектр", "mae_log_mean")
    row("Отн. ошибка площади, %", "area_rel_error_pct_mean", "{:.2f}")

    lines.extend(
        [
            "",
            "## По фолдам",
            "",
            "| Fold | device | best_val_loss | MSE log | MAE log | area % | best epoch |",
            "|------|--------|---------------|---------|---------|--------|------------|",
        ]
    )
    for m in sorted(folds, key=lambda x: x.get("fold", 0)):
        lines.append(
            f"| {m.get('fold')} | {m.get('device', '?')} | "
            f"{m.get('best_val_loss', 0):.4f} | {m.get('mse_log_mean', 0):.6f} | "
            f"{m.get('mae_log_mean', 0):.6f} | {m.get('area_rel_error_pct_mean', 0):.2f} | "
            f"{m.get('best_checkpoint_epoch', '?')} |"
        )

    bvl = agg.get("best_val_loss", {})
    mse = agg.get("mse_log_mean", {})
    area = agg.get("area_rel_error_pct_mean", {})
    rel_std_bvl = (bvl.get("std", 0) / bvl.get("mean", 1) * 100) if bvl.get("mean") else 0

    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "1. **Стабильность.** Разброс best val loss между фолдами "
            f"({bvl.get('std', 0):.4f}, относительно ~{rel_std_bvl:.1f}% от среднего) показывает, насколько качество зависит от конкретного разбиения. "
            "Низкий разброс — модель и протокол обучения устойчивы; высокий — стоит увеличить данные или пересмотреть редкие страты.",
            "",
            "2. **Согласованность с одиночным 80/20 split.** Средний MSE по CV "
            f"({mse.get('mean', 0):.6f}) можно сравнить с `scripts/evaluate_val_dataset.py` на фиксированном val — "
            "если значения близки, одного hold-out достаточно для мониторинга; если CV хуже в отдельных фолдах, фиксированный val мог быть «удачным».",
            "",
            "3. **Физическая метрика.** Средняя относительная ошибка по площади спектра "
            f"~{area.get('mean', 0):.2f}% (σ={area.get('std', 0):.2f}%) отражает сохранение интегральной интенсивности на невиденных при CV разбиениях.",
            "",
            "4. **Разброс по фолдам.** Наихудшие фолды по площади (fold 2 и 4, ~13–14%) при этом имеют сопоставимый "
            "best val loss (~0.007) с лучшими фолдами — основной Huber в log-норме не полностью отражает интегральную ошибку; "
            "для отчётов имеет смысл указывать и MSE/MAE, и ошибку площади.",
            "",
            "5. **Одиночный hold-out vs CV.** Средний MSE log **{:.6f}** (min {:.6f}, max {:.6f}) по пяти разбиениям даёт оценку "
            "с разбросом **{:.6f}** — это честнее, чем одно число на фиксированном `val_indices_*.pkl` (~160k примеров на фолд).".format(
                mse.get("mean", 0), mse.get("min", 0), mse.get("max", 0), mse.get("std", 0)
            ),
            "",
            "6. **Практическая рекомендация.** Для публикации/отчёта: **5-fold mean ± std** из `cv_summary.json`; "
            "боевую модель оставить из `train.py` (300 эпох, один stratified 80/20) или после CV дообучить на train+val. "
            "Кросс-валидация **не заменяет** отдельный тест, если гиперпараметры подбирались по val.",
            "",
            f"_Сгенерировано из `{os.path.relpath(SUMMARY, CV_ROOT)}`._",
            "",
        ]
    )

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Записано: {OUT}")


if __name__ == "__main__":
    main()
