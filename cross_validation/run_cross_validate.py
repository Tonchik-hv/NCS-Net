#!/usr/bin/env python3
"""Стратифицированная K-fold кросс-валидация NCS-Net (DirectComptonModel)."""
import argparse
import functools

print = functools.partial(print, flush=True)
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

CV_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CV_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compton_averaging.config import (  # noqa: E402
    DATASET_NUM_SAMPLES,
    DATASET_SEED,
    DATASET_TIMESTEPS,
    dataset_cache_dir,
)
from compton_averaging.dataset import ComptonDataset  # noqa: E402
from compton_averaging.model import DirectComptonModel  # noqa: E402
from compton_averaging.normalizer import DataNormalizer  # noqa: E402

if CV_ROOT not in sys.path:
    sys.path.insert(0, CV_ROOT)

from config import (  # noqa: E402
    BATCH_SIZE,
    CV_RESULTS_DIR,
    EPOCHS,
    LEARNING_RATE,
    N_FOLDS,
    STRATIFY_N_BINS,
    STRATIFY_SEED_OFFSET,
    fold_dir,
    fold_history_plot_path,
    fold_metrics_path,
    fold_model_path,
    fold_normalizer_path,
    folds_indices_file,
    summary_path,
)
from kfold import fold_sizes, stratified_kfold_indices  # noqa: E402
from metrics import evaluate_val_loader  # noqa: E402
from training_cv import train_model  # noqa: E402


def _load_p_all(dataset):
    if getattr(dataset, "_cache_ready", False) and dataset._cache_p is not None:
        return np.asarray(dataset._cache_p)
    return np.stack([dataset.data[i][0] for i in range(len(dataset))], axis=0)


def _get_or_create_folds(dataset, n_folds: int, seed: int, force_rebuild: bool):
    path = folds_indices_file(n_folds, dataset.num_samples, DATASET_TIMESTEPS, seed)
    os.makedirs(CV_RESULTS_DIR, exist_ok=True)
    if os.path.isfile(path) and not force_rebuild:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Загружены индексы фолдов: {path}")
        return data["folds"], path

    p_all = _load_p_all(dataset)
    folds = stratified_kfold_indices(
        p_all,
        n_folds=n_folds,
        seed=seed + STRATIFY_SEED_OFFSET,
        gamma=dataset.gamma,
        n_bins=STRATIFY_N_BINS,
    )
    payload = {
        "folds": folds,
        "n_folds": n_folds,
        "num_samples": dataset.num_samples,
        "timesteps": DATASET_TIMESTEPS,
        "seed": seed,
        "stratify_seed": seed + STRATIFY_SEED_OFFSET,
        "sizes": fold_sizes(folds),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Сохранены индексы фолдов: {path} | размеры (train, val): {payload['sizes']}")
    return folds, path


def _fold_done(fold: int) -> bool:
    return os.path.isfile(fold_metrics_path(fold)) and os.path.isfile(fold_model_path(fold))


def _fold_matches_epochs(fold: int, epochs: int) -> bool:
    """True если metrics.json есть и в нём столько же эпох обучения, сколько запрошено."""
    path = fold_metrics_path(fold)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return int(m.get("epochs", -1)) == int(epochs)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def _should_skip_fold(fold: int, args) -> bool:
    """С --resume пропускаем фолд только если есть метрики, веса и совпадает число эпох."""
    if not args.resume:
        return False
    if not _fold_done(fold):
        return False
    if not _fold_matches_epochs(fold, args.epochs):
        print(
            f"Фолд {fold}: найдены старые результаты (эпохи ≠ {args.epochs}) — переобучение."
        )
        return False
    return True


def run_fold(fold_idx, train_idx, val_idx, dataset, args):
    os.makedirs(fold_dir(fold_idx), exist_ok=True)
    if _should_skip_fold(fold_idx, args):
        print(f"Фолд {fold_idx}: пропуск (уже есть metrics + checkpoint, эпох={args.epochs}).")
        with open(fold_metrics_path(fold_idx), "r", encoding="utf-8") as f:
            return json.load(f)

    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\n=== Фолд {fold_idx} | train={len(train_idx)} val={len(val_idx)} ===")

    t0 = time.perf_counter()
    norm_path = fold_normalizer_path(fold_idx)
    partial_resume = (
        args.resume
        and os.path.isfile(fold_model_path(fold_idx))
        and not os.path.isfile(fold_metrics_path(fold_idx))
    )
    if partial_resume and os.path.isfile(norm_path):
        print(f"Фолд {fold_idx}: частичное возобновление (checkpoint + normalizer).")
        with open(norm_path, "rb") as f:
            normalizer = pickle.load(f)
    else:
        normalizer = DataNormalizer()
        normalizer.fit(train_loader)
        normalizer_cpu = DataNormalizer()
        normalizer_cpu.p_mean = normalizer.p_mean.cpu()
        normalizer_cpu.p_std = normalizer.p_std.cpu()
        normalizer_cpu.y_mean = normalizer.y_mean.cpu()
        normalizer_cpu.y_std = normalizer.y_std.cpu()
        with open(norm_path, "wb") as f:
            pickle.dump(normalizer_cpu, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer.to(device)

    model = DirectComptonModel()
    train_t0 = time.perf_counter()
    train_out = train_model(
        model,
        train_loader,
        val_loader,
        normalizer,
        epochs=args.epochs,
        lr=args.lr,
        save_path=fold_model_path(fold_idx),
        history_plot_path=fold_history_plot_path(fold_idx),
        resume_from_path=fold_model_path(fold_idx) if partial_resume else None,
    )
    train_seconds = time.perf_counter() - train_t0

    try:
        checkpoint = torch.load(
            fold_model_path(fold_idx), map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(fold_model_path(fold_idx), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    eval_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_val_loader(model, normalizer, eval_loader, device)
    metrics.update(
        {
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "best_val_loss": float(train_out["best_val_loss"]),
            "final_val_loss": float(train_out["final_val_loss"]),
            "final_train_loss": float(train_out["final_train_loss"]),
            "best_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "train_wall_seconds": train_seconds,
            "normalizer_fit_seconds": time.perf_counter() - t0 - train_seconds,
            "device": train_out["device"],
        }
    )

    with open(fold_metrics_path(fold_idx), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(
        f"Фолд {fold_idx} готов | best_val_loss={metrics['best_val_loss']:.4f} "
        f"MSE_log={metrics['mse_log_mean']:.6f} area_err={metrics['area_rel_error_pct_mean']:.2f}%"
    )
    return metrics


def aggregate(fold_metrics_list):
    keys = [
        "best_val_loss",
        "mse_log_mean",
        "mae_log_mean",
        "area_rel_error_pct_mean",
        "train_wall_seconds",
    ]
    summary = {"per_fold": fold_metrics_list, "aggregate": {}}
    for k in keys:
        vals = [m[k] for m in fold_metrics_list]
        summary["aggregate"][k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="K-fold CV для DirectComptonModel")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Пропускать фолды, у которых уже есть metrics.json + model с тем же числом эпох, что и --epochs",
    )
    parser.add_argument("--rebuild-folds", action="store_true", help="Пересоздать pickle с индексами фолдов")
    parser.add_argument("--fold", type=int, default=None, help="Обучить только один фолд (для отладки)")
    args = parser.parse_args()

    os.makedirs(CV_RESULTS_DIR, exist_ok=True)

    cache_dir = dataset_cache_dir(REPO_ROOT)
    dataset = ComptonDataset(
        num_samples=DATASET_NUM_SAMPLES,
        timesteps=DATASET_TIMESTEPS,
        cache_dir=cache_dir,
        seed=DATASET_SEED,
    )

    folds, folds_path = _get_or_create_folds(
        dataset, args.folds, DATASET_SEED, force_rebuild=args.rebuild_folds
    )

    fold_range = range(args.folds)
    if args.fold is not None:
        fold_range = [args.fold]

    all_metrics = []
    for k in fold_range:
        train_idx, val_idx = folds[k]
        all_metrics.append(run_fold(k, train_idx, val_idx, dataset, args))

    # Сводка по всем завершённым фолдам (в т.ч. пропущенным через --resume)
    completed = []
    for k in range(args.folds):
        if os.path.isfile(fold_metrics_path(k)):
            with open(fold_metrics_path(k), "r", encoding="utf-8") as f:
                completed.append(json.load(f))
    if len(completed) < args.folds:
        print(
            f"Внимание: завершено {len(completed)}/{args.folds} фолдов; "
            "сводка построена по готовым фолдам."
        )
    summary = aggregate(completed if completed else all_metrics)
    summary["meta"] = {
        "dataset_num_samples": dataset.num_samples,
        "timesteps": DATASET_TIMESTEPS,
        "dataset_seed": DATASET_SEED,
        "n_folds": args.folds,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "folds_indices_file": folds_path,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path(), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    agg = summary["aggregate"]
    print("\n=== Сводка CV ===")
    print(f"best_val_loss: {agg['best_val_loss']['mean']:.4f} ± {agg['best_val_loss']['std']:.4f}")
    print(f"MSE (log):     {agg['mse_log_mean']['mean']:.6f} ± {agg['mse_log_mean']['std']:.6f}")
    print(f"MAE (log):     {agg['mae_log_mean']['mean']:.6f} ± {agg['mae_log_mean']['std']:.6f}")
    print(
        f"Area err %:    {agg['area_rel_error_pct_mean']['mean']:.2f} ± "
        f"{agg['area_rel_error_pct_mean']['std']:.2f}"
    )
    print(f"Сводка: {summary_path()}")


if __name__ == "__main__":
    main()
