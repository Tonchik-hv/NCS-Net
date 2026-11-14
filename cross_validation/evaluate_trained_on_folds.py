#!/usr/bin/env python3
"""
Оценка уже обученной модели (например, 300 эпох на вашем 80/20 split) на val-частях
5-fold разбиения из cv_folds_*.pkl.

Для каждого фолда выводится:
  - **full_fold_val** — весь val фолда (~160k): часть точек может входить в ваш train → оптимистичная смесь;
  - **holdout_only** — пересечение val фолда с фиксированным hold-out (`val_indices_*.pkl`): только примеры,
    которые модель не видела при обучении (честнее для одной модели).

Запуск из корня репозитория NN:
  cross_validation/.venv/bin/python cross_validation/evaluate_trained_on_folds.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

CV_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CV_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CV_ROOT not in sys.path:
    sys.path.insert(0, CV_ROOT)

from compton_averaging.config import (  # noqa: E402
    DATASET_NUM_SAMPLES,
    DATASET_SEED,
    DATASET_TIMESTEPS,
)
from compton_averaging.io import get_device, load_dataset, load_train_val_data  # noqa: E402
from compton_averaging.model import DirectComptonModel  # noqa: E402
from compton_averaging.normalizer import load_normalizer_from_pickle  # noqa: E402
from compton_averaging.config import model_file, normalizer_file  # noqa: E402

from config import CV_RESULTS_DIR  # noqa: E402
from metrics import evaluate_val_loader  # noqa: E402


def _load_folds(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "folds" in data:
        return data["folds"], data
    return data, None


def load_checkpoint(model: DirectComptonModel, path: str, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)


def main():
    p = argparse.ArgumentParser(
        description="Метрики одной обученной модели на val каждого fold (5-fold pickle)"
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default=REPO_ROOT,
        help="Корень репозитория NN (датасет, пути к модели/нормализатору)",
    )
    p.add_argument("--model", type=str, default=None, help="Путь к .pth (по умолчанию model_file(base_dir))")
    p.add_argument("--normalizer", type=str, default=None, help="Путь к .pkl (по умолчанию normalizer_file)")
    p.add_argument(
        "--folds-pkl",
        type=str,
        default=None,
        help="Пикль с разбиением (по умолчанию cross_validation/cv_results/cv_folds_k5_*.pkl)",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--only-fold",
        type=int,
        default=None,
        metavar="K",
        help="Оценить только один фолд (для проверки)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Ограничить число примеров в каждом счёте (первые N по списку индексов; только для быстрого smoke-теста)",
    )
    p.add_argument("--device", type=str, default=None, choices=("cuda", "cpu", None))
    p.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Сохранить результаты в JSON (по умолчанию cross_validation/cv_results/trained_model_on_folds.json)",
    )
    args = p.parse_args()

    root = os.path.abspath(args.base_dir)
    folds_path = args.folds_pkl or os.path.join(
        CV_RESULTS_DIR,
        f"cv_folds_k5_ns{DATASET_NUM_SAMPLES}_ts{DATASET_TIMESTEPS}_seed{DATASET_SEED}.pkl",
    )
    if not os.path.isfile(folds_path):
        raise FileNotFoundError(
            f"Нет файла фолдов: {folds_path}. Создайте через run_cross_validate или укажите --folds-pkl"
        )

    model_path = args.model or model_file(root)
    norm_path = args.normalizer or normalizer_file(root)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.isfile(norm_path):
        raise FileNotFoundError(norm_path)

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    dataset = load_dataset(root)
    _, _, holdout_val_indices = load_train_val_data(dataset, root)
    holdout_set = set(holdout_val_indices)

    folds_list, _meta = _load_folds(folds_path)

    normalizer = load_normalizer_from_pickle(norm_path)
    normalizer.to(device)

    model = DirectComptonModel()
    load_checkpoint(model, model_path, device)

    out_rows = []
    summary_full = {"mse": [], "mae": [], "area_pct": []}
    summary_hold = {"mse": [], "mae": [], "area_pct": []}

    if args.only_fold is not None:
        if args.only_fold < 0 or args.only_fold >= len(folds_list):
            raise ValueError(f"--only-fold must be in [0, {len(folds_list) - 1}]")
        fold_iter = [(args.only_fold, folds_list[args.only_fold])]
    else:
        fold_iter = list(enumerate(folds_list))

    for k, (_train_idx, val_idx) in fold_iter:
        n_full = len(val_idx)
        val_use = val_idx if args.max_samples is None else val_idx[: args.max_samples]

        val_sub = Subset(dataset, val_use)
        loader_full = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)
        m_full = evaluate_val_loader(model, normalizer, loader_full, device)

        hold_idx = sorted(set(val_idx) & holdout_set)
        if hold_idx:
            hold_use = hold_idx if args.max_samples is None else hold_idx[: args.max_samples]
            hold_sub = Subset(dataset, hold_use)
            loader_hold = DataLoader(hold_sub, batch_size=args.batch_size, shuffle=False)
            m_hold = evaluate_val_loader(model, normalizer, loader_hold, device)
        else:
            hold_use = []
            m_hold = None

        row = {
            "fold": k,
            "n_fold_val": n_full,
            "n_fold_val_evaluated": len(val_use),
            "n_holdout_in_fold_val": len(hold_idx),
            "n_holdout_evaluated": len(hold_use),
            "fraction_holdout_in_fold_val": len(hold_idx) / n_full if n_full else 0.0,
            "max_samples_cap": args.max_samples,
            "full_fold_val": m_full,
            "holdout_only": m_hold,
        }
        out_rows.append(row)
        summary_full["mse"].append(m_full["mse_log_mean"])
        summary_full["mae"].append(m_full["mae_log_mean"])
        summary_full["area_pct"].append(m_full["area_rel_error_pct_mean"])
        if m_hold:
            summary_hold["mse"].append(m_hold["mse_log_mean"])
            summary_hold["mae"].append(m_hold["mae_log_mean"])
            summary_hold["area_pct"].append(m_hold["area_rel_error_pct_mean"])

        print(f"\n--- Fold {k} ---")
        print(
            f"  full val (n={m_full['n_val_samples']}): "
            f"MSE(log)={m_full['mse_log_mean']:.6f} | "
            f"MAE(log)={m_full['mae_log_mean']:.6f} | "
            f"area%={m_full['area_rel_error_pct_mean']:.2f}"
        )
        if m_hold:
            print(
                f"  holdout ∩ fold val (n={m_hold['n_val_samples']}): "
                f"MSE(log)={m_hold['mse_log_mean']:.6f} | "
                f"MAE(log)={m_hold['mae_log_mean']:.6f} | "
                f"area%={m_hold['area_rel_error_pct_mean']:.2f}"
            )
        else:
            print("  holdout ∩ fold val: пусто (проверьте folds и val_indices)")

    def _agg(xs):
        if not xs:
            return None
        a = np.array(xs, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}

    payload = {
        "meta": {
            "model_path": model_path,
            "normalizer_path": norm_path,
            "folds_pkl": folds_path,
            "holdout_val_size": len(holdout_val_indices),
            "max_samples": args.max_samples,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "per_fold": out_rows,
        "aggregate_full_fold_val": {
            "mse_log": _agg(summary_full["mse"]),
            "mae_log": _agg(summary_full["mae"]),
            "area_rel_pct": _agg(summary_full["area_pct"]),
        },
        "aggregate_holdout_only": {
            "mse_log": _agg(summary_hold["mse"]),
            "mae_log": _agg(summary_hold["mae"]),
            "area_rel_pct": _agg(summary_hold["area_pct"]),
        },
    }

    out_path = args.out_json or os.path.join(CV_RESULTS_DIR, "trained_model_on_folds.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nСохранено: {out_path}")

    af = payload["aggregate_full_fold_val"]["area_rel_pct"]
    ah = payload["aggregate_holdout_only"]["area_rel_pct"]
    if af and ah:
        print(
            "\nСводка ошибки площади (mean ± std по фолдам):\n"
            f"  full fold val:    {af['mean']:.2f}% ± {af['std']:.2f}%\n"
            f"  holdout ∩ fold:   {ah['mean']:.2f}% ± {ah['std']:.2f}%"
        )


if __name__ == "__main__":
    main()
