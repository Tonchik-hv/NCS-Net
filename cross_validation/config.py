"""Конфигурация проекта кросс-валидации (пути и гиперпараметры CV)."""
import os

# Корень репозитория NN (исходники compton_averaging, кеш датасета)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CV_ROOT = os.path.dirname(os.path.abspath(__file__))
CV_RESULTS_DIR = os.path.join(CV_ROOT, "cv_results")

# Согласовано с train.py / compton_averaging.config
N_FOLDS = 5
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
TRAIN_FRACTION_REFERENCE = 0.8  # для сравнения с одиночным split (~80/20)

STRATIFY_N_BINS = 5
STRATIFY_SEED_OFFSET = 20003  # отдельно от DATASET_SEED и val split (+10007)


def folds_indices_file(n_folds: int, num_samples: int, timesteps: int, seed: int) -> str:
    return os.path.join(
        CV_RESULTS_DIR,
        f"cv_folds_k{n_folds}_ns{num_samples}_ts{timesteps}_seed{seed}.pkl",
    )


def fold_dir(fold: int) -> str:
    return os.path.join(CV_RESULTS_DIR, f"fold_{fold}")


def fold_model_path(fold: int) -> str:
    return os.path.join(fold_dir(fold), "model_best.pth")


def fold_normalizer_path(fold: int) -> str:
    return os.path.join(fold_dir(fold), "normalizer.pkl")


def fold_metrics_path(fold: int) -> str:
    return os.path.join(fold_dir(fold), "metrics.json")


def fold_history_plot_path(fold: int) -> str:
    return os.path.join(fold_dir(fold), "training_history.png")


def summary_path() -> str:
    return os.path.join(CV_RESULTS_DIR, "cv_summary.json")
