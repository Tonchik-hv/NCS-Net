"""Конфигурация путей и гиперпараметров проекта."""
import os

# --- Параметры датасета ---
DATASET_NUM_SAMPLES = 800000
DATASET_TIMESTEPS = 2500
DATASET_SEED = 42

# --- Визуализация и 2D-метрики ---
N_VIS_RESULTS = 30
N_THETA_MAP = 120
# None = набрать столько случайных сэмплов из val, сколько в val (долго); int — целевое число успешных сэмплов для 2D-метрик
N_VAL_2D_METRICS_MAX = 10  # 100
# из них на диск сохраняется только столько случайных пар карт (pred + analytical), метрики считаются по всем n_val_m
N_VAL_2D_MAPS_TO_SAVE = 10

# --- Оформление графиков ---
PLOT_TICK_FS = 16
PLOT_LABEL_FS = 20
PLOT_CBAR_LABEL_FS = 18
PLOT_TEXT_FS = 14
CBAR_SCALE_EXP = 4
PLOT_VIS_FONT_SCALE = 1.2 * 1.15

# --- Пути (относительно корня репозитория) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dataset_cache_dir(base_dir: str | None = None) -> str:
    root = base_dir or BASE_DIR
    return os.path.join(
        root,
        f"compton_dataset_cache_angles_pi_2_ns{DATASET_NUM_SAMPLES}_ts{DATASET_TIMESTEPS}_seed{DATASET_SEED}",
    )


def model_file(base_dir: str | None = None) -> str:
    root = base_dir or BASE_DIR
    return os.path.join(
        root,
        f"model_with_averaging_angles_pi_2_ns{DATASET_NUM_SAMPLES}_ts{DATASET_TIMESTEPS}.pth",
    )


def normalizer_file(base_dir: str | None = None) -> str:
    root = base_dir or BASE_DIR
    return os.path.join(
        root,
        f"normalizer_angles_pi_2_ns{DATASET_NUM_SAMPLES}_ts{DATASET_TIMESTEPS}.pkl",
    )


def val_indices_file(base_dir: str | None = None) -> str:
    root = base_dir or BASE_DIR
    return os.path.join(
        root,
        f"val_indices_angles_pi_2_ns{DATASET_NUM_SAMPLES}_ts{DATASET_TIMESTEPS}.pkl",
    )


def val_2d_metrics_maps_dir(base_dir: str | None = None) -> str:
    root = base_dir or BASE_DIR
    return os.path.join(root, "val_2d_metrics_maps")
