"""Загрузка датасета, модели и нормализатора для скриптов инференса."""
import os
import pickle

import torch
from torch.utils.data import DataLoader

from compton_averaging.config import (
    BASE_DIR,
    DATASET_NUM_SAMPLES,
    DATASET_SEED,
    DATASET_TIMESTEPS,
    dataset_cache_dir,
    model_file,
    normalizer_file,
    val_indices_file,
)
from compton_averaging.dataset import ComptonDataset, make_stratified_train_val_subsets
from compton_averaging.model import DirectComptonModel
from compton_averaging.normalizer import DataNormalizer


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(base_dir: str | None = None) -> ComptonDataset:
    root = base_dir or BASE_DIR
    return ComptonDataset(
        num_samples=DATASET_NUM_SAMPLES,
        timesteps=DATASET_TIMESTEPS,
        cache_dir=dataset_cache_dir(root),
        seed=DATASET_SEED,
    )


def load_train_val_data(
    dataset: ComptonDataset | None = None,
    base_dir: str | None = None,
    train_fraction: float = 0.8,
):
    """Возвращает (train_data, val_data, val_indices)."""
    root = base_dir or BASE_DIR
    if dataset is None:
        dataset = load_dataset(root)

    val_idx_path = val_indices_file(root)
    if os.path.exists(val_idx_path):
        print(f"Загрузка индексов валидационного датасета из {val_idx_path}...")
        with open(val_idx_path, "rb") as f:
            saved_val_indices = pickle.load(f)
        n = len(dataset)
        if saved_val_indices and all(i < n for i in saved_val_indices):
            val_indices = saved_val_indices
            train_indices = [i for i in range(n) if i not in val_indices]
            train_data = torch.utils.data.Subset(dataset, train_indices)
            val_data = torch.utils.data.Subset(dataset, val_indices)
            print(f"Валидационный датасет загружен: {len(val_data)} примеров")
            return train_data, val_data, val_indices

        print("Сохранённые индексы не подходят — создаём сбалансированное stratified train/val.")
    else:
        print("Создание сбалансированного stratified train/val...")

    train_data, val_data, val_indices = make_stratified_train_val_subsets(
        dataset, train_fraction=train_fraction, seed=DATASET_SEED
    )
    with open(val_idx_path, "wb") as f:
        pickle.dump(val_indices, f)
    print(f"Индексы валидационного датасета сохранены в {val_idx_path}")
    return train_data, val_data, val_indices


def load_normalizer(
    device: torch.device,
    train_loader: DataLoader | None = None,
    base_dir: str | None = None,
) -> DataNormalizer:
    root = base_dir or BASE_DIR
    norm_path = normalizer_file(root)

    if os.path.exists(norm_path):
        print(f"Загрузка нормализатора из {norm_path}...")
        with open(norm_path, 'rb') as f:
            normalizer = pickle.load(f)
        normalizer.to(device)
        return normalizer

    if train_loader is None:
        raise FileNotFoundError(
            f"Файл нормализатора не найден: {norm_path}. "
            "Сначала запустите train.py для обучения нормализатора."
        )

    print("Обучение нормализатора на train датасете...")
    normalizer = DataNormalizer()
    normalizer.fit(train_loader)

    normalizer_cpu = DataNormalizer()
    normalizer_cpu.p_mean = normalizer.p_mean.cpu()
    normalizer_cpu.p_std = normalizer.p_std.cpu()
    normalizer_cpu.y_mean = normalizer.y_mean.cpu()
    normalizer_cpu.y_std = normalizer.y_std.cpu()
    with open(norm_path, 'wb') as f:
        pickle.dump(normalizer_cpu, f)
    normalizer.to(device)
    return normalizer


def load_model(
    device: torch.device,
    base_dir: str | None = None,
) -> DirectComptonModel:
    root = base_dir or BASE_DIR
    mpath = model_file(root)

    model = DirectComptonModel()
    if not os.path.exists(mpath):
        raise FileNotFoundError(
            f"Файл модели не найден: {mpath}. Сначала запустите train.py."
        )

    print(f"Загрузка модели из {mpath}...")
    checkpoint = torch.load(mpath, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)
    return model


def load_for_inference(base_dir: str | None = None):
    """Загружает device, model, normalizer, val_data для скриптов оценки."""
    root = base_dir or BASE_DIR
    device = get_device()
    dataset = load_dataset(root)
    _, val_data, _ = load_train_val_data(dataset, root)
    normalizer = load_normalizer(device, base_dir=root)
    model = load_model(device, root)
    return device, model, normalizer, val_data
