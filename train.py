#!/usr/bin/env python3
"""Главный скрипт: обучение DirectComptonModel и сохранение модели."""
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from compton_averaging.config import (  # noqa: E402
    DATASET_NUM_SAMPLES,
    DATASET_SEED,
    DATASET_TIMESTEPS,
    dataset_cache_dir,
    model_file,
    normalizer_file,
    val_indices_file,
)
from compton_averaging.dataset import (  # noqa: E402
    ComptonDataset,
    make_stratified_train_val_subsets,
)
from compton_averaging.model import DirectComptonModel  # noqa: E402
from compton_averaging.normalizer import DataNormalizer  # noqa: E402
from compton_averaging.training import train_model  # noqa: E402


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATASET_CACHE_DIR = dataset_cache_dir(BASE_DIR)
    MODEL_FILE = model_file(BASE_DIR)
    NORMALIZER_FILE = normalizer_file(BASE_DIR)
    VAL_INDICES_FILE = val_indices_file(BASE_DIR)

    dataset = ComptonDataset(
        num_samples=DATASET_NUM_SAMPLES,
        timesteps=DATASET_TIMESTEPS,
        cache_dir=DATASET_CACHE_DIR,
        seed=DATASET_SEED,
    )

    if os.path.exists(VAL_INDICES_FILE):
        print(f"Загрузка индексов валидационного датасета из {VAL_INDICES_FILE}...")
        with open(VAL_INDICES_FILE, "rb") as f:
            saved_val_indices = pickle.load(f)
        n = len(dataset)
        if saved_val_indices and all(i < n for i in saved_val_indices):
            val_indices = saved_val_indices
            train_indices = [i for i in range(n) if i not in val_indices]
            train_data = torch.utils.data.Subset(dataset, train_indices)
            val_data = torch.utils.data.Subset(dataset, val_indices)
            print(f"Валидационный датасет загружен: {len(val_data)} примеров")
        else:
            print("Сохранённые индексы не подходят — создаём сбалансированное stratified train/val.")
            train_data, val_data, val_indices = make_stratified_train_val_subsets(
                dataset, train_fraction=0.8, seed=DATASET_SEED
            )
            with open(VAL_INDICES_FILE, "wb") as f:
                pickle.dump(val_indices, f)
            print(f"Индексы валидационного датасета сохранены в {VAL_INDICES_FILE}")
    else:
        print("Создание сбалансированного stratified train/val...")
        train_data, val_data, val_indices = make_stratified_train_val_subsets(
            dataset, train_fraction=0.8, seed=DATASET_SEED
        )
        with open(VAL_INDICES_FILE, "wb") as f:
            pickle.dump(val_indices, f)
        print(f"Индексы валидационного датасета сохранены в {VAL_INDICES_FILE}")

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

    if os.path.exists(NORMALIZER_FILE):
        print(f"Загрузка нормализатора из {NORMALIZER_FILE}...")
        with open(NORMALIZER_FILE, 'rb') as f:
            normalizer = pickle.load(f)
        normalizer.to(device)
    else:
        print("Обучение нормализатора на train датасете...")
        normalizer = DataNormalizer()
        normalizer.fit(train_loader)

        normalizer_cpu = DataNormalizer()
        normalizer_cpu.p_mean = normalizer.p_mean.cpu()
        normalizer_cpu.p_std = normalizer.p_std.cpu()
        normalizer_cpu.y_mean = normalizer.y_mean.cpu()
        normalizer_cpu.y_std = normalizer.y_std.cpu()
        with open(NORMALIZER_FILE, 'wb') as f:
            pickle.dump(normalizer_cpu, f)
        normalizer.to(device)

    model = DirectComptonModel()

    if os.path.exists(MODEL_FILE):
        print(f"Найден файл {MODEL_FILE}. Загрузка...")
        checkpoint = torch.load(MODEL_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    else:
        print("Файл не найден. Обучение...")
        train_model(model, train_loader, val_loader, normalizer, epochs=300, lr=1e-3, save_path=MODEL_FILE)


if __name__ == "__main__":
    main()
