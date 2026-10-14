"""Пакет обучения и инференса DirectComptonModel для усреднённых спектров."""

from compton_averaging.dataset import (
    ComptonDataset,
    make_stratified_train_val_subsets,
    stratified_train_val_indices,
)
from compton_averaging.model import DirectComptonModel, LearnableGaussian1d
from compton_averaging.normalizer import DataNormalizer
from compton_averaging.training import train_model

__all__ = [
    "ComptonDataset",
    "DataNormalizer",
    "DirectComptonModel",
    "LearnableGaussian1d",
    "make_stratified_train_val_subsets",
    "stratified_train_val_indices",
    "train_model",
]
