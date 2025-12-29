"""
Fruit Classifier - Yapay Zeka Destekli Görüntü Sınıflandırıcı

Bu paket, meyve görsellerini sınıflandırmak için gerekli modülleri içerir.
"""

from .config import Config
from .preprocessing import create_transforms
from .dataset import FruitDataset, create_dataloaders
from .model import create_model, load_model
from .inference import FruitPredictor

__version__ = "1.0.0"
__author__ = "Student"

__all__ = [
    "Config",
    "create_transforms",
    "FruitDataset",
    "create_dataloaders",
    "create_model",
    "load_model",
    "FruitPredictor",
]
