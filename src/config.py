"""
Yapay Zeka Destekli Meyve Siniflandirici - Yapilandirma Modulu

Bu modul, projenin tum yapilandirma ayarlarini icerir.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class Config:
    """
    Proje yapilandirma sinifi.

    Attributes:
        image_size: Goruntulerin yeniden boyutlandirilacagi boyut (genislik, yukseklik).
        num_channels: Goruntu kanal sayisi (RGB icin 3).
        batch_size: Egitim sirasinda kullanilacak batch boyutu.
        num_epochs: Toplam egitim epoch sayisi.
        learning_rate: Optimizer icin ogrenme orani.
        num_workers: DataLoader icin worker sayisi.
        train_split: Egitim verisi orani (0-1 arasi).
        val_split: Dogrulama verisi orani (0-1 arasi).
        random_seed: Tekrarlanabilirlik icin rastgele tohum.
        model_name: Kullanilacak model mimarisi.
        pretrained: Onceden egitilmis agirliklar kullanilsin mi.
        dropout_rate: Dropout orani.
        data_dir: Veri seti dizini.
        model_dir: Model kayit dizini.
        class_names: Sinif isimleri listesi.
        mean: Normalizasyon icin ortalama degerler.
        std: Normalizasyon icin standart sapma degerleri.
    """

    # Goruntu ayarlari
    image_size: Tuple[int, int] = (224, 224)
    num_channels: int = 3

    # Egitim ayarlari
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 0.001
    num_workers: int = 4

    # Veri bolme ayarlari
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split otomatik olarak 1 - train_split - val_split olacak

    # Rastgelelik ayarlari
    random_seed: int = 42

    # Model ayarlari
    model_name: str = "resnet18"
    pretrained: bool = True
    dropout_rate: float = 0.5

    # Dizin ayarlari
    data_dir: str = "data"
    model_dir: str = "models"

    # Sinif isimleri (veri setinden otomatik yuklenecek)
    class_names: List[str] = field(default_factory=list)

    # ImageNet normalizasyon degerleri
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Augmentation ayarlari
    use_augmentation: bool = True
    rotation_degrees: int = 15
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1

    def __post_init__(self):
        """Yapilandirma sonrasi islemler."""
        # Dizinlerin var oldugundan emin ol
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    @property
    def test_split(self) -> float:
        """Test verisi oranini hesapla."""
        return 1.0 - self.train_split - self.val_split

    @property
    def num_classes(self) -> int:
        """Sinif sayisini dondur."""
        return len(self.class_names)

    @property
    def model_path(self) -> str:
        """Model kayit yolunu dondur."""
        return os.path.join(self.model_dir, "best_model.pth")

    def update_class_names(self, class_names: List[str]) -> None:
        """
        Sinif isimlerini guncelle.

        Args:
            class_names: Yeni sinif isimleri listesi.
        """
        self.class_names = sorted(class_names)

    def to_dict(self) -> dict:
        """Yapilandirmayi dictionary olarak dondur."""
        return {
            "image_size": self.image_size,
            "num_channels": self.num_channels,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "num_workers": self.num_workers,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "random_seed": self.random_seed,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "dropout_rate": self.dropout_rate,
            "data_dir": self.data_dir,
            "model_dir": self.model_dir,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "mean": self.mean,
            "std": self.std,
            "use_augmentation": self.use_augmentation,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """
        Dictionary'den Config olustur.

        Args:
            config_dict: Yapilandirma dictionary'si.

        Returns:
            Config nesnesi.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)
