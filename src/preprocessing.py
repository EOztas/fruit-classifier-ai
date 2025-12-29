"""
Yapay Zeka Destekli Meyve Siniflandirici - On Isleme Modulu

Bu modul, goruntu on isleme ve veri artirma (augmentation) islemlerini icerir.
"""

from typing import Tuple, Optional, Callable
from PIL import Image
import numpy as np

import torch
from torchvision import transforms


class ImagePreprocessor:
    """
    Goruntu on isleme sinifi.

    Bu sinif, gorsellerin normalize edilmesi, yeniden boyutlandirilmasi
    ve augmentation tekniklerinin uygulanmasindan sorumludur.

    Attributes:
        image_size: Hedef goruntu boyutu (genislik, yukseklik).
        mean: Normalizasyon icin ortalama degerler.
        std: Normalizasyon icin standart sapma degerleri.
        use_augmentation: Augmentation kullanilip kullanilmayacagi.

    Time Complexity:
        - transform: O(H * W) - Goruntu boyutuna bagli lineer
        - normalize: O(H * W * C) - Piksel basina islem
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_augmentation: bool = False,
        rotation_degrees: int = 15,
        horizontal_flip_prob: float = 0.5,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
    ):
        """
        ImagePreprocessor sinifini baslat.

        Args:
            image_size: Hedef goruntu boyutu.
            mean: Normalizasyon ortalamasi.
            std: Normalizasyon standart sapmasi.
            use_augmentation: Augmentation kullanimi.
            rotation_degrees: Rastgele donme derecesi.
            horizontal_flip_prob: Yatay cevirme olasiligi.
            color_jitter_brightness: Parlaklik degisim miktari.
            color_jitter_contrast: Kontrast degisim miktari.
            color_jitter_saturation: Doygunluk degisim miktari.
            color_jitter_hue: Renk tonu degisim miktari.
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_augmentation = use_augmentation
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue

        self._transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """
        Transform pipeline olustur.

        Returns:
            Compose transform nesnesi.
        """
        transform_list = []

        if self.use_augmentation:
            # Egitim icin augmentation
            transform_list.extend([
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                transforms.RandomRotation(degrees=self.rotation_degrees),
                transforms.ColorJitter(
                    brightness=self.color_jitter_brightness,
                    contrast=self.color_jitter_contrast,
                    saturation=self.color_jitter_saturation,
                    hue=self.color_jitter_hue
                ),
            ])
        else:
            # Cikarsama icin basit boyutlandirma
            transform_list.extend([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
            ])

        # Ortak donusumler
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        return transforms.Compose(transform_list)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Goruntuyu on isle.

        Args:
            image: PIL Image nesnesi.

        Returns:
            On islenmis tensor.
        """
        return self._transform(image)

    def preprocess_single(self, image: Image.Image) -> torch.Tensor:
        """
        Tek bir goruntuyu cikarsama icin on isle.

        Args:
            image: PIL Image nesnesi.

        Returns:
            Batch boyutuna sahip tensor (1, C, H, W).
        """
        # RGB'ye donustur
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self._transform(image)
        return tensor.unsqueeze(0)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Normalize edilmis tensoru goruntuye geri donustur.

        Args:
            tensor: Normalize edilmis tensor.

        Returns:
            Numpy array olarak goruntu (0-255 arasi).
        """
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)

        tensor = tensor.clone()
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)

        # CHW -> HWC ve numpy'a donustur
        image = tensor.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)

        return image


def create_transforms(
    config,
    is_training: bool = True
) -> Callable:
    """
    Yapilandirmaya gore transform olustur.

    Args:
        config: Config nesnesi.
        is_training: Egitim modu mu.

    Returns:
        Transform fonksiyonu.

    Time Complexity: O(1) - Sabit zamanli olusturma
    """
    preprocessor = ImagePreprocessor(
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        use_augmentation=is_training and config.use_augmentation,
        rotation_degrees=config.rotation_degrees,
        horizontal_flip_prob=config.horizontal_flip_prob,
        color_jitter_brightness=config.color_jitter_brightness,
        color_jitter_contrast=config.color_jitter_contrast,
        color_jitter_saturation=config.color_jitter_saturation,
        color_jitter_hue=config.color_jitter_hue,
    )
    return preprocessor


def get_train_transforms(config) -> transforms.Compose:
    """
    Egitim icin transform pipeline dondur.

    Args:
        config: Config nesnesi.

    Returns:
        Egitim transformlari.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            config.image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
        transforms.RandomRotation(degrees=config.rotation_degrees),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def get_val_transforms(config) -> transforms.Compose:
    """
    Dogrulama/test icin transform pipeline dondur.

    Args:
        config: Config nesnesi.

    Returns:
        Dogrulama transformlari.
    """
    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def get_inference_transforms(config) -> transforms.Compose:
    """
    Cikarsama (inference) icin transform pipeline dondur.

    Args:
        config: Config nesnesi.

    Returns:
        Cikarsama transformlari.
    """
    return get_val_transforms(config)
