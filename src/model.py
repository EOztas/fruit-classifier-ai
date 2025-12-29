"""
Yapay Zeka Destekli Meyve Siniflandirici - Model Modulu

Bu modul, CNN model tanimlarini ve model yukleme/kaydetme islemlerini icerir.
Transfer learning destekli ResNet, EfficientNet ve ozel CNN modelleri sunar.
"""

import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torchvision import models


class FruitClassifierCNN(nn.Module):
    """
    Ozel CNN modeli meyve siniflandirmasi icin.

    Basit ve etkili bir CNN mimarisi. Kucuk veri setleri veya
    hizli egitim gerektiren durumlar icin uygundur.

    Architecture:
        - 4 Convolutional block (Conv2d + BatchNorm + ReLU + MaxPool)
        - Global Average Pooling
        - Fully Connected layers with Dropout

    Attributes:
        num_classes: Sinif sayisi.
        dropout_rate: Dropout orani.
    """

    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        """
        FruitClassifierCNN sinifini baslat.

        Args:
            num_classes: Cikis sinif sayisi.
            dropout_rate: Dropout orani.
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 14x14x256 -> 7x7x512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ileri gecis.

        Args:
            x: Giris tensoru (B, 3, H, W).

        Returns:
            Cikis tensoru (B, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class TransferLearningModel(nn.Module):
    """
    Transfer learning tabanli model.

    Onceden egitilmis modelleri (ResNet, EfficientNet vb.) kullanarak
    meyve siniflandirmasi yapar.

    Attributes:
        model_name: Temel model ismi.
        num_classes: Sinif sayisi.
        pretrained: Onceden egitilmis agirliklar kullanilsin mi.
        freeze_backbone: Backbone agirlikları dondurulsun mu.
    """

    SUPPORTED_MODELS = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "mobilenet_v2": models.mobilenet_v2,
        "mobilenet_v3_small": models.mobilenet_v3_small,
    }

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
    ):
        """
        TransferLearningModel sinifini baslat.

        Args:
            model_name: Temel model ismi.
            num_classes: Cikis sinif sayisi.
            pretrained: Onceden egitilmis agirliklar kullanilsin mi.
            freeze_backbone: Feature extractor dondurulsun mu.
            dropout_rate: Dropout orani.
        """
        super().__init__()

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Desteklenmeyen model: {model_name}. "
                f"Desteklenen modeller: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.num_classes = num_classes

        # Temel modeli yukle
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = self.SUPPORTED_MODELS[model_name](weights=weights)

        # Backbone'u dondur (istege bagli)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Son katmani degistir
        self._replace_classifier(dropout_rate)

    def _replace_classifier(self, dropout_rate: float) -> None:
        """Son siniflandirma katmanini degistir."""
        if self.model_name.startswith("resnet"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, self.num_classes),
            )

        elif self.model_name.startswith("efficientnet"):
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, self.num_classes),
            )

        elif self.model_name.startswith("mobilenet"):
            if hasattr(self.backbone, "classifier"):
                if isinstance(self.backbone.classifier, nn.Sequential):
                    in_features = self.backbone.classifier[-1].in_features
                    self.backbone.classifier[-1] = nn.Linear(in_features, self.num_classes)
                else:
                    in_features = self.backbone.classifier.in_features
                    self.backbone.classifier = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(in_features, self.num_classes),
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ileri gecis.

        Args:
            x: Giris tensoru (B, 3, H, W).

        Returns:
            Cikis tensoru (B, num_classes).
        """
        return self.backbone(x)

    def unfreeze_backbone(self) -> None:
        """Backbone katmanlarini egitim icin ac."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(
    config,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """
    Yapilandirmaya gore model olustur.

    Args:
        config: Config nesnesi.
        num_classes: Sinif sayisi (opsiyonel, config'den alinir).

    Returns:
        PyTorch model.
    """
    if num_classes is None:
        num_classes = config.num_classes

    if config.model_name == "custom_cnn":
        model = FruitClassifierCNN(
            num_classes=num_classes,
            dropout_rate=config.dropout_rate,
        )
    else:
        model = TransferLearningModel(
            model_name=config.model_name,
            num_classes=num_classes,
            pretrained=config.pretrained,
            dropout_rate=config.dropout_rate,
        )

    return model


def save_model(
    model: nn.Module,
    config,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    path: Optional[str] = None,
) -> str:
    """
    Modeli diske kaydet.

    Args:
        model: Kaydedilecek model.
        config: Config nesnesi.
        optimizer: Optimizer durumu (opsiyonel).
        epoch: Mevcut epoch (opsiyonel).
        metrics: Egitim metrikleri (opsiyonel).
        path: Kayit yolu (opsiyonel).

    Returns:
        Kayit yolu.
    """
    if path is None:
        path = config.model_path

    # Dizin yoksa olustur
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "class_names": config.class_names,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)
    return path


def load_model(
    path: str,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Modeli diskten yukle.

    Args:
        path: Model dosya yolu.
        device: Hedef cihaz.

    Returns:
        (model, config_dict, class_names) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config_dict = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", [])

    # Config'den model olustur
    from .config import Config
    config = Config.from_dict(config_dict)
    config.class_names = class_names

    model = create_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config_dict, class_names


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    """
    Model ozet bilgilerini dondur.

    Args:
        model: PyTorch model.
        input_size: Ornek giris boyutu.

    Returns:
        Model ozet bilgileri.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "input_size": input_size,
    }
