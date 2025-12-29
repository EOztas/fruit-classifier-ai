"""
Yapay Zeka Destekli Meyve Siniflandirici - Cikarsama Modulu

Bu modul, egitilmis model ile tahmin yapma islemlerini icerir.
"""

import os
from typing import Tuple, List, Dict, Optional, Union
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import load_model
from .preprocessing import get_inference_transforms
from .config import Config


class FruitPredictor:
    """
    Meyve siniflandirma tahmini sinifi.

    Egitilmis bir modeli kullanarak gorseller uzerinde
    siniflandirma tahmini yapar.

    Attributes:
        model: Egitilmis PyTorch modeli.
        class_names: Sinif isimleri listesi.
        device: Hesaplama cihazi (CPU/GPU).
        transform: Goruntu on isleme fonksiyonu.

    Time Complexity:
        - predict: O(H * W) + O(model_forward)
        - predict_batch: O(B * H * W) + O(B * model_forward)
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
    ):
        """
        FruitPredictor sinifini baslat.

        Args:
            model_path: Egitilmis model dosya yolu.
            device: Hesaplama cihazi (otomatik secim icin None).

        Raises:
            FileNotFoundError: Model dosyasi bulunamazsa.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_path}")

        # Cihazi belirle
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Modeli yukle
        self.model, config_dict, self.class_names = load_model(model_path, self.device)
        self.model.eval()

        # Transform olustur
        self.config = Config.from_dict(config_dict)
        self.transform = get_inference_transforms(self.config)

    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        top_k: int = 5,
    ) -> Dict[str, any]:
        """
        Tek bir goruntu icin tahmin yap.

        Args:
            image: Goruntu (dosya yolu, PIL Image veya numpy array).
            top_k: Dondurulecek en yuksek k tahmin.

        Returns:
            Tahmin sonuclari:
                - predicted_class: Tahmin edilen sinif ismi.
                - confidence: Tahmin guven skoru (0-1).
                - top_predictions: En yuksek k tahmin listesi.
                - probabilities: Tum siniflar icin olasiliklar.
        """
        # Goruntuyu yukle ve on isle
        image_tensor = self._prepare_image(image)

        # Tahmin yap
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        # Sonuclari isle
        probs = probabilities.cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        top_predictions = [
            {
                "class": self.class_names[idx],
                "probability": float(probs[idx]),
            }
            for idx in top_indices
        ]

        predicted_idx = top_indices[0]

        return {
            "predicted_class": self.class_names[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "top_predictions": top_predictions,
            "probabilities": {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            },
        }

    def predict_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        top_k: int = 5,
    ) -> List[Dict[str, any]]:
        """
        Birden fazla goruntu icin tahmin yap.

        Args:
            images: Goruntu listesi.
            top_k: Her goruntu icin dondurulecek en yuksek k tahmin.

        Returns:
            Tahmin sonuclari listesi.

        Time Complexity: O(B * (H * W + model_forward))
        """
        results = []
        for image in images:
            result = self.predict(image, top_k)
            results.append(result)
        return results

    def _prepare_image(
        self,
        image: Union[str, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Goruntuyu model girdisi icin hazirla.

        Args:
            image: Ham goruntu (cesitli formatlar).

        Returns:
            Hazirlanmis tensor (1, C, H, W).
        """
        # Dosya yolundan yukle
        if isinstance(image, str):
            image = Image.open(image)

        # Numpy array'den donustur
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # RGB'ye donustur
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Transform uygula ve batch boyutu ekle
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def get_class_names(self) -> List[str]:
        """Sinif isimlerini dondur."""
        return self.class_names

    def get_num_classes(self) -> int:
        """Sinif sayisini dondur."""
        return len(self.class_names)


def quick_predict(
    image_path: str,
    model_path: str = "models/best_model.pth",
) -> Tuple[str, float]:
    """
    Hizli tahmin fonksiyonu.

    Args:
        image_path: Goruntu dosya yolu.
        model_path: Model dosya yolu.

    Returns:
        (sinif_adi, guven_skoru) tuple.
    """
    predictor = FruitPredictor(model_path)
    result = predictor.predict(image_path)
    return result["predicted_class"], result["confidence"]


def format_prediction_result(result: Dict) -> str:
    """
    Tahmin sonucunu okunabilir formata donustur.

    Args:
        result: predict() fonksiyonundan donen sonuc.

    Returns:
        Formatlanmis metin.
    """
    lines = []
    lines.append(f"Tahmin: {result['predicted_class']}")
    lines.append(f"Guven: {result['confidence']:.2%}")
    lines.append("\nEn Yuksek 5 Tahmin:")
    lines.append("-" * 30)

    for i, pred in enumerate(result["top_predictions"], 1):
        lines.append(f"{i}. {pred['class']}: {pred['probability']:.2%}")

    return "\n".join(lines)
