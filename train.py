"""
Yapay Zeka Destekli Meyve Siniflandirici - Egitim Scripti

Bu script, modelin egitilmesi, degerlendirilmesi ve kaydedilmesi islemlerini yapar.
Accuracy, Precision, Recall, F1-Score metrikleri ile performans olcumu yapar.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Proje modullerini import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from src.dataset import create_dataloaders
from src.model import create_model, save_model, get_model_summary


class Trainer:
    """
    Model egitim sinifi.

    Bu sinif, modelin egitilmesi, degerlendirilmesi ve
    performans metriklerinin hesaplanmasindan sorumludur.

    Attributes:
        model: Egitilecek PyTorch modeli.
        config: Yapilandirma nesnesi.
        device: Hesaplama cihazi.
        criterion: Kayip fonksiyonu.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        history: Egitim gecmisi.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ):
        """
        Trainer sinifini baslat.

        Args:
            model: Egitilecek model.
            config: Yapilandirma nesnesi.
            train_loader: Egitim DataLoader.
            val_loader: Dogrulama DataLoader.
            test_loader: Test DataLoader.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Cihazi belirle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Kayip fonksiyonu
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        # Egitim gecmisi
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        # En iyi model takibi
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> Tuple[float, float]:
        """
        Bir epoch egitim yap.

        Returns:
            (ortalama_kayip, dogruluk) tuple.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Egitim", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Ileri gecis
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Geri yayilim
            loss.backward()
            self.optimizer.step()

            # Istatistikleri guncelle
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Progress bar guncelle
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self, loader: DataLoader, desc: str = "Dogrulama") -> Tuple[float, float]:
        """
        Dogrulama veya test yap.

        Args:
            loader: DataLoader.
            desc: Progress bar aciklamasi.

        Returns:
            (ortalama_kayip, dogruluk) tuple.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(loader, desc=desc, leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Tam egitim dongusu.

        Args:
            num_epochs: Epoch sayisi (opsiyonel, config'den alinir).

        Returns:
            Egitim gecmisi.
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print(f"\n{'='*60}")
        print(f"Egitim Baslatiliyor")
        print(f"{'='*60}")
        print(f"Cihaz: {self.device}")
        print(f"Model: {self.config.model_name}")
        print(f"Epoch Sayisi: {num_epochs}")
        print(f"Batch Boyutu: {self.config.batch_size}")
        print(f"Ogrenme Orani: {self.config.learning_rate}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # Egitim
            train_loss, train_acc = self.train_epoch()

            # Dogrulama
            val_loss, val_acc = self.validate(self.val_loader)

            # Scheduler guncelle
            self.scheduler.step(val_loss)

            # Gecmisi kaydet
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            # Sonuclari yazdir
            print(f"Egitim - Kayip: {train_loss:.4f}, Dogruluk: {train_acc:.4f}")
            print(f"Dogrulama - Kayip: {val_loss:.4f}, Dogruluk: {val_acc:.4f}")
            print(f"Ogrenme Orani: {current_lr:.6f}")

            # En iyi modeli kaydet
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                save_model(
                    self.model,
                    self.config,
                    self.optimizer,
                    epoch + 1,
                    {"val_acc": val_acc, "val_loss": val_loss},
                )
                print(f"*** En iyi model kaydedildi! (Dogruluk: {val_acc:.4f}) ***")

        print(f"\n{'='*60}")
        print(f"Egitim Tamamlandi!")
        print(f"En Iyi Epoch: {self.best_epoch}")
        print(f"En Iyi Dogrulama Dogrulugu: {self.best_val_acc:.4f}")
        print(f"{'='*60}\n")

        return self.history

    def evaluate(self) -> Dict:
        """
        Test seti uzerinde detayli degerlendirme yap.

        Returns:
            Degerlendirme metrikleri.
        """
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Test"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Metrikleri hesapla
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

        # Sinif bazli rapor
        class_report = classification_report(
            all_labels,
            all_predictions,
            target_names=self.config.class_names,
            output_dict=True,
            zero_division=0,
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
        }

        # Sonuclari yazdir
        print(f"\n{'='*60}")
        print("TEST SONUCLARI")
        print(f"{'='*60}")
        print(f"Dogruluk (Accuracy): {accuracy:.4f}")
        print(f"Kesinlik (Precision): {precision:.4f}")
        print(f"Duyarlilik (Recall): {recall:.4f}")
        print(f"F1-Skor: {f1:.4f}")
        print(f"{'='*60}\n")

        return metrics

    def plot_training_history(self, save_path: str = "training_history.png") -> None:
        """
        Egitim gecmisini gorsellestin.

        Args:
            save_path: Grafik kayit yolu.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Kayip grafigi
        axes[0].plot(self.history["train_loss"], label="Egitim Kaybi", marker="o")
        axes[0].plot(self.history["val_loss"], label="Dogrulama Kaybi", marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Kayip")
        axes[0].set_title("Egitim ve Dogrulama Kaybi")
        axes[0].legend()
        axes[0].grid(True)

        # Dogruluk grafigi
        axes[1].plot(self.history["train_acc"], label="Egitim Dogrulugu", marker="o")
        axes[1].plot(self.history["val_acc"], label="Dogrulama Dogrulugu", marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dogruluk")
        axes[1].set_title("Egitim ve Dogrulama Dogrulugu")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Egitim grafigi kaydedildi: {save_path}")

    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        save_path: str = "confusion_matrix.png"
    ) -> None:
        """
        Confusion matrix gorsellestin.

        Args:
            conf_matrix: Confusion matrix.
            save_path: Grafik kayit yolu.
        """
        # Sinif sayisina gore boyut ayarla
        num_classes = len(self.config.class_names)
        fig_size = max(10, num_classes // 3)

        plt.figure(figsize=(fig_size, fig_size))

        # Cok sinif varsa etiketleri gosterme
        if num_classes > 20:
            sns.heatmap(
                conf_matrix,
                cmap="Blues",
                fmt="d",
                cbar=True,
            )
            plt.title("Confusion Matrix (Sinif sayisi fazla oldugu icin etiketler gizlendi)")
        else:
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.config.class_names,
                yticklabels=self.config.class_names,
            )
            plt.title("Confusion Matrix")

        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gercek")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix kaydedildi: {save_path}")


def main():
    """Ana egitim fonksiyonu."""
    parser = argparse.ArgumentParser(description="Meyve Siniflandirici Egitimi")
    parser.add_argument("--data_dir", type=str, default="data_simple", help="Veri seti dizini")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model mimarisi")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch boyutu")
    parser.add_argument("--epochs", type=int, default=20, help="Epoch sayisi")
    parser.add_argument("--lr", type=float, default=0.001, help="Ogrenme orani")
    parser.add_argument("--no_pretrained", action="store_true", help="Onceden egitilmis agirlik kullanma")
    args = parser.parse_args()

    # Yapilandirma olustur
    config = Config(
        data_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        pretrained=not args.no_pretrained,
    )

    print(f"\n{'='*60}")
    print("MEYVE SINIFLANDIRICI - EGITIM")
    print(f"{'='*60}")
    print(f"Veri Seti: {config.data_dir}")
    print(f"Model: {config.model_name}")
    print(f"Pretrained: {config.pretrained}")
    print(f"{'='*60}\n")

    # Veri yukleyicileri olustur
    print("Veri seti yukleniyor...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(config)

    # Sinif isimlerini guncelle
    config.update_class_names(class_names)

    print(f"Sinif Sayisi: {config.num_classes}")
    print(f"Egitim Ornekleri: {len(train_loader.dataset)}")
    print(f"Dogrulama Ornekleri: {len(val_loader.dataset)}")
    print(f"Test Ornekleri: {len(test_loader.dataset)}")

    # Model olustur
    print("\nModel olusturuluyor...")
    model = create_model(config)

    # Model ozetini yazdir
    summary = get_model_summary(model)
    print(f"Toplam Parametre: {summary['total_parameters']:,}")
    print(f"Egitelebilir Parametre: {summary['trainable_parameters']:,}")

    # Trainer olustur ve egit
    trainer = Trainer(model, config, train_loader, val_loader, test_loader)
    history = trainer.train()

    # Egitim grafiklerini kaydet
    trainer.plot_training_history()

    # Test degerlendirmesi
    print("\nTest seti uzerinde degerlendirme yapiliyor...")
    metrics = trainer.evaluate()

    # Confusion matrix kaydet
    trainer.plot_confusion_matrix(np.array(metrics["confusion_matrix"]))

    # Metrikleri JSON olarak kaydet
    metrics_path = os.path.join(config.model_dir, "metrics.json")
    metrics_to_save = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "best_epoch": trainer.best_epoch,
        "best_val_acc": trainer.best_val_acc,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    print(f"Metrikler kaydedildi: {metrics_path}")

    print(f"\n{'='*60}")
    print("EGITIM TAMAMLANDI!")
    print(f"Model: {config.model_path}")
    print(f"Test Dogrulugu: {metrics['accuracy']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
