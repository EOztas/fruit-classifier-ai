"""
Yapay Zeka Destekli Meyve Siniflandirici - Veri Seti Modulu

Bu modul, veri seti yukleme ve DataLoader olusturma islemlerini icerir.
Fruits 360 Dataset formati ile uyumludur.
"""

import os
from typing import Tuple, List, Dict, Optional, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .preprocessing import get_train_transforms, get_val_transforms


class FruitDataset(Dataset):
    """
    Meyve gorselleri veri seti sinifi.

    Fruits 360 formatindaki verileri yukler. Klasor yapisi:
    data/
        Training/
            Apple/
            Banana/
            ...
        Test/
            Apple/
            Banana/
            ...

    Attributes:
        root_dir: Veri seti kok dizini.
        transform: Goruntu donusum fonksiyonu.
        class_names: Sinif isimleri listesi.
        class_to_idx: Sinif adi -> indeks eslestirmesi.
        samples: (goruntu_yolu, etiket) listesi.

    Time Complexity:
        - __init__: O(N) - N: toplam goruntu sayisi
        - __getitem__: O(H * W) - Goruntu okuma ve donusum
        - __len__: O(1)
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        FruitDataset sinifini baslat.

        Args:
            root_dir: Goruntulerin bulundugu kok dizin.
            transform: Uygulanacak transform fonksiyonu.
            class_names: Onceden belirlenmis sinif isimleri (opsiyonel).
        """
        self.root_dir = root_dir
        self.transform = transform

        # Sinif isimlerini bul
        if class_names is not None:
            self.class_names = sorted(class_names)
        else:
            self.class_names = self._discover_classes()

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Goruntu yollarini ve etiketleri topla
        self.samples = self._load_samples()

    def _discover_classes(self) -> List[str]:
        """
        Dizin yapisindan sinif isimlerini kes.

        Returns:
            Sirali sinif isimleri listesi.

        Time Complexity: O(C) - C: sinif sayisi
        """
        classes = []
        if os.path.exists(self.root_dir):
            for item in os.listdir(self.root_dir):
                item_path = os.path.join(self.root_dir, item)
                if os.path.isdir(item_path):
                    classes.append(item)
        return sorted(classes)

    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Tum goruntu yollarini ve etiketlerini topla.

        Returns:
            (goruntu_yolu, etiket_indeksi) tuple listesi.

        Time Complexity: O(N) - N: toplam goruntu sayisi
        """
        samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_idx = self.class_to_idx[class_name]

            for filename in os.listdir(class_dir):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in self.SUPPORTED_FORMATS:
                    file_path = os.path.join(class_dir, filename)
                    samples.append((file_path, class_idx))

        return samples

    def __len__(self) -> int:
        """Veri seti boyutunu dondur."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Belirtilen indeksteki ornegi dondur.

        Args:
            idx: Ornek indeksi.

        Returns:
            (goruntu_tensor, etiket) tuple.
        """
        img_path, label = self.samples[idx]

        # Goruntuyu yukle
        image = Image.open(img_path).convert("RGB")

        # Transform uygula
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """
        Indeksten sinif ismini dondur.

        Args:
            idx: Sinif indeksi.

        Returns:
            Sinif ismi.
        """
        return self.class_names[idx]

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Sinif dagilimini hesapla.

        Returns:
            Sinif adi -> ornek sayisi eslestirmesi.

        Time Complexity: O(N) - N: toplam ornek sayisi
        """
        distribution = {name: 0 for name in self.class_names}
        for _, label_idx in self.samples:
            class_name = self.class_names[label_idx]
            distribution[class_name] += 1
        return distribution


def create_dataloaders(
    config,
    data_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Egitim, dogrulama ve test DataLoader'lari olustur.

    Args:
        config: Config nesnesi.
        data_path: Veri seti yolu (opsiyonel, config'den alinir).

    Returns:
        (train_loader, val_loader, test_loader, class_names) tuple.

    Time Complexity: O(N) - N: toplam goruntu sayisi
    """
    if data_path is None:
        data_path = config.data_dir

    # Fruits 360 veri setinin iç yapısını kontrol et
    # Veri seti şu yapıya sahip olabilir:
    # data/fruits-360_original-size/fruits-360-original-size/Training/
    # data/fruits-360_original-size/fruits-360-original-size/Test/
    # data/Training/
    # data/Test/

    train_path = os.path.join(data_path, "Training")
    test_path = os.path.join(data_path, "Test")
    val_path = os.path.join(data_path, "Validation")

    # Iç klasörleri kontrol et
    for subdir in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir)
        if os.path.isdir(subdir_path):
            potential_train = os.path.join(subdir_path, "Training")
            potential_test = os.path.join(subdir_path, "Test")
            if os.path.exists(potential_train):
                train_path = potential_train
                test_path = potential_test if os.path.exists(potential_test) else None
                val_path = os.path.join(subdir_path, "Validation")
                break

    # Eger Training/Test klasorleri yoksa, tek klasor olarak kabul et
    if not os.path.exists(train_path):
        train_path = data_path
    if not os.path.exists(test_path):
        test_path = None
    if not os.path.exists(val_path):
        val_path = None

    # Transform'lari olustur
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)

    # Egitim veri setini yukle
    full_train_dataset = FruitDataset(
        root_dir=train_path,
        transform=None,  # Sonra ayri ayri uygulayacagiz
    )

    class_names = full_train_dataset.class_names

    # Test veri setini yukle (varsa)
    if test_path and os.path.exists(test_path):
        test_dataset = FruitDataset(
            root_dir=test_path,
            transform=val_transforms,
            class_names=class_names,
        )

        # Egitim verisini train ve val olarak bol
        train_size = int(len(full_train_dataset) * (config.train_split / (config.train_split + config.val_split)))
        val_size = len(full_train_dataset) - train_size

        train_indices, val_indices = random_split(
            range(len(full_train_dataset)),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.random_seed)
        )

        train_dataset = SubsetWithTransform(full_train_dataset, train_indices.indices, train_transforms)
        val_dataset = SubsetWithTransform(full_train_dataset, val_indices.indices, val_transforms)

    else:
        # Tek klasorden train/val/test bol
        total_size = len(full_train_dataset)
        train_size = int(total_size * config.train_split)
        val_size = int(total_size * config.val_split)
        test_size = total_size - train_size - val_size

        train_indices, val_indices, test_indices = random_split(
            range(total_size),
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.random_seed)
        )

        train_dataset = SubsetWithTransform(full_train_dataset, train_indices.indices, train_transforms)
        val_dataset = SubsetWithTransform(full_train_dataset, val_indices.indices, val_transforms)
        test_dataset = SubsetWithTransform(full_train_dataset, test_indices.indices, val_transforms)

    # DataLoader'lari olustur
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


class SubsetWithTransform(Dataset):
    """
    Transform destekli Subset sinifi.

    Bu sinif, belirli indekslerdeki orneklere farkli transform
    uygulamak icin kullanilir.

    Time Complexity:
        - __getitem__: O(H * W) - Goruntu okuma ve donusum
    """

    def __init__(
        self,
        dataset: FruitDataset,
        indices: List[int],
        transform: Optional[Callable] = None
    ):
        """
        SubsetWithTransform sinifini baslat.

        Args:
            dataset: Kaynak veri seti.
            indices: Kullanilacak indeksler.
            transform: Uygulanacak transform.
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        img_path, label = self.dataset.samples[original_idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_sample_images(
    dataset: FruitDataset,
    num_samples: int = 5,
    class_name: Optional[str] = None
) -> List[Tuple[Image.Image, str]]:
    """
    Veri setinden ornek goruntuler al.

    Args:
        dataset: FruitDataset nesnesi.
        num_samples: Alinacak ornek sayisi.
        class_name: Belirli bir siniftan mi alinacak (opsiyonel).

    Returns:
        (goruntu, sinif_adi) tuple listesi.
    """
    samples = []
    indices = list(range(len(dataset)))

    if class_name is not None:
        class_idx = dataset.class_to_idx.get(class_name)
        if class_idx is not None:
            indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]

    import random
    random.shuffle(indices)

    for idx in indices[:num_samples]:
        img_path, label = dataset.samples[idx]
        image = Image.open(img_path).convert("RGB")
        class_label = dataset.class_names[label]
        samples.append((image, class_label))

    return samples
