# Yapay Zeka Destekli Meyve Siniflandirici

Bu proje, derin ogrenme kullanarak meyve gorsellerini siniflandiran bir yapay zeka uygulamasidir. PyTorch ile egitilen model, Gradio tabanli kullanici dostu bir web arayuzu ile sunulmaktadir.

## Ozellikler

- **Goruntu Siniflandirma**: 130+ meyve turunu tanimlayabilme
- **Transfer Learning**: ResNet, EfficientNet, MobileNet destegi
- **Veri Artirma**: Otomatik augmentation teknikleri
- **Web Arayuzu**: Kullanici dostu Gradio arayuzu
- **Performans Metrikleri**: Accuracy, Precision, Recall, F1-Score

## Kurulum

### Gereksinimler

- Python 3.8+
- CUDA destekli GPU (opsiyonel, CPU ile de calisir)

### Adimlar

1. **Depoyu klonlayin:**
```bash
git clone https://github.com/kullanici/fruit-classifier.git
cd fruit-classifier
```

2. **Sanal ortam olusturun:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Bagimliliklari yukleyin:**
```bash
pip install -r requirements.txt
```

4. **Veri setini indirin:**

[Fruits 360 Dataset](https://www.kaggle.com/datasets/moltean/fruits) veri setini indirin ve `data/` klasorune cikartin:

```
data/
├── Training/
│   ├── Apple Braeburn/
│   ├── Apple Golden 1/
│   ├── Banana/
│   └── ...
└── Test/
    ├── Apple Braeburn/
    ├── Apple Golden 1/
    ├── Banana/
    └── ...
```

## Kullanim

### Model Egitimi

```bash
python train.py --data_dir data --epochs 20 --batch_size 32
```

**Egitim Parametreleri:**
| Parametre | Varsayilan | Aciklama |
|-----------|------------|----------|
| `--data_dir` | `data` | Veri seti dizini |
| `--model_name` | `resnet18` | Model mimarisi |
| `--batch_size` | `32` | Batch boyutu |
| `--epochs` | `20` | Epoch sayisi |
| `--lr` | `0.001` | Ogrenme orani |
| `--no_pretrained` | `False` | Onceden egitilmis agirlik kullanma |

**Desteklenen Modeller:**
- `resnet18`, `resnet34`, `resnet50`
- `efficientnet_b0`, `efficientnet_b1`
- `mobilenet_v2`, `mobilenet_v3_small`
- `custom_cnn` (ozel CNN mimarisi)

### Web Arayuzunu Baslatma

```bash
python app.py
```

Tarayicinizda `http://localhost:7860` adresine gidin.

### Programatik Kullanim

```python
from src.inference import FruitPredictor

# Predictor olustur
predictor = FruitPredictor("models/best_model.pth")

# Tahmin yap
result = predictor.predict("meyve_resmi.jpg")

print(f"Tahmin: {result['predicted_class']}")
print(f"Guven: {result['confidence']:.2%}")
```

## Proje Yapisi

```
fruit-classifier/
├── data/                   # Veri seti dizini
│   ├── Training/          # Egitim verileri
│   └── Test/              # Test verileri
├── models/                 # Kaydedilen modeller
│   └── best_model.pth     # En iyi model
├── src/                    # Kaynak kod
│   ├── __init__.py        # Paket baslangici
│   ├── config.py          # Yapilandirma sinifi
│   ├── preprocessing.py   # Goruntu on isleme
│   ├── dataset.py         # Veri seti sinifi
│   ├── model.py           # Model tanimlari
│   └── inference.py       # Tahmin sinifi
├── train.py               # Egitim scripti
├── app.py                 # Gradio web arayuzu
├── requirements.txt       # Bagimliliklar
├── .gitignore            # Git ignore dosyasi
└── README.md             # Bu dosya
```

## Teknik Detaylar

### On Isleme

Goruntuler su adimlardan gecer:
1. **Yeniden Boyutlandirma**: 224x224 piksel
2. **Normalizasyon**: ImageNet ortalama ve standart sapma degerleri
3. **Augmentation** (egitim sirasinda):
   - Rastgele kesme (Random Crop)
   - Yatay cevirme (Horizontal Flip)
   - Renk degisimi (Color Jitter)
   - Rastgele donme (Random Rotation)

### Model Mimarisi

Varsayilan olarak **ResNet-18** transfer learning modeli kullanilmaktadir:
- Onceden egitilmis ImageNet agirliklari
- Son siniflandirma katmani meyve siniflarina uyarlanmis
- Dropout ile asiri ogrenmeden korunma

### Performans Metrikleri

Egitim sonunda asagidaki metrikler hesaplanir:
- **Accuracy**: Genel dogruluk orani
- **Precision**: Kesinlik (yanlis pozitif orani)
- **Recall**: Duyarlilik (yanlis negatif orani)
- **F1-Score**: Precision ve Recall'un harmonik ortalamasi

Metrikler `models/metrics.json` dosyasina kaydedilir.

## Egitim Sonuclari

| Metrik | Deger |
|--------|-------|
| Accuracy | %94.20 |
| Precision | %94+ |
| Recall | %94+ |
| F1-Score | %94+ |

**ÖNEMLI NOT:** Bu model **Fruits 360 Dataset** ile egitilmistir. En iyi sonuc icin:
- ✅ Beyaz arka planli goruntular kullanin
- ✅ Meyveyi merkeze yerlestirin
- ✅ Iyi isiklandirma saglayin
- ❌ Dogal arka plan, karmasik sahneler daha dusuk dogruluk verebilir

**Test Icin:** `data_simple/Test/` klasorundeki gorselleri kullanin.

*Not: Gercek dunya gorselleri icin daha fazla epoch ve data augmentation gerekebilir.*

## Ekran Goruntuleri

### Web Arayuzu

![Web Arayuzu](screenshots/Ekran%20görüntüsü%202025-12-28%20235921.png)

Gradio tabanli kullanici dostu arayuz ile:
- Gorsel yukleme alani
- Otomatik tahmin
- Top-5 sonuc gosterimi
- Olasilik dagilimi grafigi

### Egitim Grafikleri

Egitim sonrasinda otomatik olusturulan dosyalar:
- `training_history.png` - Egitim/dogrulama kaybi ve dogrulugu grafikleri
- `confusion_matrix.png` - Siniflar arasi karisiklik matrisi
- `models/metrics.json` - Detayli performans metrikleri

### Demo Video

[![Proje Tanıtım Videosu](https://img.youtube.com/vi/aVbCoQfeeVM/0.jpg)](https://youtu.be/aVbCoQfeeVM)

**Video İçeriği:**
- Web arayüzü kullanımı
- Meyve görseli yükleme ve sınıflandırma
- Tahmin sonuçları ve güven skorları
- Olasılık dağılımı grafiği

## Lisans

Bu proje MIT lisansi altinda lisanslanmistir.

## Katki

Katkilarinizi bekliyoruz! Pull request gondermeden once lutfen:
1. Yeni bir branch olusturun
2. Degisikliklerinizi test edin
3. Aciklayici bir commit mesaji yazin

## Iletisim

Sorulariniz icin issue acabilirsiniz.

---

*Bu proje, yapay zeka ve derin ogrenme egitimi kapsaminda gelistirilmistir.*
