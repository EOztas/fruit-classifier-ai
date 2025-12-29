"""
Yapay Zeka Destekli Meyve Siniflandirici - Web Arayuzu

Bu script, Gradio kullanarak kullanici dostu bir web arayuzu olusturur.
Kullanicilar gorsel yukleyerek meyve siniflandirmasi yapabilir.
"""

import os
import sys

import gradio as gr
import numpy as np
from PIL import Image

# Proje modullerini import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference import FruitPredictor


# Global predictor nesnesi
predictor = None


def load_predictor(model_path: str = "models/best_model.pth") -> bool:
    """
    Model yukleyiciyi baslat.

    Args:
        model_path: Model dosya yolu.

    Returns:
        Basarili mi.
    """
    global predictor

    if not os.path.exists(model_path):
        return False

    try:
        predictor = FruitPredictor(model_path)
        return True
    except Exception as e:
        print(f"Model yukleme hatasi: {e}")
        return False


def predict_fruit(image: Image.Image) -> tuple:
    """
    Yuklenen goruntu icin tahmin yap.

    Args:
        image: PIL Image nesnesi.

    Returns:
        (tahmin_sonucu, olasilik_grafigi) tuple.
    """
    if predictor is None:
        return "Model yuklenmedi! Lutfen oncelikle modeli egitiniz.", None

    if image is None:
        return "Lutfen bir goruntu yukleyiniz.", None

    try:
        # Tahmin yap
        print(f"\n[TAHMIN] Gorsel alindi: {type(image)}, Size: {image.size if hasattr(image, 'size') else 'unknown'}")
        result = predictor.predict(image, top_k=5)

        # Sonuc metni olustur
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        print(f"[SONUC] {predicted_class} ({confidence:.2%})")

        result_text = f"**Tahmin Edilen Meyve:** {predicted_class}\n"
        result_text += f"**Guven Skoru:** {confidence:.2%}\n\n"
        result_text += "### En Yuksek 5 Tahmin:\n"

        for i, pred in enumerate(result["top_predictions"], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            result_text += f"{emoji} **{i}.** {pred['class']}: {pred['probability']:.2%}\n"

        # Olasilik dagilimi (bar chart icin)
        labels = [pred["class"] for pred in result["top_predictions"]]
        values = [pred["probability"] for pred in result["top_predictions"]]

        return result_text, dict(zip(labels, values))

    except Exception as e:
        return f"Tahmin sirasinda hata olustu: {str(e)}", None


def create_demo() -> gr.Blocks:
    """
    Gradio demo arayuzunu olustur.

    Returns:
        Gradio Blocks nesnesi.
    """
    with gr.Blocks(title="Meyve Siniflandirici") as demo:

        # Baslik
        gr.Markdown(
            """
            # 🍎 Yapay Zeka Destekli Meyve Siniflandirici

            Bu uygulama, yuklediginiz meyve gorsellerini yapay zeka kullanarak siniflandirir.

            **Kullanim:**
            1. Asagidaki alana bir meyve gorseli yukleyin
            2. "Tahmin Et" butonuna tiklayin
            3. Sonuclari gorun!
            """,
            elem_classes=["header"]
        )

        with gr.Row():
            # Sol panel - Goruntu yukleme
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Goruntu Yukle")

                image_input = gr.Image(
                    type="pil",
                    label="Meyve Gorseli",
                    height=350,
                    sources=["upload", "clipboard"],
                )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Temizle", variant="secondary")
                    predict_btn = gr.Button("🔍 Tahmin Et", variant="primary", size="lg")

                # Ornek gorseller
                gr.Markdown("### 📁 Ornek Gorseller")
                gr.Markdown("*Ornek gorseller icin `data/` klasorune meyve resimleri ekleyebilirsiniz.*")

            # Sag panel - Sonuclar
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Tahmin Sonuclari")

                result_text = gr.Markdown(
                    value="Sonuclar burada gorunecek...",
                    elem_classes=["result-box"]
                )

                probability_chart = gr.BarPlot(
                    x="Sinif",
                    y="Olasilik",
                    title="Olasilik Dagilimi",
                    height=300,
                )

        # Alt bilgi
        gr.Markdown(
            """
            ---
            ### 📖 Hakkinda

            Bu proje, **PyTorch** ve **Gradio** kullanilarak gelistirilmistir.
            Model, transfer learning (ResNet) kullanilarak egitilmistir.

            **Desteklenen Formatlar:** JPG, JPEG, PNG, BMP, GIF

            ---
            *Gelistirici: Yapay Zeka Ogrencisi | 2024*
            """
        )

        # Event handlers
        def on_predict(image):
            result, probs = predict_fruit(image)
            if probs:
                # BarPlot icin veri formati
                import pandas as pd
                df = pd.DataFrame({
                    "Sinif": list(probs.keys()),
                    "Olasilik": list(probs.values())
                })
                return result, df
            return result, None

        predict_btn.click(
            fn=on_predict,
            inputs=[image_input],
            outputs=[result_text, probability_chart],
        )

        image_input.change(
            fn=on_predict,
            inputs=[image_input],
            outputs=[result_text, probability_chart],
        )

        clear_btn.click(
            fn=lambda: (None, "Sonuclar burada gorunecek...", None),
            inputs=[],
            outputs=[image_input, result_text, probability_chart],
        )

    return demo


def main():
    """Ana fonksiyon - uygulamayi baslat."""
    print("\n" + "=" * 60)
    print("MEYVE SINIFLANDIRICI - WEB ARAYUZU")
    print("=" * 60)

    # Modeli yukle
    model_path = "models/best_model.pth"
    print(f"\nModel yukleniyor: {model_path}")

    if load_predictor(model_path):
        print(f"Model basariyla yuklendi!")
        print(f"Sinif Sayisi: {predictor.get_num_classes()}")
        print(f"Ilk 5 sinif: {predictor.get_class_names()[:5]}")

        # Test et
        import os
        test_img = "data_simple/Test/Apple 10/r0_103.jpg"
        if os.path.exists(test_img):
            test_result = predictor.predict(test_img, top_k=1)
            print(f"Test: {test_img} -> {test_result['predicted_class']} ({test_result['confidence']:.2%})")
    else:
        print(f"UYARI: Model bulunamadi!")
        print(f"Lutfen oncelikle 'python train.py' komutu ile modeli egitiniz.")
        print(f"Uygulama model olmadan da calisacak, ancak tahmin yapilamayacak.")

    # Gradio arayuzunu olustur ve baslat
    print("\nWeb arayuzu baslatiliyor...")
    demo = create_demo()

    # Uygulamayi baslat
    demo.launch(
        server_name="127.0.0.1",  # localhost only
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
