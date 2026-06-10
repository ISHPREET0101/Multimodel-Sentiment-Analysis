"""
app.py  ─  Gradio Interface for Multimodal Sentiment Analysis
─────────────────────────────────────────────────────────────
Upload a meme image + type/paste its text → get sentiment prediction.

Run:
    python app.py                          # uses best_model.pt if available
    python app.py --model results/best_model.pt
    python app.py --demo                   # runs without trained weights (random init, for UI demo)
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import gradio as gr
from transformers import DistilBertTokenizer
from torchvision import transforms

from models.multimodal_model import build_model

# ── Constants ──────────────────────────────────────────────────────────────────
IDX2LABEL  = {0: "Positive 😊", 1: "Neutral 😐", 2: "Negative 😞"}
LABEL_COLORS = {"Positive 😊": "#2ecc71", "Neutral 😐": "#f39c12", "Negative 😞": "#e74c3c"}
MAX_LEN    = 64

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model(model_path: str, device: str):
    model = build_model(device)
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[App] Loaded weights from {model_path}  (epoch {ckpt.get('epoch','?')})")
    else:
        print("[App] WARNING: No checkpoint found — using random weights (demo mode).")
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(image: Image.Image, text: str, model, tokenizer, device: str) -> dict:
    if image is None:
        return {l: 0.0 for l in IDX2LABEL.values()}

    # Preprocess image
    img_tensor = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    # Tokenize text
    enc = tokenizer(
        [text or ""],
        padding        = True,
        truncation     = True,
        max_length     = MAX_LEN,
        return_tensors = "pt",
    )
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    logits = model(img_tensor, ids, mask)              # [1, 3]
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return {IDX2LABEL[i]: float(probs[i]) for i in range(3)}


# ── Gradio UI ──────────────────────────────────────────────────────────────────
def build_gradio_app(model, tokenizer, device):

    def inference_fn(image, text):
        if image is None:
            return (
                gr.update(value="⚠️ Please upload an image.", visible=True),
                None,
                ""
            )
        scores = predict(image, text, model, tokenizer, device)
        top_label = max(scores, key=scores.get)
        confidence = scores[top_label]

        # Format text output
        result_md = f"## {top_label}  ({confidence*100:.1f}%)\n\n"
        result_md += "| Sentiment | Confidence |\n|-----------|------------|\n"
        for label, score in scores.items():
            bar = "█" * int(score * 20)
            result_md += f"| {label} | {bar} {score*100:.1f}% |\n"

        return result_md, scores, top_label

    # ── Layout ─────────────────────────────────────────────────────────────────
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Multimodal Sentiment Analysis | ELC 2025-26",
        css="""
        .header-text { text-align: center; }
        .result-box  { border-radius: 12px; padding: 10px; }
        footer { display: none !important; }
        """
    ) as demo:

        gr.Markdown(
            """
            <div class='header-text'>
            <h1>🧠 Multimodal Sentiment Analysis</h1>
            <h3>ELC 2025-26 | Computer Vision | MemoTion-7k</h3>
            <p>Upload a meme image and provide its text/caption to predict sentiment using combined visual + language understanding.</p>
            </div>
            """,
            elem_classes=["header-text"]
        )

        with gr.Row():
            # ── Left: Inputs ───────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Inputs")
                image_input = gr.Image(
                    type="pil",
                    label="Upload Meme Image",
                    height=300,
                )
                text_input = gr.Textbox(
                    label="Meme Text / Caption (OCR or manual)",
                    placeholder="Type or paste the text from the meme...",
                    lines=4,
                )
                submit_btn = gr.Button("🔍 Analyse Sentiment", variant="primary", size="lg")
                clear_btn  = gr.Button("🗑️ Clear", variant="secondary")

            # ── Right: Outputs ─────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                result_md   = gr.Markdown(value="*Results will appear here after analysis.*")
                bar_chart   = gr.BarPlot(
                    x="Sentiment",
                    y="Confidence (%)",
                    title="Sentiment Confidence",
                    color="Sentiment",
                    color_map={
                        "Positive 😊": "#2ecc71",
                        "Neutral 😐":  "#f39c12",
                        "Negative 😞": "#e74c3c",
                    },
                    height=250,
                    visible=False,
                )
                top_label_out = gr.Textbox(label="Top Prediction", visible=False)

        # ── Examples ───────────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 💡 How to Use")
        gr.Markdown(
            """
            1. **Upload** a meme image from MemoTion-7k dataset (or any meme)
            2. **Paste** the OCR text or manually type the meme's caption
            3. Click **Analyse Sentiment** to get the prediction
            4. The model uses both **visual features** (ResNet-50) and **text features** (DistilBERT) fused together

            **Sentiment Classes:**
            - 😊 **Positive** — happy, humorous, uplifting content
            - 😐 **Neutral**  — factual or ambiguous content  
            - 😞 **Negative** — angry, sad, offensive content
            """
        )

        # ── Model info accordion ───────────────────────────────────────────────
        with gr.Accordion("ℹ️ Model Architecture Details", open=False):
            gr.Markdown(
                """
                | Component | Details |
                |-----------|---------|
                | Visual Encoder | ResNet-50 (ImageNet pretrained) → Linear(2048→256) |
                | Text Encoder   | DistilBERT-base-uncased → [CLS] → Linear(768→256)  |
                | Fusion         | Concat(256+256) → MLP(512→128→3)                   |
                | Dataset        | MemoTion-7k (6992 meme images + OCR text)           |
                | Loss           | Cross-Entropy                                       |
                | Optimizer      | AdamW, lr=2e-5, cosine schedule                    |
                | Classes        | Positive / Neutral / Negative                       |
                """
            )

        # ── Wire up events ─────────────────────────────────────────────────────
        import pandas as pd

        def run_and_format(image, text):
            if image is None:
                return "*⚠️ Please upload an image first.*", gr.update(visible=False), ""

            scores = predict(image, text, model, tokenizer, device)
            top_label  = max(scores, key=scores.get)
            confidence = scores[top_label]

            result_str  = f"## {top_label}  ({confidence*100:.1f}% confidence)\n\n"
            result_str += "| Sentiment | Score |\n|-----------|-------|\n"
            for lbl, sc in scores.items():
                bar = "█" * int(sc * 25)
                result_str += f"| {lbl} | {bar} `{sc*100:.1f}%` |\n"

            chart_data = pd.DataFrame({
                "Sentiment":      list(scores.keys()),
                "Confidence (%)": [v * 100 for v in scores.values()],
            })

            return result_str, gr.BarPlot(
                value   = chart_data,
                x       = "Sentiment",
                y       = "Confidence (%)",
                title   = "Sentiment Confidence Scores",
                color   = "Sentiment",
                height  = 280,
                visible = True,
            ), top_label

        submit_btn.click(
            fn      = run_and_format,
            inputs  = [image_input, text_input],
            outputs = [result_md, bar_chart, top_label_out],
        )

        clear_btn.click(
            fn      = lambda: (None, "", "*Results will appear here after analysis.*",
                               gr.update(visible=False), ""),
            inputs  = [],
            outputs = [image_input, text_input, result_md, bar_chart, top_label_out],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str,  default="./results/best_model.pt")
    parser.add_argument("--demo",   action="store_true", help="Run without weights")
    parser.add_argument("--port",   type=int,  default=7860)
    parser.add_argument("--share",  action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[App] Device: {device}")

    model_path = None if args.demo else args.model
    model      = load_model(model_path, device)
    tokenizer  = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    demo = build_gradio_app(model, tokenizer, device)
    print(f"[App] Launching on http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
