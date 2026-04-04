"""
STEP 3 — Image Classifier Web App
====================================
Run after training is complete.

    python app.py

Open browser at: http://localhost:7860

Features:
- Upload any image → get prediction with confidence bars
- See how confident (or confused) the model is
- View training results: loss curve, confusion matrix, most confused images
- Understand what the model is actually doing under the hood
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image
from torchvision import transforms, models


# ─────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH   = Path("classifier_model.pth")
CLASSES_PATH = Path("classes.json")
RESULTS_DIR  = Path("results")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "classifier_model.pth not found.\n"
            "Run: python download_data.py → python train.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes    = checkpoint["classes"]
    num_classes = len(classes)

    # Rebuild same architecture as train.py
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model loaded. Classes: {classes}")
    return model, classes


try:
    MODEL, CLASSES = load_model()
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    MODEL, CLASSES = None, ["cats", "dogs", "birds"]   # Fallback for UI demo


# ─────────────────────────────────────────────
# 2. INFERENCE PIPELINE
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def classify_image(image: np.ndarray):
    """
    Full inference pipeline:
    1. Preprocess image (resize, normalise)
    2. Forward pass through ResNet18
    3. Softmax → probabilities
    4. Return confidence for each class

    Why Softmax?
    ────────────
    The model's final layer outputs raw scores (logits).
    Softmax converts these to probabilities that sum to 1.0.
    A score of [2.1, 0.3, -1.4] becomes roughly [0.75, 0.18, 0.07].
    The highest probability = the model's best guess.

    Watch the confidence values carefully:
    - 0.98 on cats → very confident → probably correct
    - 0.42 on cats → uncertain → inspect the image
    - 0.35/0.33/0.32 split → model has no idea → bad image or edge case
    """
    if MODEL is None:
        return {c: 0.0 for c in CLASSES}

    if image is None:
        return {c: 0.0 for c in CLASSES}

    pil_img = Image.fromarray(image).convert("RGB")
    tensor  = INFER_TRANSFORM(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().numpy()

    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


# ─────────────────────────────────────────────
# 3. GRADIO APP
# ─────────────────────────────────────────────
CSS = """
.title { text-align: center; font-size: 1.8em; font-weight: bold; margin-bottom: 0.2em; }
.subtitle { text-align: center; color: #555; margin-bottom: 1.5em; font-size: 1em; }
.concept-box { background: #f0f4ff; border-left: 4px solid #4a6cf7;
               padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
"""

RESULTS_EXISTS = RESULTS_DIR.exists()
LOSS_CURVE     = str(RESULTS_DIR / "loss_curve.png")    if RESULTS_EXISTS else None
CONFUSION      = str(RESULTS_DIR / "confusion.png")     if RESULTS_EXISTS else None
MOST_CONFUSED  = str(RESULTS_DIR / "most_confused.png") if RESULTS_EXISTS else None


with gr.Blocks(title="Image Classifier") as demo:

    gr.HTML('<div class="title">🐱🐶🐦 Image Classifier</div>')
    gr.HTML(
        '<div class="subtitle">ResNet18 fine-tuned on Cats · Dogs · Birds<br>'
        'Transfer learning · CrossEntropyLoss · Data Augmentation</div>'
    )

    with gr.Tabs():

        # ── Tab 1: Classify ────────────────────────────────────
        with gr.Tab("🔍  Classify an Image"):

            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Upload a cat, dog, or bird photo",
                        type="numpy",
                        image_mode="RGB",
                        height=320,
                    )
                    classify_btn = gr.Button("Classify", variant="primary", size="lg")

                with gr.Column(scale=1):
                    label_output = gr.Label(
                        label="Prediction Confidence",
                        num_top_classes=len(CLASSES)
                    )
                    gr.Markdown("""
                    **How to read the confidence bars:**
                    - **> 90%** on one class → Model is very confident
                    - **50–90%** → Reasonably confident
                    - **Split evenly** → Model is confused — inspect the image
                    - **High confidence + wrong answer** → Bad training data
                    """)

            classify_btn.click(fn=classify_image, inputs=img_input, outputs=label_output)
            img_input.change(fn=classify_image, inputs=img_input, outputs=label_output)

        # ── Tab 2: Training Results ────────────────────────────
        with gr.Tab("📈  Training Results"):
            gr.Markdown("### What happened during training?")

            if LOSS_CURVE and Path(LOSS_CURVE).exists():
                gr.Markdown("""
                **Loss Curve — The most important diagnostic tool**

                Watch the gap between Train and Val loss:
                - Curves close together → Good generalisation
                - Val loss rising while train falls → **Overfitting** (memorisation, not learning)
                - Both curves flat → Learning rate too low, or model too small
                """)
                gr.Image(value=LOSS_CURVE, label="Training vs Validation Loss", show_label=True)
            else:
                gr.Markdown("*Run `python train.py` first to generate training plots.*")

        # ── Tab 3: Confusion Matrix ────────────────────────────
        with gr.Tab("🔢  Confusion Matrix"):
            gr.Markdown("### Which classes is the model confusing?")

            if CONFUSION and Path(CONFUSION).exists():
                gr.Markdown("""
                **How to read this:**
                - **Diagonal cells** → Correct predictions (want high %)
                - **Off-diagonal cells** → Mistakes (want 0%)

                Example: If cats/dogs row shows 85%/15% → Model mistakes 15% of cats for dogs.
                This tells you: your cat and dog images may look too similar, or you need more diverse training data.

                In automotive AI — if your model confuses STOP signs with speed limit signs,
                the confusion matrix pinpoints it immediately. Accuracy alone won't tell you that.
                """)
                gr.Image(value=CONFUSION, label="Confusion Matrix", show_label=True)
            else:
                gr.Markdown("*Run `python train.py` first to generate confusion matrix.*")

        # ── Tab 4: Data Cleaning View ──────────────────────────
        with gr.Tab("🧹  Data Cleaning View"):
            gr.Markdown("### Images the model is most wrong about")

            if MOST_CONFUSED and Path(MOST_CONFUSED).exists():
                gr.Markdown("""
                **These are the highest-loss images from your validation set.**

                Each image shown here is one the model got wrong — or barely got right.

                For each image, ask yourself:
                1. Is it mislabelled? (Dog labelled as cat?)
                2. Is it ambiguous? (A hybrid that looks like multiple classes?)
                3. Is it a bad image? (Watermark, partial view, wrong subject?)

                **The action:** Remove or relabel bad images, then retrain.
                This is real-world ML workflow — not a one-shot process.
                A model is only as good as its data.

                > *"Garbage in, garbage out"* — this visualization makes garbage visible.
                """)
                gr.Image(value=MOST_CONFUSED, label="Most Confused Images (Validation Set)", show_label=True)
            else:
                gr.Markdown("*Run `python train.py` first to generate data cleaning view.*")

        # ── Tab 5: Concepts Explained ──────────────────────────
        with gr.Tab("📚  Concepts Explained"):
            gr.Markdown("""
            ## What this app teaches you

            ### Transfer Learning
            ResNet18 was trained on 1.2M ImageNet images. It already knows how to detect
            edges, textures, shapes, and object parts. We replace only the final layer
            and retrain it for our 3 classes. This is why we get 90%+ accuracy with
            only 450 images — we're standing on the shoulders of 1.2M.

            ---

            ### Fine-Tuning in Two Phases
            **Phase 1 — Frozen backbone:**
            Only the new classifier head learns. LR = 1e-3.
            Why? The new layer starts with random weights. A high LR here is fine.
            The backbone weights (valuable pretrained knowledge) stay frozen.

            **Phase 2 — Full fine-tuning:**
            All layers learn, but with LR = 1e-4 (10× smaller).
            Why lower? We don't want to destroy the pretrained representations.
            Small nudges to adapt them to our domain is all we need.

            ---

            ### Cost Function (CrossEntropyLoss)
            The loss function is what the model is trying to minimise.

            For a cat image where the model says [cat=0.1, dog=0.8, bird=0.1]:
            Loss = -log(0.1) = 2.3 → Very high → Model gets punished hard

            For the same image where model says [cat=0.9, dog=0.05, bird=0.05]:
            Loss = -log(0.9) = 0.1 → Very low → Model gets small punishment

            Backpropagation uses this loss to compute how much each weight
            contributed to the mistake, then adjusts all weights accordingly.

            ---

            ### Overfitting
            When a model memorises training data instead of learning patterns.
            Signs: Training accuracy 98%, Validation accuracy 70%.
            Fixes: More data, data augmentation, dropout, early stopping.

            ---

            ### Data Augmentation
            We apply random flips, rotations, colour changes to training images.
            This is NOT cheating — a cat is still a cat when slightly rotated.
            It forces the model to learn position/lighting-invariant features.

            ---

            ### Validation Split
            20% of data is held out during training — model never sees it.
            This gives us an honest measure of real-world performance.
            If you validated on training data, you'd be measuring memorisation.
            """)

    gr.Markdown(
        "<center><small>Part of Manoj's 52-Week Principal Architect Roadmap | "
        "P1.1 Deliverable — Image Classifier Portfolio Project</small></center>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
        theme=gr.themes.Soft(),
    )
