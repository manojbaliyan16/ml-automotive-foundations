"""
Digit Recognizer — Local Web App
=================================
Make sure you have trained the model first:
    python train.py

Then run this app:
    python app.py

Open browser at: http://localhost:7860
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import gradio as gr


# ─────────────────────────────────────────────
# 1. LOAD THE TRAINED CNN
# ─────────────────────────────────────────────
class DigitCNN(nn.Module):
    """Same architecture as train.py — must match exactly to load weights."""
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3)
        self.conv2   = nn.Conv2d(32, 64, 3)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(64 * 6 * 6, 128)
        self.fc2     = nn.Linear(128, 10)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model():
    model = DigitCNN()
    try:
        model.load_state_dict(torch.load("digit_model.pth", map_location="cpu"))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("ERROR: digit_model.pth not found. Run 'python train.py' first.")
        raise
    return model


MODEL = load_model()


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
# MNIST images are: 28×28, grayscale, white digit on BLACK background
# Normalize with the same stats used during training
TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Convert raw image (from canvas draw or file upload) into
    the exact tensor format the CNN expects.

    Steps:
    1. Convert to PIL
    2. Invert if needed (MNIST expects white digit on black bg)
    3. Resize to 28×28
    4. Normalize
    5. Add batch dimension: [1, 28, 28] → [1, 1, 28, 28]
    """
    if image is None:
        return None

    # Handle RGBA (canvas output) and RGB (uploaded image)
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            # RGBA canvas: use alpha channel as the digit mask
            pil_img = Image.fromarray(image, mode='RGBA')
            # Create white background, paste using alpha
            background = Image.new('RGB', pil_img.size, (0, 0, 0))
            background.paste(pil_img, mask=pil_img.split()[3])
            pil_img = background.convert('L')
        else:
            pil_img = Image.fromarray(image).convert('L')
    else:
        pil_img = image.convert('L')

    # Auto-invert: if image is mostly white (uploaded scan), invert it
    # so the digit is white on black — matching MNIST format
    arr = np.array(pil_img)
    if arr.mean() > 127:
        pil_img = ImageOps.invert(pil_img)

    tensor = TRANSFORM(pil_img)           # [1, 28, 28]
    tensor = tensor.unsqueeze(0)          # [1, 1, 28, 28]
    return tensor


# ─────────────────────────────────────────────
# 3. PREDICTION
# ─────────────────────────────────────────────
def predict(image: np.ndarray) -> dict:
    """
    Run inference and return confidence scores for all 10 digits.

    The model outputs raw logits (unnormalised scores).
    We apply Softmax to convert to probabilities that sum to 1.
    """
    if image is None:
        return {str(i): 0.0 for i in range(10)}

    tensor = preprocess(image)
    if tensor is None:
        return {str(i): 0.0 for i in range(10)}

    with torch.no_grad():
        logits = MODEL(tensor)                   # Raw scores: [1, 10]
        probs  = F.softmax(logits, dim=1)        # Probabilities: [1, 10]
        probs  = probs.squeeze().numpy()         # [10]

    # Return as dict: digit → confidence (for Gradio label component)
    return {str(i): float(probs[i]) for i in range(10)}


# ─────────────────────────────────────────────
# 4. GRADIO UI
# ─────────────────────────────────────────────
with gr.Blocks(
    title="Digit Recognizer",
    theme=gr.themes.Soft(),
    css="""
        .title { text-align: center; font-size: 2em; font-weight: bold; margin-bottom: 0.2em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
    """
) as demo:

    gr.HTML('<div class="title">Digit Recognizer</div>')
    gr.HTML('<div class="subtitle">CNN trained from scratch on MNIST · 99%+ accuracy</div>')

    with gr.Tabs():

        # ── Tab 1: Draw ──────────────────────────────
        with gr.Tab("✏️  Draw a Digit"):
            gr.Markdown("**Draw any digit (0–9) in the box below. Prediction updates live.**")
            with gr.Row():
                with gr.Column(scale=1):
                    draw_input = gr.Image(
                        label="Draw here",
                        type="numpy",
                        image_mode="RGBA",
                        height=300,
                        width=300,
                        tool="color-sketch",
                        source="canvas",
                        brush_radius=18,
                        invert_colors=True,  # White digit on black bg — MNIST style
                    )
                    gr.Markdown(
                        "*Tip: Draw thick strokes, centred in the box. "
                        "The model sees a 28×28 version of what you draw.*"
                    )
                with gr.Column(scale=1):
                    draw_output = gr.Label(
                        label="Prediction Confidence",
                        num_top_classes=10
                    )

            draw_input.change(
                fn=predict,
                inputs=draw_input,
                outputs=draw_output
            )

        # ── Tab 2: Upload ────────────────────────────
        with gr.Tab("📁  Upload an Image"):
            gr.Markdown(
                "**Upload a photo or scan of a handwritten digit.**\n\n"
                "Works best with: clear handwriting, high contrast, single digit per image."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.Image(
                        label="Upload digit image",
                        type="numpy",
                        image_mode="RGB",
                        height=300,
                    )
                with gr.Column(scale=1):
                    upload_output = gr.Label(
                        label="Prediction Confidence",
                        num_top_classes=10
                    )

            upload_btn = gr.Button("Predict", variant="primary")
            upload_btn.click(
                fn=predict,
                inputs=upload_input,
                outputs=upload_output
            )

    # ── How it works section ─────────────────────
    with gr.Accordion("How does this work?", open=False):
        gr.Markdown("""
        **CNN Architecture:**
        ```
        Input: 1 × 28 × 28 (grayscale)
            ↓
        Conv2D(32 filters, 3×3) → ReLU → MaxPool(2×2)
            ↓
        Conv2D(64 filters, 3×3) → ReLU → MaxPool(2×2)
            ↓
        Flatten → Linear(1600 → 128) → ReLU → Dropout(0.5)
            ↓
        Linear(128 → 10) → Softmax
            ↓
        Output: Probability for each digit (0–9)
        ```

        **Key concepts:**
        - **Convolution**: A small filter slides across the image, detecting patterns like edges and curves
        - **ReLU**: Removes negative values — adds non-linearity so the network can learn complex patterns
        - **MaxPool**: Shrinks the image by keeping only the strongest activations — reduces compute, adds position tolerance
        - **Dropout**: Randomly turns off neurons during training — prevents the model from memorising, forces generalisation
        - **Softmax**: Converts raw scores into probabilities that sum to 1.0
        - **CrossEntropyLoss**: Penalises wrong predictions — drives the training
        """)


# ─────────────────────────────────────────────
# 5. LAUNCH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # Set to True to get a public link
        inbrowser=True,    # Automatically opens the browser
    )
