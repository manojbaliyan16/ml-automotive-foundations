"""
STEP 2 — Fine-Tune ResNet18 Image Classifier
=============================================
Run after download_data.py has built your dataset.

    python train.py

What you'll learn here:
─────────────────────────────────────────────
1. TRANSFER LEARNING  — Why we don't train from scratch
2. FINE-TUNING        — Two-phase strategy: freeze → unfreeze
3. DATA AUGMENTATION  — Making the model robust to variations
4. TRAIN/VAL SPLIT    — Why you need a separate validation set
5. COST FUNCTION      — CrossEntropyLoss and what it measures
6. OVERFITTING        — How to detect it from the loss curve
7. CONFUSION MATRIX   — Which classes are being confused and why
8. MOST CONFUSED      — Finding dirty data by its loss value

Outputs:
    classifier_model.pth     — Saved model weights
    results/loss_curve.png   — Training vs validation loss plot
    results/confusion.png    — Confusion matrix heatmap
    results/most_confused.png — Top images the model got wrong
"""

import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
MODEL_PATH  = Path("classifier_model.pth")
CLASSES_PATH = Path("classes.json")

RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# HYPERPARAMETERS — The knobs you tune
# ─────────────────────────────────────────────
# These are the decisions a Principal Architect must justify in interviews:
BATCH_SIZE  = 32      # How many images per gradient update
IMG_SIZE    = 224     # ResNet expects 224×224 (ImageNet standard)
VAL_SPLIT   = 0.2     # 20% of data held back for validation
PHASE1_EPOCHS = 5     # Epochs with backbone frozen
PHASE2_EPOCHS = 10    # Epochs with all layers unfrozen
LR_PHASE1   = 1e-3    # Higher LR for new head (random weights)
LR_PHASE2   = 1e-4    # Lower LR for fine-tuning (pretrained weights)


# ─────────────────────────────────────────────
# 1. DATA TRANSFORMS — Augmentation + Normalisation
# ─────────────────────────────────────────────
def get_transforms():
    """
    WHY AUGMENTATION?
    -----------------
    Your 150 cat images are 150 specific photos.
    But in the real world, cats appear at different angles, lighting,
    zoom levels. Augmentation artificially creates this variety.

    Random flip:    A cat is still a cat when mirrored
    Random crop:    Teaches model to use partial views
    Color jitter:   Lighting changes shouldn't fool the model
    Rotation:       Orientation shouldn't matter for classification

    WHY NORMALIZE?
    --------------
    ResNet was pretrained on ImageNet with these exact mean/std values.
    Using the same normalisation puts your inputs in the same space
    the model's weights were designed to process.
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] — memorise these.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),          # Random crop from slightly larger image
        transforms.RandomHorizontalFlip(p=0.5),   # 50% chance of mirror flip
        transforms.RandomRotation(degrees=15),    # Rotate up to ±15 degrees
        transforms.ColorJitter(                   # Random brightness/contrast/saturation
            brightness=0.3, contrast=0.3, saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        # NO augmentation on validation — we want a stable, consistent measure
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform


# ─────────────────────────────────────────────
# 2. DATA LOADING — Train/Val Split
# ─────────────────────────────────────────────
def get_dataloaders():
    """
    WHY THREE DATASETS (train/val/test)?
    ─────────────────────────────────────
    Train:      Model sees these images and learns from them
    Validation: Model NEVER trains on these — used only to measure
                how well it generalises to new data
    Test:       Held out until final evaluation (we skip this for now,
                but in production you'd always have a test set)

    If you only had train and validated on the same data, you'd be
    measuring memorisation, not learning. The val loss tells you
    the truth.
    """
    _, val_transform = get_transforms()
    train_transform, _ = get_transforms()

    full_dataset = datasets.ImageFolder(root=str(DATA_DIR))
    classes = full_dataset.classes
    print(f"\nClasses found: {classes}")

    # Save class names — app.py needs this to label predictions
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f)

    n_total = len(full_dataset)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    # Apply different transforms to train vs val
    # We need wrapper datasets since random_split doesn't support per-subset transforms
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset   = TransformSubset(val_subset,   val_transform)

    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Total             : {n_total}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, val_loader, classes


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset to apply a specific transform."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = transforms.ToPILImage()(transforms.ToTensor()(img)) if not isinstance(img, Image.Image) else img
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)


# ─────────────────────────────────────────────
# 3. THE MODEL — Fine-Tuning ResNet18
# ─────────────────────────────────────────────
def build_model(num_classes: int):
    """
    WHY TRANSFER LEARNING?
    ──────────────────────
    ResNet18 was trained on ImageNet — 1.2 million images, 1000 classes.
    Its early layers already know how to detect:
      - Layer 1: Edges, gradients
      - Layer 2: Textures, corners
      - Layer 3: Parts (eyes, wheels, feathers)
      - Layer 4: High-level semantics

    Training these from scratch on 450 images would fail.
    We take these learned representations and just retrain the
    final classifier layer for our 3 classes.

    This is exactly how NVIDIA tunes AI models for DriveOS —
    they don't train perception models from scratch.

    ResNet18 Architecture:
    ──────────────────────
    Input (224×224×3)
    → 4 ResNet blocks (each with skip connections)
    → GlobalAvgPool → 512-dim vector
    → FC(512 → 1000)   ← We REPLACE this with FC(512 → 3)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # PHASE 1: Freeze everything — only the new head will learn
    # Why? The pretrained features are valuable. If we update them
    # immediately with a high learning rate, we "destroy" the knowledge.
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer (1000 ImageNet classes → our num_classes)
    in_features = model.fc.in_features   # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),               # Regularisation
        nn.Linear(in_features, num_classes)
    )
    # New layers are trainable by default

    return model


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    """
    One pass over training data.
    Returns average loss and accuracy.
    """
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # COST FUNCTION: CrossEntropyLoss
        # ─────────────────────────────────
        # For a batch of images, it:
        # 1. Applies Softmax to convert raw scores → probabilities
        # 2. Takes the -log of the probability of the correct class
        # 3. Averages across the batch
        #
        # Why -log? If model is very confident (prob→1), loss→0
        #            If model is wrong (prob→0), loss→∞
        # This heavily penalises confident wrong predictions.
        #
        # In automotive AI: a wrong confident prediction (STOP sign = speed limit)
        # is catastrophic. The loss function encodes this priority.
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


def eval_epoch(model, loader, criterion, device):
    """One pass over validation data. No gradients."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), 100 * correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# 5. VISUALISATION TOOLS
# ─────────────────────────────────────────────
def plot_loss_curve(train_losses, val_losses, train_accs, val_accs):
    """
    The loss curve is your most important debugging tool.

    WHAT TO LOOK FOR:
    ──────────────────
    ✓ Both curves going down  → model is learning
    ✗ Val loss going UP while train loss goes down → OVERFITTING
      (model memorising training data, failing to generalise)
    ✗ Both losses stuck → learning rate too low, or model too small
    ✗ Both losses jumping around → learning rate too high
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Metrics — Watch for Overfitting!", fontsize=14, fontweight='bold')

    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses,   'r-o', label='Val Loss',   markersize=4)
    ax1.set_title("Loss per Epoch\n(Val loss rising = overfitting)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy', markersize=4)
    ax2.plot(epochs, val_accs,   'r-o', label='Val Accuracy',   markersize=4)
    ax2.set_title("Accuracy per Epoch\n(Gap between lines = generalisation error)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / "loss_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Loss curve saved → {path}")


def plot_confusion_matrix(all_labels, all_preds, classes):
    """
    CONFUSION MATRIX — What is the model actually confusing?
    ─────────────────────────────────────────────────────────
    Rows = actual class, Columns = predicted class
    Diagonal = correct predictions (want high numbers here)
    Off-diagonal = mistakes (want zeros)

    In automotive: confusion between STOP sign and speed limit sign
    is a safety-critical failure. The confusion matrix shows you exactly
    where your model fails — not just an accuracy number.
    """
    cm = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_pct,
        annot=True, fmt='.1f', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5
    )
    ax.set_title("Confusion Matrix (%)\nDiagonal = correct. Off-diagonal = mistakes.", fontweight='bold')
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.tight_layout()

    path = RESULTS_DIR / "confusion.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {path}")

    # Print text report too
    print("\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=3))


def find_most_confused(model, val_loader, classes, device, top_n=12):
    """
    DATA CLEANING VIEW — Find the images the model is most wrong about.
    ────────────────────────────────────────────────────────────────────
    High-loss images on the validation set are either:
    1. Genuinely hard images (ambiguous, unusual angle)
    2. Mislabelled images  (the real enemy)
    3. Corrupted / watermarked images

    In a real ML pipeline, this is how you iteratively clean your dataset.
    A data scientist spends 80% of their time on data, not models.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')   # Loss per image

    all_losses, all_imgs, all_preds, all_labels = [], [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            losses  = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)

            all_losses.extend(losses.cpu().numpy())
            all_imgs.extend(images.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Sort by loss — highest loss = most confused
    sorted_idx = np.argsort(all_losses)[::-1][:top_n]

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cols = 4
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle("Most Confused Images — Candidates for Data Cleaning",
                 fontsize=13, fontweight='bold')

    for i, idx in enumerate(sorted_idx):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]

        # Denormalize for display
        img = all_imgs[idx] * IMAGENET_STD + IMAGENET_MEAN
        img = img.permute(1, 2, 0).numpy().clip(0, 1)

        true_label = classes[all_labels[idx]]
        pred_label = classes[all_preds[idx]]
        loss_val   = all_losses[idx]

        ax.imshow(img)
        correct = (true_label == pred_label)
        color = "green" if correct else "red"
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label} (loss={loss_val:.2f})",
            fontsize=9, color=color
        )
        ax.axis("off")

    # Hide unused subplots
    for i in range(len(sorted_idx), rows * cols):
        axes[i // cols][i % cols].axis("off")

    plt.tight_layout()
    path = RESULTS_DIR / "most_confused.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Most confused images saved → {path}")


# ─────────────────────────────────────────────
# 6. MAIN — TRAINING ORCHESTRATION
# ─────────────────────────────────────────────
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if not DATA_DIR.exists():
        print("ERROR: data/ folder not found. Run 'python download_data.py' first.")
        return

    train_loader, val_loader, classes = get_dataloaders()
    num_classes = len(classes)

    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    # ────────────────────────────────────────────
    # PHASE 1: Train only the new head (frozen backbone)
    # ────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Training classifier head (backbone frozen)")
    print("  Only the final Linear layer learns.")
    print("  LR is high because these weights start random.")
    print("="*60)

    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1
    )
    scheduler_p1 = optim.lr_scheduler.OneCycleLR(
        optimizer_p1, max_lr=LR_PHASE1,
        steps_per_epoch=len(train_loader), epochs=PHASE1_EPOCHS
    )

    best_val_acc = 0

    for epoch in range(1, PHASE1_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer_p1, criterion, device)
        vl_loss, vl_acc, preds, labels = eval_epoch(model, val_loader, criterion, device)
        scheduler_p1.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        print(f"  Epoch {epoch:2d}/{PHASE1_EPOCHS} │ "
              f"Train {tr_acc:.1f}% loss={tr_loss:.4f} │ "
              f"Val {vl_acc:.1f}% loss={vl_loss:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_PATH)

    # ────────────────────────────────────────────
    # PHASE 2: Unfreeze all layers, fine-tune end-to-end
    # ────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2 — Fine-tuning full network (all layers unfrozen)")
    print("  All layers now update, but with a MUCH lower learning rate.")
    print("  Why lower? Pretrained weights are valuable — small nudges only.")
    print("="*60)

    for param in model.parameters():
        param.requires_grad = True

    optimizer_p2 = optim.Adam(model.parameters(), lr=LR_PHASE2)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=PHASE2_EPOCHS
    )

    for epoch in range(1, PHASE2_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer_p2, criterion, device)
        vl_loss, vl_acc, preds, labels = eval_epoch(model, val_loader, criterion, device)
        scheduler_p2.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        print(f"  Epoch {epoch:2d}/{PHASE2_EPOCHS} │ "
              f"Train {tr_acc:.1f}% loss={tr_loss:.4f} │ "
              f"Val {vl_acc:.1f}% loss={vl_loss:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_PATH)
            print(f"  ✓ Best model saved! Val Acc: {vl_acc:.1f}%")

    # ────────────────────────────────────────────
    # VISUALISATIONS — Run all on best model
    # ────────────────────────────────────────────
    print("\n" + "="*60)
    print("GENERATING VISUALISATIONS")
    print("="*60)

    # Load best weights for visualisation
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # 1. Loss curve
    plot_loss_curve(train_losses, val_losses, train_accs, val_accs)

    # 2. Confusion matrix (final epoch)
    _, _, final_preds, final_labels = eval_epoch(model, val_loader, criterion, device)
    plot_confusion_matrix(final_labels, final_preds, classes)

    # 3. Most confused images
    find_most_confused(model, val_loader, classes, device)

    print("\n" + "="*60)
    print(f"TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to          : {MODEL_PATH}")
    print(f"Visualisations in       : {RESULTS_DIR}/")
    print(f"\nNext step: Run  python app.py")
    print("="*60)


if __name__ == "__main__":
    main()
