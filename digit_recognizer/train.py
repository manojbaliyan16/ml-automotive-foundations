"""
MNIST Digit Classifier — Train CNN from Scratch
================================================
Run this first to train the model and save it.

    python train.py

What this script teaches you:
- How a CNN is structured (conv layers, pooling, fully connected)
- What a training loop actually does
- How loss decreases over epochs (watch the numbers!)
- How validation accuracy tells you if the model is learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# 1. THE CNN ARCHITECTURE
# ─────────────────────────────────────────────
class DigitCNN(nn.Module):
    """
    A simple but effective CNN for MNIST digit classification.

    Architecture:
        Input:  1 × 28 × 28  (grayscale image)
        Conv1:  32 filters of 3×3 → ReLU → MaxPool → 32 × 13 × 13
        Conv2:  64 filters of 3×3 → ReLU → MaxPool → 64 × 6 × 6
        Flatten → 2304 values
        FC1:    2304 → 128 → ReLU → Dropout(0.5)
        FC2:    128 → 10  (one score per digit 0-9)
        Output: 10 class probabilities via Softmax (done in loss fn)
    """

    def __init__(self):
        super().__init__()

        # Conv Layer 1:
        # - in_channels=1 because MNIST is grayscale
        # - out_channels=32 means we learn 32 different filters
        # - kernel_size=3 means each filter looks at a 3×3 patch
        # Each filter learns to detect a specific pattern (edges, curves, etc.)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

        # Conv Layer 2:
        # - Takes 32 feature maps as input, outputs 64
        # - Learns combinations of the patterns found in conv1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # MaxPool: Slides a 2×2 window and keeps only the max value
        # Purpose: Reduces spatial size, makes model tolerant to small shifts
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout: Randomly zeros out 50% of neurons during training
        # Purpose: Forces the network to not rely on any single neuron → prevents overfitting
        self.dropout = nn.Dropout(p=0.5)

        # Fully Connected layers — classic feedforward NN at the end
        # Input size = 64 channels × 6 × 6 spatial = 2304
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0–9)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]

        # Block 1: Convolve → Activate → Pool
        x = self.conv1(x)       # [batch, 32, 26, 26]  (28-3+1=26)
        x = self.relu(x)
        x = self.pool(x)        # [batch, 32, 13, 13]

        # Block 2: Convolve → Activate → Pool
        x = self.conv2(x)       # [batch, 64, 11, 11]  (13-3+1=11)
        x = self.relu(x)
        x = self.pool(x)        # [batch, 64, 5, 5] → wait, actually 5×5

        # Flatten: Convert 3D feature maps into 1D vector for FC layers
        x = x.view(x.size(0), -1)   # [batch, 64*5*5] = [batch, 1600]

        # Fully Connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)     # Only active during training
        x = self.fc2(x)         # Raw scores (logits) for each of 10 classes

        return x


# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────
def get_dataloaders(batch_size=64):
    """
    Downloads MNIST and returns train + test DataLoaders.

    Transforms applied:
    - ToTensor():    Converts PIL image to tensor, scales pixels from [0,255] to [0,1]
    - Normalize():   Shifts to mean=0.1307, std=0.3081 (MNIST dataset statistics)
                     This helps gradients flow better during training
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"Training samples : {len(train_dataset):,}")
    print(f"Test samples     : {len(test_dataset):,}")
    print(f"Classes          : {train_dataset.classes}")

    return train_loader, test_loader


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    One full pass over the training data.

    For each batch:
    1. Forward pass:  model predicts class scores
    2. Loss:          compare predictions to true labels
    3. Backward pass: compute gradients (backpropagation)
    4. Update:        optimizer adjusts weights using gradients
    """
    model.train()  # Enables dropout, batch norm training mode
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients — IMPORTANT: gradients accumulate by default in PyTorch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Loss: CrossEntropyLoss = Softmax + NegativeLogLikelihood
        # It penalises the model heavily when it's confidently wrong
        loss = criterion(outputs, labels)

        # Backward pass: compute ∂loss/∂weight for every weight in the network
        loss.backward()

        # Update weights: weight = weight - learning_rate × gradient
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Step {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    print(f"  → Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on test set — no gradient tracking needed."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disables gradient computation — faster, less memory
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    print(f"  → Val Loss:   {avg_loss:.4f} | Val Accuracy:   {accuracy:.2f}%")
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# 4. MAIN — PUT IT ALL TOGETHER
# ─────────────────────────────────────────────
def main():
    EPOCHS     = 10
    BATCH_SIZE = 64
    LR         = 0.001   # Learning rate — how big each weight update step is

    # Use GPU if available (MPS on Mac M-series, CUDA on NVIDIA), else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}\n")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model     = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}\n")

    best_val_acc = 0
    print("=" * 60)
    print("TRAINING STARTED")
    print("Watch the loss go down and accuracy go up each epoch!")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "digit_model.pth")
            print(f"  ✓ New best model saved! Val Accuracy: {val_acc:.2f}%")

    print(f"\n{'='*60}")
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: digit_model.pth")
    print(f"Now run: python app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
