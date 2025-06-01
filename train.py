# Training script for Siamese U-Net Change Detection
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm

# Assuming models.py and dataset.py are in the same directory or accessible
from dataset import ChangeDetectionDataset
from models import SiameseUNet

# --- Configuration ---
ROOT_DIR = "/Users/mac/Desktop/MAYNA/Code/Change_Detection_Package"  # User specified root
DATASET_SUBDIR = "Onera Satellite Change Detection Dataset" # User specified dataset subdir
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "siamese_checkpoints") # Separate checkpoints for Siamese model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2 # Adjust based on GPU memory
NUM_EPOCHS = 50 # Start with a reasonable number, adjust later
LEARNING_RATE = 1e-4
# N_CHANNELS is now fixed to 3 for RGB PNGs
N_CHANNELS = 3
N_CLASSES = 1 # Binary change map
TARGET_SIZE = (128, 128) # Target size for Siamese model training
SAVE_EVERY = 5 # Save checkpoint every N epochs

print(f"Using device: {DEVICE}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Dice Loss ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice

# --- Combined Loss ---
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth_dice=1.):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha # Weight for BCE
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth_dice)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float().unsqueeze(1)) # BCE expects N,C,H,W, target N,H,W
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

# --- Training Function ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        img1 = batch["image1"].to(DEVICE)
        img2 = batch["image2"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(img1, img2)

        # Ensure label has the same spatial dimensions as output if needed (usually handled by dataset/transforms)
        # If outputs are N,1,H,W and labels are N,H,W
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)

# --- Validation Function ---
def validate(model, loader, criterion):
    model.eval()
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            img1 = batch["image1"].to(DEVICE)
            img2 = batch["image2"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)

# --- Main Training Loop ---
def main():
    # Datasets and Dataloaders
    # Note: OSCD doesn't have a standard val split, often a subset of train or specific cities are used.
    # Here, we'll just use the train set for both train/val demonstration purposes.
    # A proper split (e.g., holding out some training cities) is recommended for real training.
    print("Loading datasets...")
    train_dataset = ChangeDetectionDataset(root_dir=ROOT_DIR, dataset_subdir=DATASET_SUBDIR, mode="train", target_size=TARGET_SIZE)
    # For validation, ideally use a separate validation set or split train set
    val_dataset = ChangeDetectionDataset(root_dir=ROOT_DIR, dataset_subdir=DATASET_SUBDIR, mode="val", target_size=TARGET_SIZE) # Using val split defined in dataset class

    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Check dataset path and structure.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Model
    print("Initializing model...")
    model = SiameseUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)

    # Loss and Optimizer
    criterion = CombinedLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")

        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()

