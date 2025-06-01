# Training script for Siamese U-Net Change Detection
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import argparse # For command-line arguments
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import scheduler

# Assuming models.py and dataset.py are in the same directory or accessible
from dataset import create_change_detection_dataset, collate_fn_skip_none # Use the factory function
from models import SiameseUNet

# --- Configuration (Defaults, can be overridden by args) ---
ROOT_DIR = "/Users/mac/Desktop/MAYNA/Code/Change_Detection_Package" # User specified root
DATASET_SUBDIR_DEFAULT = "Onera Satellite Change Detection Dataset" # Default dataset subdir
SYNTHETIC_DATA_DIR_DEFAULT = "synthetic_data"
CHECKPOINT_DIR_DEFAULT = "siamese_checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_DEFAULT = 4
NUM_EPOCHS_DEFAULT = 50
LEARNING_RATE_DEFAULT = 1e-4
N_CHANNELS = 3 # Fixed for RGB PNGs
N_CLASSES = 1 # Binary change map
TARGET_SIZE_DEFAULT = (128, 128)
SAVE_EVERY_DEFAULT = 5
USE_SYNTHETIC_DEFAULT = False
LOSS_TYPE_DEFAULT = "combined" # Default loss

# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combines BCEWithLogitsLoss and DiceLoss."""
    def __init__(self, alpha=0.5, smooth_dice=1.):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth_dice)

    def forward(self, logits, targets):
        if targets.ndim == 3: # N, H, W -> N, 1, H, W
            targets_float = targets.float().unsqueeze(1)
        elif targets.ndim == 4 and targets.shape[1] == 1:
            targets_float = targets.float()
        else:
            raise ValueError(f"Unexpected target shape: {targets.shape}")

        bce_loss = self.bce(logits, targets_float)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

class FocalLoss(nn.Module):
    """Focal Loss for binary classification tasks with class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Weighting factor for positive class
        self.gamma = gamma # Focusing parameter
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none") # Calculate BCE per element

    def forward(self, logits, targets):
        # Ensure target is float and has channel dim
        if targets.ndim == 3:
            targets_float = targets.float().unsqueeze(1)
        elif targets.ndim == 4 and targets.shape[1] == 1:
            targets_float = targets.float()
        else:
            raise ValueError(f"Unexpected target shape: {targets.shape}")

        bce_loss = self.bce_loss(logits, targets_float)
        probs = torch.sigmoid(logits)
        pt = torch.exp(-bce_loss) # p_t = P(y|x) = exp(-BCE(logits, target))

        # Calculate Focal Loss: alpha * (1-pt)^gamma * log(pt)
        # alpha_t: alpha for positive class, (1-alpha) for negative class
        alpha_t = self.alpha * targets_float + (1 - self.alpha) * (1 - targets_float)
        focal_loss = alpha_t * (1 - pt)**self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# --- Training Function ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        if batch is None: continue
        img1 = batch["image1"].to(device)
        img2 = batch["image2"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader) if len(loader) > 0 else 0.0

# --- Validation Function ---
def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            if batch is None: continue
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)
            labels = batch["label"].to(device)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader) if len(loader) > 0 else 0.0

# --- Main Training Loop ---
def main(args):
    print(f"Using device: {DEVICE}")
    checkpoint_dir = os.path.join(args.root_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Parse target size
    try:
        target_h, target_w = map(int, args.target_size.split("x"))
        target_size = (target_h, target_w)
    except ValueError:
        print("Error: target_size must be in format HxW (e.g., 128x128)")
        return

    # Datasets and Dataloaders
    print("Loading datasets...")
    train_dataset = create_change_detection_dataset(
        root_dir=args.root_dir,
        dataset_subdir=args.dataset_subdir,
        synthetic_data_dir=args.synthetic_data_dir,
        mode="train",
        target_size=target_size,
        use_synthetic=args.use_synthetic
    )
    val_dataset = create_change_detection_dataset(
        root_dir=args.root_dir,
        dataset_subdir=args.dataset_subdir,
        mode="val",
        target_size=target_size,
        use_synthetic=False
    )

    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Check paths and data.")
        return
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Check paths and data.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn_skip_none)
    print(f"Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Model
    print("Initializing model...")
    model = SiameseUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)

    # Loss Function Selection
    if args.loss == "combined":
        criterion = CombinedLoss().to(DEVICE)
        print("Using Combined (BCE + Dice) Loss.")
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE) # Default alpha/gamma, can be tuned
        print("Using Focal Loss.")
    else:
        print(f'Error: Unknown loss type "{args.loss}". Choose "combined" or "focal".')
        return

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5) # Removed verbose=True
    print("Using ReduceLROnPlateau LR scheduler.")

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"nEpoch {epoch}/{args.num_epochs} - LR: {current_lr:.1e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model saved to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"Error saving best model checkpoint: {e}")

        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint at epoch {epoch}: {e}")

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese U-Net for Change Detection")
    parser.add_argument("--root-dir", type=str, default=ROOT_DIR, help="Root project directory") # Corrected ROOT_DIR_DEFAULT to ROOT_DIR
    parser.add_argument("--dataset-subdir", type=str, default=DATASET_SUBDIR_DEFAULT, help="Subdirectory containing the Onera dataset")
    parser.add_argument("--synthetic-data-dir", type=str, default=SYNTHETIC_DATA_DIR_DEFAULT, help="Directory containing generated synthetic data")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR_DEFAULT, help="Directory to save model checkpoints")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS_DEFAULT, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE_DEFAULT, help="Initial learning rate")
    parser.add_argument("--target-size", type=str, default=f"{TARGET_SIZE_DEFAULT[0]}x{TARGET_SIZE_DEFAULT[1]}", help="Target image size HxW (e.g., 128x128)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY_DEFAULT, help="Save checkpoint every N epochs")
    parser.add_argument("--use-synthetic", action="store_true", default=USE_SYNTHETIC_DEFAULT, help="Include synthetic data during training")
    parser.add_argument("--loss", type=str, choices=["combined", "focal"], default=LOSS_TYPE_DEFAULT, help="Loss function type (combined or focal)") # New argument

    args = parser.parse_args()

    # Update ROOT_DIR if provided via command line
    # This helps if the script is run from a different location than expected
    if args.root_dir != ROOT_DIR: # Changed ROOT_DIR_DEFAULT to ROOT_DIR
        print(f"Overriding default ROOT_DIR with: {args.root_dir}")
        # You might want to adjust other paths based on this new root_dir if they aren't absolute

    main(args)

