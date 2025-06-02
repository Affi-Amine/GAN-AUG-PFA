# Training script for Siamese U-Net Change Detection
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna  # For hyperparameter tuning

# Assuming models.py and dataset.py are in the same directory or accessible
from dataset import create_change_detection_dataset, collate_fn_skip_none
from models import SiameseUNet

# --- Configuration (Defaults, can be overridden by args) ---
ROOT_DIR = "/Users/mac/Desktop/MAYNA/Code/Change_Detection_Package"  # User specified root
DATASET_SUBDIR_DEFAULT = "Onera Satellite Change Detection Dataset"
SYNTHETIC_DATA_DIR_DEFAULT = "synthetic_data"
CHECKPOINT_DIR_DEFAULT = "siamese_checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_DEFAULT = 4
NUM_EPOCHS_DEFAULT = 50
LEARNING_RATE_DEFAULT = 1e-4
N_CHANNELS = 3  # Fixed for RGB PNGs
N_CLASSES = 1  # Binary change map
TARGET_SIZE_DEFAULT = (128, 128)
SAVE_EVERY_DEFAULT = 5
USE_SYNTHETIC_DEFAULT = False

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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Weight for the positive class
        self.reduction = reduction

    def forward(self, logits, targets):
        # Ensure targets are float and have the same shape as logits for BCE
        if targets.ndim == 3: # Assuming [B, H, W]
            targets_float = targets.float().unsqueeze(1) # to [B, 1, H, W]
        elif targets.ndim == 4 and targets.shape[1] == 1: # Assuming [B, 1, H, W]
            targets_float = targets.float()
        else:
            # This case might occur if labels are already float and correctly shaped
            targets_float = targets.float() 
            # Add a check to ensure compatibility if needed, or raise error
            if logits.shape != targets_float.shape:
                 raise ValueError(f"Logits shape {logits.shape} and targets shape {targets_float.shape} mismatch in FocalLoss")

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_float, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Calculate alpha_t for weighting
        alpha_t = targets_float * self.alpha + (1 - targets_float) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth_dice=1., class_weight=[1.0, 9.0]):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight[1]).to(DEVICE))
        self.dice = DiceLoss(smooth=smooth_dice)

    def forward(self, logits, targets):
        # Ensure targets are float and have the same shape as logits for BCE
        if targets.ndim == 3: # Assuming [B, H, W]
            targets_float = targets.float().unsqueeze(1) # to [B, 1, H, W]
        elif targets.ndim == 4 and targets.shape[1] == 1: # Assuming [B, 1, H, W]
            targets_float = targets.float()
        else:
            # This case might occur if labels are already float and correctly shaped
            targets_float = targets.float()
            if logits.shape != targets_float.shape:
                 raise ValueError(f"Logits shape {logits.shape} and targets shape {targets_float.shape} mismatch in CombinedLoss BCE part")

        bce_loss = self.bce(logits, targets_float)
        # DiceLoss expects raw targets (long or int) if it handles conversion internally, or float if not.
        # The existing DiceLoss seems to work with sigmoid probs and float targets.
        dice_loss = self.dice(logits, targets_float) # Pass float targets to DiceLoss as well for consistency
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

# --- New FocalDiceLoss ---
class FocalDiceLoss(nn.Module):
    def __init__(self, beta=0.5, focal_gamma=2, focal_alpha=0.75, dice_smooth=1.):
        super(FocalDiceLoss, self).__init__()
        self.beta = beta
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(self, logits, targets):
        # Ensure targets are float and have the same shape as logits for FocalLoss
        if targets.ndim == 3: # Assuming [B, H, W]
            targets_float = targets.float().unsqueeze(1) # to [B, 1, H, W]
        elif targets.ndim == 4 and targets.shape[1] == 1: # Assuming [B, 1, H, W]
            targets_float = targets.float()
        else:
            targets_float = targets.float()
            if logits.shape != targets_float.shape:
                 raise ValueError(f"Logits shape {logits.shape} and targets shape {targets_float.shape} mismatch in FocalDiceLoss")

        focal = self.focal_loss(logits, targets_float)
        dice = self.dice_loss(logits, targets_float) # DiceLoss also needs float targets if probs are sigmoided internally
        return self.beta * focal + (1 - self.beta) * dice

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

# --- Hyperparameter Tuning with Optuna ---
def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 5e-3) # Wider range, slightly lower upper bound
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8]) # Categorical for typical batch sizes
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam'])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    
    # FocalDiceLoss parameters
    focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9) # Broader range for alpha
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0) # Range for gamma
    loss_beta = trial.suggest_float('loss_beta', 0.3, 0.7) # Weight between Focal and Dice
    dice_smooth = trial.suggest_loguniform('dice_smooth', 1e-7, 1e-4)

    # Augmentation parameters (optional, can make tuning much longer)
    # Example: aug_color_brightness = trial.suggest_float('aug_color_brightness', 0.1, 0.5)

    current_epochs_for_trial = 15  # Increased epochs for a more stable evaluation

    try:
        target_h, target_w = map(int, args.target_size.split("x"))
        target_size = (target_h, target_w)
    except ValueError:
        print("Error: target_size must be in format HxW (e.g., 128x128)")
        raise optuna.exceptions.TrialPruned("Invalid target_size format")

    # Create datasets with potentially tuned augmentation parameters if added
    # For now, using fixed augmentation settings from args or defaults
    train_dataset = create_change_detection_dataset(
        root_dir=args.root_dir,
        dataset_subdir=args.dataset_subdir,
        synthetic_data_dir=args.synthetic_data_dir,
        mode="train",
        target_size=target_size,
        use_synthetic=args.use_synthetic,
        augment=True # Assuming augmentation is desired during tuning
    )
    val_dataset = create_change_detection_dataset(
        root_dir=args.root_dir,
        dataset_subdir=args.dataset_subdir,
        mode="val",
        target_size=target_size,
        use_synthetic=False,
        augment=False # No augmentation for validation
    )

    if len(train_dataset) == 0:
        print("Error: Training dataset is empty for this trial. Pruning.")
        raise optuna.exceptions.TrialPruned("Empty training dataset")
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty for this trial. No validation loss can be computed.")
        # Depending on Optuna setup, might need to return a very high loss or prune
        raise optuna.exceptions.TrialPruned("Empty validation dataset")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn_skip_none, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn_skip_none, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    model = SiameseUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)
    criterion = FocalDiceLoss(beta=loss_beta, focal_gamma=focal_gamma, focal_alpha=focal_alpha, dice_smooth=dice_smooth).to(DEVICE)
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Adam also has weight_decay
    else: # Fallback, though categorical should prevent this
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optional: Learning rate scheduler within the trial
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=False)

    print(f"Trial {trial.number}: lr={lr:.2e}, batch={batch_size}, opt={optimizer_name}, wd={weight_decay:.2e}, f_alpha={focal_alpha:.2f}, f_gamma={focal_gamma:.2f}, loss_beta={loss_beta:.2f}, dice_smooth={dice_smooth:.2e}")

    best_trial_val_loss = float('inf')
    for epoch in range(1, current_epochs_for_trial + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        # scheduler.step(val_loss) # If scheduler is used
        
        # Optuna pruning (optional, but good for long trials)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()
        
        if val_loss < best_trial_val_loss:
            best_trial_val_loss = val_loss

        print(f"  Epoch {epoch}/{current_epochs_for_trial} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return best_trial_val_loss # Return the best validation loss achieved in this trial

# --- Main Training Loop ---
def main(args):
    print(f"Using device: {DEVICE}")
    checkpoint_dir = os.path.join(args.root_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        target_h, target_w = map(int, args.target_size.split("x"))
        target_size = (target_h, target_w)
    except ValueError:
        print("Error: target_size must be in format HxW (e.g., 128x128)")
        return
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
    print("Initializing model...")
    model = SiameseUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)
    criterion = FocalDiceLoss(focal_alpha=0.6030489822904476, focal_gamma=1.7930869982898021, beta=0.6699803915247974, dice_smooth=1.956571276926647e-06).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.1180726948943663e-05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=7) # Adjusted factor and patience, removed verbose
    print("Using AdamW optimizer and ReduceLROnPlateau LR scheduler with updated parameters.")
    print("Starting training...")
    best_val_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.num_epochs} - LR: {current_lr:.1e}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
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
    parser.add_argument("--root-dir", type=str, default=ROOT_DIR, help="Root project directory")
    parser.add_argument("--dataset-subdir", type=str, default=DATASET_SUBDIR_DEFAULT, help="Subdirectory for the Onera dataset")
    parser.add_argument("--synthetic-data-dir", type=str, default=SYNTHETIC_DATA_DIR_DEFAULT, help="Directory for synthetic data")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR_DEFAULT, help="Directory to save model checkpoints")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS_DEFAULT, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.00010152447097322304, help="Initial learning rate")
    parser.add_argument("--target-size", type=str, default=f"{TARGET_SIZE_DEFAULT[0]}x{TARGET_SIZE_DEFAULT[1]}", help="Target image size HxW (e.g., 128x128)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY_DEFAULT, help="Save checkpoint every N epochs")
    parser.add_argument("--use-synthetic", action="store_true", default=USE_SYNTHETIC_DEFAULT, help="Include synthetic data during training")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    args = parser.parse_args()
    if args.tune:
        # Optuna Study Configuration
        storage_name = "sqlite:///optuna_study.db" # For persistent storage
        study_name = "siamese_unet_tuning_v3" # Increment version for new studies
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction='minimize',
            load_if_exists=True, # Load previous results if study_name exists
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        )
        # You can query the number of finished trials to decide if you want to run more
        # n_already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        # n_trials_to_run = 50 - n_already_done
        # if n_trials_to_run > 0:
        #    study.optimize(objective, n_trials=n_trials_to_run, timeout=3600*6) # e.g., run for 50 trials or 6 hours
        study.optimize(objective, n_trials=50, n_jobs=1) # n_jobs=1 for simplicity, can be >1 if resources allow and code is safe
        
        print("\n--- Optuna Study Complete ---")
        print(f"Study name: {study.study_name}")
        print(f"Number of finished trials: {len(study.trials)}")
        
        best_trial = study.best_trial
        print(f"Best trial number: {best_trial.number}")
        print(f"Best validation loss: {best_trial.value:.4f}")
        print("Best hyperparameters:")
        for key, value in best_trial.params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4e}") # Scientific notation for floats
            else:
                print(f"  {key}: {value}")
        
        # Optional: Save study results or plot them
        # fig = optuna.visualization.plot_optimization_history(study)
        # fig.show()
        # fig = optuna.visualization.plot_slice(study)
        # fig.show()
    else:
        main(args)