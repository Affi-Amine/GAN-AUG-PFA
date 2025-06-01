# Evaluation script for Change Detection Model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Assuming models.py and dataset.py are accessible
from dataset import BaseChangeDetectionDataset, scan_dataset # Import scan_dataset
from models import SiameseUNet

# --- Configuration ---
ROOT_DIR = "/Users/mac/Desktop/MAYNA/Code/Change_Detection_Package"  # User specified root
# Define specific paths for images and labels, accounting for nested structure
IMAGES_DATA_DIR = os.path.join(ROOT_DIR, "Onera Satellite Change Detection Dataset", "images", "Onera Satellite Change Detection dataset - Images")
LABELS_DATA_DIR = os.path.join(ROOT_DIR, "Onera Satellite Change Detection Dataset", "train_labels", "Onera Satellite Change Detection dataset - Train Labels")

CHECKPOINT_PATH = os.path.join(ROOT_DIR, "siamese_checkpoints", "best_model.pth") # Default to best model
OUTPUT_DIR = os.path.join(ROOT_DIR, "evaluation_results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2 # Use the same batch size as training or smaller if needed
# N_CHANNELS is now fixed to 3 for RGB PNGs
N_CHANNELS = 3
N_CLASSES = 1 # Binary change map
TARGET_SIZE = (128, 128) # Must match the size used for training
NUM_VISUALIZATIONS = 5 # Number of validation samples to visualize

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# --- Metrics Calculation ---
def calculate_metrics(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float() # Apply threshold to get binary predictions
    preds = preds.view(-1)
    targets = targets.view(-1)

    # True Positives, False Positives, False Negatives
    tp = (preds * targets).sum()
    fp = ((1 - targets) * preds).sum()
    fn = (targets * (1 - preds)).sum()
    tn = ((1 - targets) * (1 - preds)).sum()

    # Precision, Recall, F1 Score
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    # Intersection over Union (IoU)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    # Accuracy
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item()
    }

# --- Visualization Function ---
def visualize_sample(img1, img2, label, pred, city_name, index, output_dir):
    # Convert tensors to numpy arrays for plotting
    # Select RGB bands if using all bands
    if img1.shape[0] == 13:
        img1_rgb = img1[[3, 2, 1], :, :].cpu().numpy().transpose(1, 2, 0) # Select B4, B3, B2
        img2_rgb = img2[[3, 2, 1], :, :].cpu().numpy().transpose(1, 2, 0)
    else:
        img1_rgb = img1.cpu().numpy().transpose(1, 2, 0)
        img2_rgb = img2.cpu().numpy().transpose(1, 2, 0)

    label_np = label.cpu().numpy()
    pred_np = (pred.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8) # Apply threshold

    # Clip values for display if normalization was basic
    img1_rgb = np.clip(img1_rgb, 0, 1)
    img2_rgb = np.clip(img2_rgb, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Validation Sample {index} - City: {city_name}")

    axes[0].imshow(img1_rgb)
    axes[0].set_title("Image 1 (RGB)")
    axes[0].axis("off")

    axes[1].imshow(img2_rgb)
    axes[1].set_title("Image 2 (RGB)")
    axes[1].axis("off")

    axes[2].imshow(label_np, cmap="gray")
    axes[2].set_title("Ground Truth Change")
    axes[2].axis("off")

    axes[3].imshow(pred_np, cmap="gray")
    axes[3].set_title("Predicted Change")
    axes[3].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    save_path = os.path.join(output_dir, f"validation_sample_{city_name}_{index}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

# --- Evaluation Function ---
def evaluate_model(model, loader, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    total_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
    num_samples = 0
    visualized_count = 0

    progress_bar = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        print(f"Starting evaluation loop with {len(loader)} batches.") # Added log
        for i, batch in enumerate(progress_bar):
            print(f"Processing batch {i+1}/{len(loader)}") # Added log
            img1 = batch["image1"].to(DEVICE)
            img2 = batch["image2"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            cities = batch["city"]

            outputs = model(img1, img2)
            preds = torch.sigmoid(outputs) # Apply sigmoid to get probabilities

            # Store predictions and labels for overall metrics
            # Note: For large datasets, store metrics per batch instead of all preds/labels
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # Calculate metrics per batch
            batch_metrics = calculate_metrics(preds.cpu(), labels.cpu())
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key] * len(cities)
            num_samples += len(cities)

            # Visualize some samples
            print(f"  Batch {i+1}: Visualizing up to {NUM_VISUALIZATIONS} samples if conditions met.") # Added log
            for j in range(len(cities)):
                if visualized_count < NUM_VISUALIZATIONS:
                    print(f"    Visualizing sample {visualized_count + 1} (City: {cities[j]}, Index in batch: {j})") # Added log
                    visualize_sample(img1[j], img2[j], labels[j], preds[j], cities[j], visualized_count, output_dir)
                    visualized_count += 1

    # Calculate average metrics
    avg_metrics = {key: val / num_samples for key, val in total_metrics.items()}

    print(f"Finished evaluation loop. Processed {num_samples} samples across {len(loader)} batches.") # Added log
    print("\n--- Evaluation Metrics ---")
    for key, val in avg_metrics.items():
        print(f"{key.capitalize()}: {val:.4f}")

    # Optionally calculate metrics over the entire dataset (if memory allows)
    # all_preds_tensor = torch.cat(all_preds, dim=0)
    # all_labels_tensor = torch.cat(all_labels, dim=0)
    # overall_metrics = calculate_metrics(all_preds_tensor, all_labels_tensor)
    # print("\n--- Overall Metrics (Calculated on all samples) ---")
    # for key, val in overall_metrics.items():
    #     print(f"{key.capitalize()}: {val:.4f}")

    return avg_metrics

# --- Main Evaluation ---
def main():
    # Dataset and Dataloader for Validation set
    print("Loading validation dataset...")

    # Scan for validation samples
    print("--- Scanning Real Data (val) ---")
    print(f"Starting scan_dataset with data_dir: {IMAGES_DATA_DIR} and label_dir: {LABELS_DATA_DIR}") # Added log
    # Adjust city_list for validation as needed, or remove if all cities are used for validation
    # For now, assuming all found samples in the specified directories are for validation
    samples_list_val = scan_dataset(
        data_dir=IMAGES_DATA_DIR,  # Corrected: images_dir -> data_dir
        label_dir=LABELS_DATA_DIR, # Corrected: labels_dir -> label_dir
        is_synthetic=False # Assuming evaluation is on real data
        # dataset_type and mode are not params of scan_dataset, filtering might be needed post-scan
    )
    print(f"scan_dataset finished. Found {len(samples_list_val) if samples_list_val else 0} validation samples.") # Added log

    if not samples_list_val:
        print("Error: No validation samples found. Check dataset paths and structure.")
        return

    # Use the updated ChangeDetectionDataset
    val_dataset = BaseChangeDetectionDataset(
        samples_list=samples_list_val,
        target_size=TARGET_SIZE,
        augment=False # No augmentation for validation
    )

    if len(val_dataset) == 0:
        print("Error: Validation dataset is empty. Check dataset path, structure, and mode.")
        return

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Validation dataset loaded: {len(val_dataset)} samples.")

    # Load Model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = SiameseUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # Evaluate
    evaluate_model(model, val_loader, OUTPUT_DIR)

if __name__ == "__main__":
    # Need to install matplotlib if not present
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib not found. Installing...")
        os.system("pip3 install matplotlib")
        import matplotlib
    matplotlib.use("Agg") # Use non-interactive backend suitable for scripts
    import matplotlib.pyplot as plt
    main()


