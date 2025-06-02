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
    # Ensure label is a tensor, if it's a PIL Image, convert it (already handled in evaluate_single_pair by passing tensor)
    # if isinstance(label, Image.Image):
    #     label = T.ToTensor()(label)
    # Convert tensors to numpy arrays for plotting
    # Select RGB bands if using all bands
    if img1.shape[0] == 13:
        img1_rgb = img1[[3, 2, 1], :, :].cpu().numpy().transpose(1, 2, 0) # Select B4, B3, B2
        img2_rgb = img2[[3, 2, 1], :, :].cpu().numpy().transpose(1, 2, 0)
    else:
        img1_rgb = img1.cpu().numpy().transpose(1, 2, 0)
        img2_rgb = img2.cpu().numpy().transpose(1, 2, 0)

    # Handle cases where label might be None or not a tensor with data
    if label is not None and hasattr(label, 'cpu') and label.numel() > 0: # Check if tensor has elements
        label_np = label.cpu().numpy()
        if label_np.ndim == 3 and label_np.shape[0] == 1: # Ensure it's [H, W] for imshow
            label_np = label_np.squeeze(0)
        elif label_np.ndim == 2: # Already [H,W]
            pass # It's fine
        else:
            # Fallback for unexpected label shape
            h, w = pred.shape[-2:] if pred is not None and pred.ndim >=2 else TARGET_SIZE
            label_np = np.zeros((h, w), dtype=np.uint8)
            print(f"Warning: Unexpected label shape {label.shape}, displaying empty ground truth.")
    else:
        # Create a dummy black image if no label for visualization consistency
        h, w = pred.shape[-2:] if pred is not None and pred.ndim >=2 else TARGET_SIZE
        label_np = np.zeros((h, w), dtype=np.uint8)

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
    per_city_metrics = {}
    per_city_counts = {}
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

            # Calculate and accumulate metrics
            for k in range(len(cities)):
                city = cities[k]
                pred_sample = preds[k:k+1].cpu()
                label_sample = labels[k:k+1].cpu()
                sample_metrics = calculate_metrics(pred_sample, label_sample)

                if city not in per_city_metrics:
                    per_city_metrics[city] = {key: 0.0 for key in sample_metrics}
                    per_city_counts[city] = 0
                
                for key in sample_metrics:
                    per_city_metrics[city][key] += sample_metrics[key]
                    total_metrics[key] += sample_metrics[key] # Accumulate for overall average
                per_city_counts[city] += 1
            num_samples += len(cities)

            # Visualize some samples
            print(f"  Batch {i+1}: Visualizing up to {NUM_VISUALIZATIONS} samples if conditions met.") # Added log
            for j in range(len(cities)):
                if visualized_count < NUM_VISUALIZATIONS:
                    print(f"    Visualizing sample {visualized_count + 1} (City: {cities[j]}, Index in batch: {j})") # Added log
                    visualize_sample(img1[j], img2[j], labels[j], preds[j], cities[j], visualized_count, output_dir)
                    visualized_count += 1

    # Calculate average overall metrics
    avg_overall_metrics = {key: val / num_samples for key, val in total_metrics.items() if num_samples > 0}

    print(f"Finished evaluation loop. Processed {num_samples} samples across {len(loader)} batches.")
    print("\n--- Overall Evaluation Metrics ---")
    for key, val in avg_overall_metrics.items():
        print(f"{key.capitalize()}: {val:.4f}")

    print("\n--- Per-City Evaluation Metrics ---")
    for city, metrics in per_city_metrics.items():
        count = per_city_counts[city]
        if count > 0:
            avg_city_metrics = {key: val / count for key, val in metrics.items()}
            print(f"City: {city} (Samples: {count})")
            for key, val in avg_city_metrics.items():
                print(f"  {key.capitalize()}: {val:.4f}")
        else:
            print(f"City: {city} (Samples: 0) - No metrics calculated")

    # Optionally calculate metrics over the entire dataset (if memory allows)
    # all_preds_tensor = torch.cat(all_preds, dim=0)
    # all_labels_tensor = torch.cat(all_labels, dim=0)
    # overall_metrics = calculate_metrics(all_preds_tensor, all_labels_tensor)
    # print("\n--- Overall Metrics (Calculated on all samples) ---")
    # for key, val in overall_metrics.items():
    #     print(f"{key.capitalize()}: {val:.4f}")

    return avg_overall_metrics

from PIL import Image
import torchvision.transforms as T

# --- Function to evaluate a single pair of images ---
def evaluate_single_pair(model, img1_path, img2_path, city_name, label_path=None, target_size=(128, 128), device="cpu", output_dir="evaluation_results"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Assuming ImageNet normalization for RGB
    ])

    try:
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: One or both image paths not found: {img1_path}, {img2_path}")
        return

    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    label_tensor_resized = None
    label_for_viz = None # This will be a PIL Image or None

    if label_path:
        try:
            label_pil = Image.open(label_path).convert("L") # Grayscale PIL Image
            
            # For visualization, resize PIL image
            label_for_viz = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)(label_pil)
            
            # For metrics, transform to tensor and resize
            label_transform_metric = T.Compose([
                T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor() # This will be [C, H, W]
            ])
            label_tensor_resized = label_transform_metric(label_pil).unsqueeze(0).to(device) # [B, C, H, W]

        except FileNotFoundError:
            print(f"Warning: Label path not found: {label_path}. Proceeding without metrics.")
            label_path = None # Ensure it's None if file not found

    with torch.no_grad():
        output = model(img1_tensor, img2_tensor)
        pred = torch.sigmoid(output) # Apply sigmoid

    # Prepare label for visualization (even if None)
    # visualize_sample expects a tensor for label
    if label_for_viz is not None:
        label_viz_tensor = T.ToTensor()(label_for_viz) # Convert PIL to Tensor
    else:
        # Create a dummy black tensor if no label for visualization consistency
        h, w = pred.shape[-2:] if pred is not None and pred.ndim >=2 else target_size
        label_viz_tensor = torch.zeros((1, h, w), dtype=torch.float32) # [C, H, W]

    print(f"Visualizing single pair for city: {city_name}")
    visualize_sample(img1_tensor.squeeze(0), img2_tensor.squeeze(0), 
                     label_viz_tensor, # Pass the resized tensor or dummy
                     pred.squeeze(0), 
                     city_name, "single_eval", output_dir)

    if label_tensor_resized is not None and label_path:
        metrics = calculate_metrics(pred.cpu(), label_tensor_resized.cpu())
        print(f"\n--- Metrics for {city_name} ({os.path.basename(img1_path)}, {os.path.basename(img2_path)}) ---")
        for key, val in metrics.items():
            print(f"{key.capitalize()}: {val:.4f}")
    elif not label_path:
        print("No label path provided, skipping metrics calculation.")
    # Implicitly, if label_path was provided but file not found, it's handled by the earlier print and label_path being set to None


# --- Main Evaluation ---
def main(args):
    # Check if running in single pair evaluation mode
    if args.image1_path and args.image2_path and args.city_name:
        print(f"Evaluating single image pair for city: {args.city_name}")
        model = SiameseUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)
        if os.path.exists(CHECKPOINT_PATH):
            try:
                model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
                print(f"Model loaded from {CHECKPOINT_PATH}")
            except Exception as e:
                print(f"Error loading model state_dict: {e}")
                return
        else:
            print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}. Cannot evaluate single pair.")
            return
        evaluate_single_pair(model, args.image1_path, args.image2_path, args.city_name, 
                             label_path=args.label_path, target_size=TARGET_SIZE, 
                             device=DEVICE, output_dir=OUTPUT_DIR)
        return # Exit after single pair evaluation

    # Original main logic for full dataset evaluation
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
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Change Detection Model")
    parser.add_argument("--image1-path", type=str, help="Path to the first image (before change)")
    parser.add_argument("--image2-path", type=str, help="Path to the second image (after change)")
    parser.add_argument("--label-path", type=str, default=None, help="Optional path to the ground truth change mask")
    parser.add_argument("--city-name", type=str, help="Name of the city/area for identification in output")
    # Add existing args if any, or ensure they don't conflict
    # For example, if CHECKPOINT_PATH was an arg:
    # parser.add_argument("--checkpoint-path", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")

    # Need to install matplotlib if not present
    try:
        import matplotlib
        matplotlib.use("Agg") # Use non-interactive backend suitable for scripts
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Please install it manually: pip install matplotlib")
        # os.system("pip3 install matplotlib") # Avoid auto-installing in general purpose tool
        # import matplotlib # Try importing again
        # matplotlib.use("Agg") 
        # import matplotlib.pyplot as plt
        # For now, if matplotlib is not there, visualization will fail later but core logic might run.
        # Or, exit if matplotlib is critical.
        print("Matplotlib is required for visualization. Please install it and try again.")
        exit(1)
        
    args = parser.parse_args()
    main(args)


