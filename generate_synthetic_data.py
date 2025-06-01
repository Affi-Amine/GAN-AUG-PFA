# Script to generate synthetic data using a trained Pix2Pix GAN generator
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

# Assuming models.py and dataset.py are accessible
from dataset import ChangeDetectionDataset, collate_fn_skip_none # Import dataset and collate function
from models import UNetGenerator # Import the Generator model

# --- Configuration ---
ROOT_DIR = "/Users/mac/Desktop/MAYNA/Code/Change_Detection_Package" # User specified root
DATASET_SUBDIR = "Onera Satellite Change Detection Dataset"
GAN_CHECKPOINT_DIR = os.path.join(ROOT_DIR, "gan_checkpoints")
GENERATOR_CHECKPOINT_NAME = "generator_epoch_100.pth" # Assumes 100 epochs were trained, adjust if needed
SYNTHETIC_DATA_DIR = os.path.join(ROOT_DIR, "synthetic_data") # Output directory for generated data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4 # Adjust based on memory
TARGET_SIZE = (256, 256) # Must match the size the GAN was trained with
N_CHANNELS = 3 # Fixed for RGB
NUM_WORKERS = 2

# Create output directories
os.makedirs(os.path.join(SYNTHETIC_DATA_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SYNTHETIC_DATA_DIR, "labels"), exist_ok=True)

print(f"Using device: {DEVICE}")

# --- Main Generation Function ---
def generate_data():
    # Load Dataset (using the training split to generate synthetic pairs)
    print("Loading original dataset (train split) for generation...")
    # Note: Dataset loads images normalized to [-1, 1] because of NormalizeTransform
    original_dataset = ChangeDetectionDataset(root_dir=ROOT_DIR, dataset_subdir=DATASET_SUBDIR, mode="train", target_size=TARGET_SIZE)

    if len(original_dataset) == 0:
        print("Error: Original training dataset is empty. Cannot generate synthetic data.")
        return

    # Use the custom collate_fn to handle potential None samples from dataset loading errors
    data_loader = DataLoader(original_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn_skip_none)
    print(f"Dataset loaded: {len(original_dataset)} original samples found.")

    # Load Generator Model
    generator_checkpoint_path = os.path.join(GAN_CHECKPOINT_DIR, GENERATOR_CHECKPOINT_NAME)
    print(f"Loading GAN generator from: {generator_checkpoint_path}")
    generator = UNetGenerator(input_nc=N_CHANNELS, output_nc=N_CHANNELS).to(DEVICE)
    try:
        generator.load_state_dict(torch.load(generator_checkpoint_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Generator checkpoint not found at {generator_checkpoint_path}")
        print("Please ensure the GAN was trained and the checkpoint exists.")
        return
    except Exception as e:
        print(f"Error loading generator state_dict: {e}")
        return
    generator.eval() # Set generator to evaluation mode

    # Generation Loop
    print("Starting synthetic data generation...")
    generated_count = 0
    progress_bar = tqdm(data_loader, desc="Generating Synthetic Data")

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            if batch is None: # Skip batch if collate_fn returned None
                print(f"Skipping batch {i} due to previous loading errors.")
                continue

            real_img1 = batch["image1"].to(DEVICE) # Input image (Image 1, already normalized to [-1, 1])
            real_labels = batch["label"] # Labels (not normalized, keep as is)
            cities = batch["city"]

            # Generate synthetic Image 2
            synthetic_img2 = generator(real_img1)

            # Process and save each sample in the batch
            for j in range(real_img1.size(0)):
                # Denormalize images from [-1, 1] to [0, 1] for saving as standard PNG
                img1_save = (real_img1[j] * 0.5) + 0.5
                img2_save = (synthetic_img2[j] * 0.5) + 0.5
                label_save = real_labels[j] # Label is already 0/1 LongTensor
                city_name = cities[j]
                sample_idx = i * BATCH_SIZE + j # Approximate index

                # Create city-specific directories if they don't exist
                city_img_dir = os.path.join(SYNTHETIC_DATA_DIR, "images", city_name)
                city_lbl_dir = os.path.join(SYNTHETIC_DATA_DIR, "labels", city_name)
                os.makedirs(city_img_dir, exist_ok=True)
                os.makedirs(city_lbl_dir, exist_ok=True)

                # Define save paths
                img1_save_path = os.path.join(city_img_dir, f"img1_synth_{sample_idx}.png")
                img2_save_path = os.path.join(city_img_dir, f"img2_synth_{sample_idx}.png")
                label_save_path = os.path.join(city_lbl_dir, f"cm_synth_{sample_idx}.png")

                # Save images and label
                try:
                    TF.to_pil_image(img1_save.cpu()).save(img1_save_path)
                    TF.to_pil_image(img2_save.cpu()).save(img2_save_path)
                    # Convert label tensor (H, W) to PIL Image (L mode)
                    # Multiply by 255 to make changes visible as white pixels
                    TF.to_pil_image(label_save.cpu().byte() * 255).save(label_save_path)
                    generated_count += 1
                except Exception as e:
                    print(f"Error saving sample {sample_idx} for city {city_name}: {e}")

    print(f"\nSynthetic data generation finished. Saved {generated_count} samples to {SYNTHETIC_DATA_DIR}")

if __name__ == "__main__":
    generate_data()

