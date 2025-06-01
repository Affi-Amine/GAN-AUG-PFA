import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

# Assuming models.py and dataset.py are accessible
from dataset import BaseChangeDetectionDataset, scan_dataset, collate_fn_skip_none
from models import UNetGenerator

# --- Configuration ---
ROOT_DIR = "/Users/mac/Desktop/MAYNA/Code/Change_Detection_Package"  # User specified root
DATASET_SUBDIR = "Onera Satellite Change Detection Dataset"
IMAGES_DATA_DIR = os.path.join(ROOT_DIR, DATASET_SUBDIR, "images", "Onera Satellite Change Detection dataset - Images")
LABELS_DATA_DIR = os.path.join(ROOT_DIR, DATASET_SUBDIR, "train_labels", "Onera Satellite Change Detection dataset - Train Labels")
GAN_CHECKPOINT_DIR = os.path.join(ROOT_DIR, "gan_checkpoints")
GENERATOR_CHECKPOINT_NAME = "generator_epoch_200.pth"  # Use the checkpoint from 200 epochs
SYNTHETIC_DATA_DIR = os.path.join(ROOT_DIR, "synthetic_data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
TARGET_SIZE = (256, 256)  # Must match GAN training size
N_CHANNELS = 3
NUM_WORKERS = 2

# Create output directories
os.makedirs(os.path.join(SYNTHETIC_DATA_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SYNTHETIC_DATA_DIR, "labels"), exist_ok=True)

print(f"Using device: {DEVICE}")

# --- Main Generation Function ---
def generate_data():
    print("Loading original dataset (train split) for generation...")
    samples_list = scan_dataset(data_dir=IMAGES_DATA_DIR, label_dir=LABELS_DATA_DIR, is_synthetic=False)
    original_dataset = BaseChangeDetectionDataset(samples_list=samples_list, target_size=TARGET_SIZE, augment=False)
    if len(original_dataset) == 0:
        print("Error: Original training dataset is empty. Cannot generate synthetic data.")
        return
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
        return
    except Exception as e:
        print(f"Error loading generator state_dict: {e}")
        return
    generator.eval()

    # Generation Loop
    print("Starting synthetic data generation...")
    generated_count = 0
    progress_bar = tqdm(data_loader, desc="Generating Synthetic Data")
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            if batch is None:
                continue
            real_img1 = batch["image1"].to(DEVICE)
            real_labels = batch["label"]
            cities = batch["city"]
            synthetic_img2 = generator(real_img1)
            for j in range(real_img1.size(0)):
                img1_save = (real_img1[j] * 0.5) + 0.5
                img2_save = (synthetic_img2[j] * 0.5) + 0.5
                label_save = real_labels[j]
                city_name = cities[j]
                sample_idx = i * BATCH_SIZE + j
                city_img_dir = os.path.join(SYNTHETIC_DATA_DIR, "images", city_name)
                city_lbl_dir = os.path.join(SYNTHETIC_DATA_DIR, "labels", city_name)
                os.makedirs(city_img_dir, exist_ok=True)
                os.makedirs(city_lbl_dir, exist_ok=True)
                img1_save_path = os.path.join(city_img_dir, f"img1_synth_{sample_idx}.png")
                img2_save_path = os.path.join(city_img_dir, f"img2_synth_{sample_idx}.png")
                label_save_path = os.path.join(city_lbl_dir, f"cm_synth_{sample_idx}.png")
                try:
                    TF.to_pil_image(img1_save.cpu()).save(img1_save_path)
                    TF.to_pil_image(img2_save.cpu()).save(img2_save_path)
                    TF.to_pil_image(label_save.cpu().byte() * 255).save(label_save_path)
                    generated_count += 1
                except Exception as e:
                    print(f"Error saving sample {sample_idx} for city {city_name}: {e}")
    print(f"\nSynthetic data generation finished. Saved {generated_count} samples to {SYNTHETIC_DATA_DIR}")

if __name__ == "__main__":
    generate_data()