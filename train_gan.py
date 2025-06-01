# Training script for Pix2Pix GAN for Change Data Augmentation
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

# Assuming models.py and dataset.py are in the same directory or accessible
from dataset import BaseChangeDetectionDataset, scan_dataset, collate_fn_skip_none # Import scan_dataset and collate_fn_skip_none
from models import UNetGenerator, NLayerDiscriminator

# --- Configuration ---
ROOT_DIR = "/Users/mac/Desktop/MAYNA/Code/Change_Detection_Package"  # User specified root
DATASET_SUBDIR = "Onera Satellite Change Detection Dataset"
# Define correct paths to image and label directories
IMAGES_DATA_DIR = os.path.join(ROOT_DIR, DATASET_SUBDIR, "images", "Onera Satellite Change Detection dataset - Images")
LABELS_DATA_DIR = os.path.join(ROOT_DIR, DATASET_SUBDIR, "train_labels", "Onera Satellite Change Detection dataset - Train Labels")

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "gan_checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "gan_samples")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # Pix2Pix often uses batch size 1
NUM_EPOCHS = 200  # Increased for better quality
LEARNING_RATE_G = 1e-4  # Adjusted learning rate
LEARNING_RATE_D = 1e-4  # Adjusted learning rate
BETA1 = 0.5
N_CHANNELS = 3  # Fixed for RGB
TARGET_SIZE = (256, 256)
LAMBDA_L1 = 100.0
SAVE_EVERY = 10
SAMPLE_EVERY = 5

print(f"Using device: {DEVICE}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Loss Functions ---
loss_GAN = nn.BCEWithLogitsLoss()
loss_L1 = nn.L1Loss()

# --- Training Function ---
def train_gan_one_epoch(gen, disc, loader, opt_g, opt_d):
    gen.train()
    disc.train()
    epoch_loss_g = 0.0
    epoch_loss_d = 0.0
    progress_bar = tqdm(loader, desc="GAN Training", leave=False)
    for batch in progress_bar:
        real_A = batch["image1"].to(DEVICE)
        real_B = batch["image2"].to(DEVICE)
        opt_d.zero_grad()
        fake_B = gen(real_A).detach()
        pred_real = disc(torch.cat((real_A, real_B), 1))
        loss_d_real = loss_GAN(pred_real, torch.ones_like(pred_real))
        pred_fake = disc(torch.cat((real_A, fake_B), 1))
        loss_d_fake = loss_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        opt_d.step()
        opt_g.zero_grad()
        fake_B_for_g = gen(real_A)
        pred_fake_for_g = disc(torch.cat((real_A, fake_B_for_g), 1))
        loss_g_gan = loss_GAN(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
        loss_g_l1 = loss_L1(fake_B_for_g, real_B) * LAMBDA_L1
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        opt_g.step()
        epoch_loss_g += loss_g.item()
        epoch_loss_d += loss_d.item()
        progress_bar.set_postfix(Loss_D=loss_d.item(), Loss_G=loss_g.item())
    return epoch_loss_d / len(loader), epoch_loss_g / len(loader)

# --- Save Sample Images ---
def save_samples(gen, loader, epoch, output_dir):
    gen.eval()
    data = next(iter(loader))
    real_A = data["image1"].to(DEVICE)
    real_B = data["image2"].to(DEVICE)
    city = data["city"][0]
    with torch.no_grad():
        fake_B = gen(real_A)
    real_A = (real_A * 0.5) + 0.5
    real_B = (real_B * 0.5) + 0.5
    fake_B = (fake_B * 0.5) + 0.5
    img_grid = torch.cat((real_A[0], fake_B[0], real_B[0]), dim=-1)
    save_path = os.path.join(output_dir, f"sample_{city}_epoch_{epoch:03d}.png")
    TF.to_pil_image(img_grid.cpu()).save(save_path)
    print(f"Saved sample image to {save_path}")

# --- Main GAN Training Loop ---
def main_gan():
    print("Loading GAN datasets...")

    # Scan for training samples
    samples_list_train = scan_dataset(
        data_dir=IMAGES_DATA_DIR,
        label_dir=LABELS_DATA_DIR, # GANs might not always need labels for input A, but if B is a label, it's needed
        is_synthetic=False # Assuming we train GAN on real data to generate synthetic-like B from real A
    )
    # Filter for 'train' cities (or all if not splitting by city for GAN)
    # This part depends on how you want to split data for GAN training.
    # For simplicity, let's assume all scanned real data is for training the GAN.
    # You might need to adapt city filtering logic from train.py if needed.
    train_dataset = BaseChangeDetectionDataset(samples_list=samples_list_train, target_size=TARGET_SIZE, augment=False) # Use augment=True if you want GAN to learn from augmented data

    # Scan for validation/sampling samples
    # Typically, for GANs, you might just sample from the training distribution or a small fixed set
    # If you have a dedicated validation set of (ImageA, ImageB) pairs:
    samples_list_val = scan_dataset(
        data_dir=IMAGES_DATA_DIR, # Assuming same source for val images
        label_dir=LABELS_DATA_DIR, # Assuming same source for val labels
        is_synthetic=False
    )
    # Filter for 'val' cities if applicable, or use a subset of train_dataset for sampling
    # Again, adapting city filtering might be needed.
    # For now, let's use the same list for simplicity, or a subset.
    # Ensure sample_dataset is not empty for save_samples
    if samples_list_val: # Or use a subset of samples_list_train
        sample_dataset = BaseChangeDetectionDataset(samples_list=samples_list_val, target_size=TARGET_SIZE, augment=False)
    else: # Fallback if val scan is empty or not configured
        sample_dataset = BaseChangeDetectionDataset(samples_list=samples_list_train[:max(1, len(samples_list_train)//10)], target_size=TARGET_SIZE, augment=False) # Use a small part of train

    if len(train_dataset) == 0:
        print("Error: GAN Training dataset is empty. Check dataset path and structure.")
        return
    if len(sample_dataset) == 0:
        print("Warning: GAN Sampling (validation) dataset is empty. Using training set for samples.")
        sample_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_skip_none)
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_skip_none)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn_skip_none)
    print(f"GAN Dataset loaded: {len(train_dataset)} train samples.")
    print("Initializing GAN models...")
    generator = UNetGenerator(input_nc=N_CHANNELS, output_nc=N_CHANNELS).to(DEVICE)
    discriminator = NLayerDiscriminator(input_nc=N_CHANNELS * 2).to(DEVICE)
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))
    print("Starting GAN training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        loss_d, loss_g = train_gan_one_epoch(generator, discriminator, train_loader, optimizer_G, optimizer_D)
        print(f"Epoch {epoch} - Avg Loss D: {loss_d:.4f}, Avg Loss G: {loss_g:.4f}")
        if epoch % SAMPLE_EVERY == 0 or epoch == NUM_EPOCHS:
            save_samples(generator, sample_loader, epoch, OUTPUT_DIR)
        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            checkpoint_path_g = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.pth")
            checkpoint_path_d = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.pth")
            torch.save(generator.state_dict(), checkpoint_path_g)
            torch.save(discriminator.state_dict(), checkpoint_path_d)
            print(f"GAN Checkpoints saved for epoch {epoch}")
    print("GAN Training finished.")

if __name__ == "__main__":
    main_gan()