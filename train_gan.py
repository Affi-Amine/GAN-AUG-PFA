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
from dataset import ChangeDetectionDataset # Use the updated dataset class directly
from models import UNetGenerator, NLayerDiscriminator # Import GAN components

# --- Configuration ---
# Adjust ROOT_DIR based on user's structure
ROOT_DIR = "/Users/mac/Desktop/MAYNA/Code/Change_Detection_Package" # User specified root
DATASET_SUBDIR = "Onera Satellite Change Detection Dataset" # User specified dataset subdir
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "gan_checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "gan_samples") # To save sample generated images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1 # Pix2Pix often uses batch size 1
NUM_EPOCHS = 100 # GANs often require more epochs, adjust as needed
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
BETA1 = 0.5 # Adam optimizer beta1 for GANs
# N_CHANNELS is now fixed to 3 for RGB PNGs
N_CHANNELS = 3
TARGET_SIZE = (256, 256) # Target size for GAN training (can be different from Siamese)
LAMBDA_L1 = 100.0 # Weight for L1 loss in Generator objective
SAVE_EVERY = 10 # Save checkpoint every N epochs
SAMPLE_EVERY = 5 # Save sample images every N epochs

print(f"Using device: {DEVICE}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Dataset Adaptation (No longer needed, use ChangeDetectionDataset directly) ---
# class GANTrainingDataset(ChangeDetectionDataset): ... (Removed)

# --- Loss Functions ---
loss_GAN = nn.BCEWithLogitsLoss() # Adversarial loss
loss_L1 = nn.L1Loss() # Pixel-wise L1 loss

# --- Training Function ---
def train_gan_one_epoch(gen, disc, loader, opt_g, opt_d):
    gen.train()
    disc.train()
    epoch_loss_g = 0.0
    epoch_loss_d = 0.0
    progress_bar = tqdm(loader, desc="GAN Training", leave=False)
    
    for batch in progress_bar:
        real_A = batch["image1"].to(DEVICE)  # Input image (Image 1)
        real_B = batch["image2"].to(DEVICE) # Target image (Image 2)

        # --- Train Discriminator ---
        opt_d.zero_grad()

        # Generate fake image
        fake_B = gen(real_A).detach() # Detach to avoid gradients flowing to Generator

        # Discriminator loss for real images (real_A paired with real_B)
        pred_real = disc(torch.cat((real_A, real_B), 1))
        loss_d_real = loss_GAN(pred_real, torch.ones_like(pred_real))

        # Discriminator loss for fake images (real_A paired with fake_B)
        pred_fake = disc(torch.cat((real_A, fake_B), 1))
        loss_d_fake = loss_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        opt_d.step()

        # --- Train Generator ---
        opt_g.zero_grad()

        # Generate fake image (without detaching this time)
        fake_B_for_g = gen(real_A)

        # Adversarial loss for Generator (wants Discriminator to think fake is real)
        pred_fake_for_g = disc(torch.cat((real_A, fake_B_for_g), 1))
        loss_g_gan = loss_GAN(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

        # L1 loss (pixel-wise difference between generated and real target)
        loss_g_l1 = loss_L1(fake_B_for_g, real_B) * LAMBDA_L1

        # Total generator loss
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
    data = next(iter(loader)) # Get one batch
    real_A = data["image1"].to(DEVICE)
    real_B = data["image2"].to(DEVICE)
    city = data["city"][0] # Get city name for filename

    with torch.no_grad():
        fake_B = gen(real_A)

    # Unnormalize images from [-1, 1] to [0, 1] for saving
    real_A = (real_A * 0.5) + 0.5
    real_B = (real_B * 0.5) + 0.5
    fake_B = (fake_B * 0.5) + 0.5

    # Save images (take first image in batch)
    img_grid = torch.cat((real_A[0], fake_B[0], real_B[0]), dim=-1) # Concatenate horizontally
    save_path = os.path.join(output_dir, f"sample_{city}_epoch_{epoch:03d}.png")
    TF.to_pil_image(img_grid.cpu()).save(save_path)
    print(f"Saved sample image to {save_path}")

# --- Main GAN Training Loop ---
def main_gan():
    # Datasets and Dataloaders
    print("Loading GAN datasets...")
    # Use the updated ChangeDetectionDataset directly
    train_dataset = ChangeDetectionDataset(root_dir=ROOT_DIR, dataset_subdir=DATASET_SUBDIR, mode="train", target_size=TARGET_SIZE)
    # Use validation set for sampling if desired, or just use train set
    # Ensure val set also uses the correct target size for consistency if used for sampling
    sample_dataset = ChangeDetectionDataset(root_dir=ROOT_DIR, dataset_subdir=DATASET_SUBDIR, mode="val", target_size=TARGET_SIZE)

    if len(train_dataset) == 0:
        print("Error: GAN Training dataset is empty. Check dataset path and structure.")
        print(f"Expected structure under: {os.path.join(ROOT_DIR, DATASET_SUBDIR)}")
        return
    if len(sample_dataset) == 0:
        print("Warning: GAN Sampling (validation) dataset is empty. Using training set for samples.")
        sample_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # Fallback to train data
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=True) # Loader for generating samples

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    print(f"GAN Dataset loaded: {len(train_dataset)} train samples.")

    # Models
    print("Initializing GAN models...")
    # Generator: Input N_CHANNELS (Image 1), Output N_CHANNELS (Image 2)
    generator = UNetGenerator(input_nc=N_CHANNELS, output_nc=N_CHANNELS).to(DEVICE)
    # Discriminator: Input N_CHANNELS + N_CHANNELS (Image 1 + Image 2)
    discriminator = NLayerDiscriminator(input_nc=N_CHANNELS * 2).to(DEVICE)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))

    # Training loop
    print("Starting GAN training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        # Pass the base dataset loader, normalization happens inside the loop
        loss_d, loss_g = train_gan_one_epoch(generator, discriminator, train_loader, optimizer_G, optimizer_D)
        print(f"Epoch {epoch} - Avg Loss D: {loss_d:.4f}, Avg Loss G: {loss_g:.4f}")

        # Save sample images
        if epoch % SAMPLE_EVERY == 0 or epoch == NUM_EPOCHS:
            save_samples(generator, sample_loader, epoch, OUTPUT_DIR)

        # Save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            checkpoint_path_g = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.pth")
            checkpoint_path_d = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.pth")
            torch.save(generator.state_dict(), checkpoint_path_g)
            torch.save(discriminator.state_dict(), checkpoint_path_d)
            print(f"GAN Checkpoints saved for epoch {epoch}")

    print("GAN Training finished.")

if __name__ == "__main__":
    main_gan()

