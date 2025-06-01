# Dataset loading logic for Custom PNG Change Detection Dataset
import os
import glob
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image, UnidentifiedImageError # Import specific error
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

# --- Utility Function for Collation ---
def collate_fn_skip_none(batch):
    """Collate function that filters out None samples."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: # If all samples in batch were None
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Transforms ---
class JointToTensor:
    """Converts PIL images in sample dict to Tensors."""
    def __call__(self, sample):
        img1 = sample["image1"]
        img2 = sample["image2"]
        label = sample["label"]

        sample["image1"] = TF.to_tensor(img1)
        sample["image2"] = TF.to_tensor(img2)
        if label is not None:
            # Convert label to binary (0 or 1) - threshold might need adjustment
            label_np = np.array(label.convert("L"))
            label_np = (label_np > 128).astype(np.uint8) # Assuming non-zero pixels are change
            sample["label"] = torch.from_numpy(label_np).long() # Return as H x W tensor
        else:
            sample["label"] = None # Ensure label is None if not present

        return sample

class JointRandomHorizontalFlip:
    """Applies horizontal flip randomly to img1, img2, and label with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["image1"] = TF.hflip(sample["image1"])
            sample["image2"] = TF.hflip(sample["image2"])
            if sample["label"] is not None:
                sample["label"] = TF.hflip(sample["label"])
        return sample

class JointRandomVerticalFlip:
    """Applies vertical flip randomly to img1, img2, and label with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["image1"] = TF.vflip(sample["image1"])
            sample["image2"] = TF.vflip(sample["image2"])
            if sample["label"] is not None:
                sample["label"] = TF.vflip(sample["label"])
        return sample

# Note: Random Rotation might be complex with labels, requires careful implementation
# if needed later. Sticking to flips for now.

class JointResize:
    """Resizes img1, img2, and label in a sample dict to a target size."""
    def __init__(self, size):
        self.size = size # (h, w)

    def __call__(self, sample):
        img1 = sample["image1"]
        img2 = sample["image2"]
        label = sample["label"]

        # Resize images (expects C, H, W tensor)
        img1_resized = TF.resize(img1, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        img2_resized = TF.resize(img2, self.size, interpolation=TF.InterpolationMode.BILINEAR)

        # Resize label (expects H, W tensor)
        if label is not None:
            # Add channel dim, resize with NEAREST, remove dim
            label_resized = TF.resize(label.unsqueeze(0), self.size, interpolation=TF.InterpolationMode.NEAREST)
            label_resized = label_resized.squeeze(0)
        else:
            label_resized = None

        sample["image1"] = img1_resized
        sample["image2"] = img2_resized
        sample["label"] = label_resized
        return sample

class JointNormalize:
    """Normalizes image tensors (img1, img2) from [0, 1] to [-1, 1]."""
    def __call__(self, sample):
        sample["image1"] = (sample["image1"] * 2.0) - 1.0
        sample["image2"] = (sample["image2"] * 2.0) - 1.0
        return sample

# --- Base Dataset Class (Handles loading single source) ---
class BaseChangeDetectionDataset(Dataset):
    """Base class to load samples from a single source (real or synthetic)."""
    def __init__(self, samples_list, target_size=(128, 128), augment=False):
        self.samples = samples_list
        self.target_size = target_size
        self.augment = augment

        # Define transforms
        transform_list = [
            JointToTensor(),
            JointResize(self.target_size)
        ]
        if self.augment:
            transform_list.extend([
                JointRandomHorizontalFlip(p=0.5),
                JointRandomVerticalFlip(p=0.5),
                # Add other augmentations like rotation here if desired
            ])
        transform_list.append(JointNormalize()) # Normalize last

        self.transform = transforms.Compose(transform_list)

    def _load_pil_image(self, file_path):
        try:
            img = Image.open(file_path).convert("RGB")
            img.load() # Force loading data
            return img
        except Exception as e:
            print(f"Error loading PIL image {file_path}: {e}")
            raise

    def _load_pil_label(self, label_file):
        if label_file is None:
            return None
        try:
            label_img = Image.open(label_file).convert("L")
            label_img.load()
            return label_img
        except Exception as e:
            print(f"Error loading PIL label {label_file}: {e}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.samples):
             print(f"Error: Index {idx} out of bounds for dataset size {len(self.samples)}.")
             return None

        sample_info = self.samples[idx]

        try:
            img1_pil = self._load_pil_image(sample_info["img1"])
            img2_pil = self._load_pil_image(sample_info["img2"])
            label_pil = self._load_pil_label(sample_info["label"])

            sample = {"image1": img1_pil, "image2": img2_pil, "label": label_pil, "city": sample_info["city"]}

            if self.transform:
                sample = self.transform(sample)

            # Ensure label is None if it started as None
            if sample_info["label"] is None:
                sample["label"] = None

            return sample

        except Exception as e:
            print(f"Failed to load/transform sample for city {sample_info.get("city", "N/A")} at index {idx}: {e}. Returning None.")
            return None

# --- Helper Function to Scan for Samples ---
def scan_dataset(data_dir, label_dir=None, is_synthetic=False):
    """Scans a directory structure (real or synthetic) for valid samples."""
    samples = []
    skipped_samples = 0
    image_folders = glob.glob(os.path.join(data_dir, "*")) # List cities/folders

    for city_folder in image_folders:
        if not os.path.isdir(city_folder):
            continue
        city_name = os.path.basename(city_folder)

        if is_synthetic:
            # Synthetic structure: images/<city>/img1_synth_*.png, labels/<city>/cm_synth_*.png
            img1_files = sorted(glob.glob(os.path.join(city_folder, "img1_synth_*.png")))
            for img1_file in img1_files:
                base_name = os.path.basename(img1_file).replace("img1_", "")
                img2_file = os.path.join(city_folder, f"img2_{base_name}")
                label_file = os.path.join(label_dir, city_name, f"cm_{base_name}") if label_dir else None

                if not os.path.exists(img2_file):
                    print(f"Warning: Missing synthetic img2 for {img1_file}. Skipping.")
                    skipped_samples += 1
                    continue
                if label_dir and not os.path.exists(label_file):
                    print(f"Warning: Missing synthetic label for {img1_file}. Skipping.")
                    skipped_samples += 1
                    continue

                if _check_image_readable(img1_file) and _check_image_readable(img2_file) and (label_file is None or _check_image_readable(label_file)):
                    samples.append({"img1": img1_file, "img2": img2_file, "label": label_file, "city": f"{city_name}_synth"})
                else:
                    print(f"Warning: Unreadable synthetic file found for city {city_name}, base {base_name}. Skipping.")
                    skipped_samples += 1
        else:
            # Real structure: images/.../<city>/pair/img1.png, labels/.../<city>/cm/cm.png
            img1_file = os.path.join(city_folder, "pair", "img1.png")
            img2_file = os.path.join(city_folder, "pair", "img2.png")
            label_file = os.path.join(label_dir, city_name, "cm", "cm.png") if label_dir else None

            if not os.path.exists(img1_file) or not os.path.exists(img2_file):
                # print(f"Warning: Missing real img1/img2 for city {city_name}. Skipping.")
                skipped_samples += 1
                continue
            if label_dir and not os.path.exists(label_file):
                # print(f"Warning: Missing real label for city {city_name}. Skipping.")
                skipped_samples += 1
                continue

            if _check_image_readable(img1_file) and _check_image_readable(img2_file) and (label_file is None or _check_image_readable(label_file)):
                samples.append({"img1": img1_file, "img2": img2_file, "label": label_file, "city": city_name})
            else:
                print(f"Warning: Unreadable real file found for city {city_name}. Skipping.")
                skipped_samples += 1

    print(f"Scanned {data_dir}. Found {len(samples)} valid samples. Skipped {skipped_samples}.")
    return samples

def _check_image_readable(file_path):
    """Checks if an image file can be opened and loaded by PIL."""
    if file_path is None:
        return True # Allow None labels
    try:
        with Image.open(file_path) as img:
            img.verify() # Verify closes the file
        with Image.open(file_path) as img:
            img.load() # Load image data
        return True
    except (FileNotFoundError, UnidentifiedImageError, SyntaxError, OSError, ValueError) as e:
        # print(f"Readability Check Error for {file_path}: {e}") # Reduce verbosity
        return False

# --- Function to Create Combined Dataset ---
def create_change_detection_dataset(root_dir, dataset_subdir="Onera Satellite Change Detection Dataset", synthetic_data_dir="synthetic_data", mode="train", target_size=(128, 128), use_synthetic=False):
    """
    Factory function to create the appropriate dataset (real, synthetic, or combined).
    Args:
        use_synthetic (bool): If True and mode is 'train', combine real and synthetic data.
    """
    # Define city splits
    ALL_CITIES = ["abudhabi", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "mumbai", "nantes", "paris", "pisa", "rennes", "saclay_e"]
    VAL_CITIES = ["pisa", "rennes", "saclay_e"]
    TRAIN_CITIES = [city for city in ALL_CITIES if city not in VAL_CITIES]

    # Define base paths for real data
    dataset_base_path = os.path.join(root_dir, dataset_subdir)
    real_image_base = os.path.join(dataset_base_path, "images", "Onera Satellite Change Detection dataset - Images")
    real_label_base = os.path.join(dataset_base_path, "train_labels", "Onera Satellite Change Detection dataset - Train Labels")

    # Define base paths for synthetic data
    synth_base_path = os.path.join(root_dir, synthetic_data_dir)
    synth_image_base = os.path.join(synth_base_path, "images")
    synth_label_base = os.path.join(synth_base_path, "labels")

    # Determine cities and augment status based on mode
    if mode == "train":
        target_cities = TRAIN_CITIES
        has_labels = True
        augment = True # Apply standard augmentation only for training
    elif mode == "val":
        target_cities = VAL_CITIES
        has_labels = True
        augment = False
    elif mode == "test":
        try:
            target_cities = [d for d in os.listdir(real_image_base) if os.path.isdir(os.path.join(real_image_base, d))]
        except FileNotFoundError:
            target_cities = []
        has_labels = False
        augment = False
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Scan for real samples
    print(f"--- Scanning Real Data ({mode}) ---")
    real_samples = scan_dataset(
        real_image_base,
        real_label_base if has_labels else None,
        is_synthetic=False
    )
    # Filter real samples by target cities for train/val
    if mode in ["train", "val"]:
        real_samples = [s for s in real_samples if s["city"] in target_cities]
        print(f"Filtered real samples for {mode} cities: {len(real_samples)} samples.")

    real_dataset = BaseChangeDetectionDataset(real_samples, target_size=target_size, augment=augment)

    # If training and use_synthetic is True, scan and add synthetic data
    if mode == "train" and use_synthetic:
        print(f"--- Scanning Synthetic Data ({mode}) ---")
        if not os.path.isdir(synth_image_base):
            print(f"Warning: Synthetic image directory not found at {synth_image_base}. Cannot use synthetic data.")
            return real_dataset # Return only real data

        synthetic_samples = scan_dataset(
            synth_image_base,
            synth_label_base if has_labels else None,
            is_synthetic=True
        )
        # Filter synthetic samples to only include those derived from training cities
        synthetic_samples = [s for s in synthetic_samples if s["city"].replace("_synth", "") in target_cities]
        print(f"Filtered synthetic samples for {mode} cities: {len(synthetic_samples)} samples.")

        if synthetic_samples:
            synthetic_dataset = BaseChangeDetectionDataset(synthetic_samples, target_size=target_size, augment=augment)
            print(f"Combining {len(real_dataset)} real samples and {len(synthetic_dataset)} synthetic samples for training.")
            return ConcatDataset([real_dataset, synthetic_dataset])
        else:
            print("No valid synthetic samples found. Using only real data for training.")
            return real_dataset
    else:
        # For val/test or if not using synthetic, return only the real dataset
        return real_dataset


