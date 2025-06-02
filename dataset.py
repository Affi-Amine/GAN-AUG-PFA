# Dataset loading logic for Custom PNG Change Detection Dataset
import os
import glob
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

# --- Utility Function for Collation ---
def collate_fn_skip_none(batch):
    """Collate function that filters out None samples."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
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
            label_np = np.array(label.convert("L"))
            label_np = (label_np > 128).astype(np.uint8)
            sample["label"] = torch.from_numpy(label_np).long()
        else:
            sample["label"] = None
        return sample

class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):
        if random.random() < self.p:
            sample["image1"] = TF.hflip(sample["image1"])
            sample["image2"] = TF.hflip(sample["image2"])
            if sample["label"] is not None:
                # sample["label"] is [H,W] LongTensor
                label_1hw = sample["label"].unsqueeze(0)  # Shape: [1, H, W], dtype: torch.long
                transformed_label_1hw = TF.hflip(label_1hw) # Shape: [1, H, W], dtype: torch.long
                sample["label"] = transformed_label_1hw.squeeze(0)  # Shape: [H, W], dtype: torch.long
        return sample

class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):
        if random.random() < self.p:
            sample["image1"] = TF.vflip(sample["image1"])
            sample["image2"] = TF.vflip(sample["image2"])
            if sample["label"] is not None:
                # sample["label"] is [H,W] LongTensor
                label_1hw = sample["label"].unsqueeze(0)  # Shape: [1, H, W], dtype: torch.long
                transformed_label_1hw = TF.vflip(label_1hw) # Shape: [1, H, W], dtype: torch.long
                sample["label"] = transformed_label_1hw.squeeze(0)  # Shape: [H, W], dtype: torch.long
        return sample

class JointRandomRotation:
    def __init__(self, degrees=30):
        self.degrees = degrees
    def __call__(self, sample):
        angle = random.uniform(-self.degrees, self.degrees)
        sample["image1"] = TF.rotate(sample["image1"], angle)
        sample["image2"] = TF.rotate(sample["image2"], angle)
        if sample["label"] is not None:
            # sample["label"] is [H,W] LongTensor from JointToTensor
            label_1hw = sample["label"].unsqueeze(0)  # Shape: [1, H, W], dtype: torch.long
            # TF.rotate with NEAREST interpolation preserves torch.long dtype
            transformed_label_1hw = TF.rotate(label_1hw, angle, interpolation=TF.InterpolationMode.NEAREST) # Shape: [1, H, W], dtype: torch.long
            sample["label"] = transformed_label_1hw.squeeze(0)  # Shape: [H, W], dtype: torch.long
        return sample

class JointRandomAffine:
    def __init__(self, degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, sample):
        affine_params = transforms.RandomAffine.get_params(
            degrees=(-self.degrees, self.degrees),
            translate=self.translate,
            scale_ranges=self.scale,
            shears=(-self.shear, self.shear),
            img_size=sample["image1"].size # PIL image size
        )
        sample["image1"] = TF.affine(sample["image1"], *affine_params, interpolation=TF.InterpolationMode.BILINEAR)
        sample["image2"] = TF.affine(sample["image2"], *affine_params, interpolation=TF.InterpolationMode.BILINEAR)
        if sample["label"] is not None:
            # Label is PIL Image here before ToTensor
            sample["label"] = TF.affine(sample["label"], *affine_params, interpolation=TF.InterpolationMode.NEAREST)
        return sample

class JointGaussianBlur:
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        # Apply GaussianBlur only to images, not labels
        # Randomly select sigma for each image independently but same kernel size
        sigma1 = random.uniform(self.sigma[0], self.sigma[1])
        sample["image1"] = TF.gaussian_blur(sample["image1"], kernel_size=self.kernel_size, sigma=sigma1)
        sigma2 = random.uniform(self.sigma[0], self.sigma[1])
        sample["image2"] = TF.gaussian_blur(sample["image2"], kernel_size=self.kernel_size, sigma=sigma2)
        return sample

class JointColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0):
        # Store parameters, ColorJitter will be created per call for PIL images
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue # Add hue if desired, though often not used for satellite

    def __call__(self, sample):
        # Create a new ColorJitter transform instance for each call to ensure different jittering
        # for img1 and img2 if that's the desired behavior, or use the same instance if they should be jittered identically.
        # For independent jittering (more common for data augmentation):
        transform1 = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        transform2 = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        
        sample["image1"] = transform1(sample["image1"]) # Applied to PIL image
        sample["image2"] = transform2(sample["image2"]) # Applied to PIL image
        return sample

class JointResize:
    def __init__(self, size):
        self.size = size
    def __call__(self, sample):
        img1 = sample["image1"]
        img2 = sample["image2"]
        label = sample["label"]
        img1_resized = TF.resize(img1, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        img2_resized = TF.resize(img2, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        if label is not None:
            label_resized = TF.resize(label.unsqueeze(0), self.size, interpolation=TF.InterpolationMode.NEAREST)
            label_resized = label_resized.squeeze(0)
        else:
            label_resized = None
        sample["image1"] = img1_resized
        sample["image2"] = img2_resized
        sample["label"] = label_resized
        return sample

class JointNormalize:
    def __call__(self, sample):
        sample["image1"] = (sample["image1"] * 2.0) - 1.0
        sample["image2"] = (sample["image2"] * 2.0) - 1.0
        return sample

# --- Base Dataset Class ---
class BaseChangeDetectionDataset(Dataset):
    def __init__(self, samples_list, target_size=(128, 128), augment=False):
        self.samples = samples_list
        self.target_size = target_size
        self.augment = augment
        # Note: Affine and GaussianBlur expect PIL images if placed before ToTensor
        # ToTensor converts PIL to Tensor. Resize also happens after ToTensor in current setup.
        # For Affine to work on PIL before ToTensor, it needs to be placed earlier.
        # Let's adjust the order: Augmentations on PIL, then ToTensor, then Resize, then Normalize.

        pil_augment_list = []
        if self.augment:
            pil_augment_list.extend([
                JointRandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
                JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # Now applied to PIL
                JointGaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])

        tensor_augment_list = []
        if self.augment:
            tensor_augment_list.extend([
                JointRandomHorizontalFlip(p=0.5),
                JointRandomVerticalFlip(p=0.5),
                JointRandomRotation(degrees=30),
            ])

        transform_list = pil_augment_list + \
                         [JointToTensor()] + \
                         tensor_augment_list + \
                         [JointResize(self.target_size), JointNormalize()]

        self.transform = transforms.Compose(transform_list)

    def _load_pil_image(self, file_path):
        try:
            img = Image.open(file_path).convert("RGB")
            img.load()
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
            if sample_info["label"] is None:
                sample["label"] = None
            return sample
        except Exception as e:
            print(f"Failed to load/transform sample for city {sample_info.get('city', 'N/A')} at index {idx}: {e}. Returning None.")
            return None

# --- Helper Function to Scan for Samples ---
def scan_dataset(data_dir, label_dir=None, is_synthetic=False):
    samples = []
    skipped_samples = 0
    image_folders = glob.glob(os.path.join(data_dir, "*"))
    for city_folder in image_folders:
        if not os.path.isdir(city_folder):
            continue
        city_name = os.path.basename(city_folder)
        if is_synthetic:
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
            img1_file = os.path.join(city_folder, "pair", "img1.png")
            img2_file = os.path.join(city_folder, "pair", "img2.png")
            label_file = os.path.join(label_dir, city_name, "cm", "cm.png") if label_dir else None
            if not os.path.exists(img1_file) or not os.path.exists(img2_file):
                skipped_samples += 1
                continue
            if label_dir and not os.path.exists(label_file):
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
    if file_path is None:
        return True
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
        return True
    except (FileNotFoundError, UnidentifiedImageError, SyntaxError, OSError, ValueError) as e:
        return False

# --- Function to Create Combined Dataset ---
def create_change_detection_dataset(root_dir, dataset_subdir="Onera Satellite Change Detection Dataset", synthetic_data_dir="synthetic_data", mode="train", target_size=(128, 128), use_synthetic=False, augment=False):
    ALL_CITIES = ["abudhabi", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "mumbai", "nantes", "paris", "pisa", "rennes", "saclay_e"]
    VAL_CITIES = ["pisa", "rennes", "saclay_e"]
    TRAIN_CITIES = [city for city in ALL_CITIES if city not in VAL_CITIES]
    dataset_base_path = os.path.join(root_dir, dataset_subdir)
    real_image_base = os.path.join(dataset_base_path, "images", "Onera Satellite Change Detection dataset - Images")
    real_label_base = os.path.join(dataset_base_path, "train_labels", "Onera Satellite Change Detection dataset - Train Labels")
    synth_base_path = os.path.join(root_dir, synthetic_data_dir)
    synth_image_base = os.path.join(synth_base_path, "images")
    synth_label_base = os.path.join(synth_base_path, "labels")
    if mode == "train":
        target_cities = TRAIN_CITIES
        has_labels = True
    elif mode == "val":
        target_cities = VAL_CITIES
        has_labels = True
    elif mode == "test":
        try:
            target_cities = [d for d in os.listdir(real_image_base) if os.path.isdir(os.path.join(real_image_base, d))]
        except FileNotFoundError:
            target_cities = []
        has_labels = False
    else:
        raise ValueError(f"Invalid mode: {mode}")
    print(f"--- Scanning Real Data ({mode}) ---")
    real_samples = scan_dataset(
        real_image_base,
        real_label_base if has_labels else None,
        is_synthetic=False
    )
    if mode in ["train", "val"]:
        real_samples = [s for s in real_samples if s["city"] in target_cities]
        print(f"Filtered real samples for {mode} cities: {len(real_samples)} samples.")
    real_dataset = BaseChangeDetectionDataset(real_samples, target_size=target_size, augment=augment)
    if mode == "train" and use_synthetic:
        print(f"--- Scanning Synthetic Data ({mode}) ---")
        if not os.path.isdir(synth_image_base):
            print(f"Warning: Synthetic image directory not found at {synth_image_base}. Cannot use synthetic data.")
            return real_dataset
        synthetic_samples = scan_dataset(
            synth_image_base,
            synth_label_base if has_labels else None,
            is_synthetic=True
        )
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
        return real_dataset