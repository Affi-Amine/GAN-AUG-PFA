# dataset.py  (all of ChangeDetectionDataset, in‐memory version)

import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF

class NormalizeTransform:
    """ Maps each image‐tensor from [0,1] → [-1,1]. """
    def __call__(self, sample):
        sample["image1"] = (sample["image1"] - 0.5) / 0.5
        sample["image2"] = (sample["image2"] - 0.5) / 0.5
        return sample

class ResizeTransform:
    """
    Resizes “image1”, “image2”, and “label” (if present) in a sample dict to a fixed (H, W).
    - image1, image2 are (3×H×W) RGB tensors
    - label is (H×W) integer mask {0,1}
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img1 = sample["image1"]
        img2 = sample["image2"]
        label = sample["label"]

        # TF.resize can handle Tensor of shape (C, H, W) when using pytorch >= 1.13
        img1_resized = TF.resize(img1, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        img2_resized = TF.resize(img2, self.size, interpolation=TF.InterpolationMode.BILINEAR)

        if label is not None:
            # label is (H, W). Add channel=1 → (1, H, W), resize with NEAREST, then squeeze back → (H, W).
            label_resized = TF.resize(label.unsqueeze(0),
                                      self.size,
                                      interpolation=TF.InterpolationMode.NEAREST)
            label_resized = label_resized.squeeze(0)
        else:
            label_resized = None

        sample["image1"] = img1_resized
        sample["image2"] = img2_resized
        sample["label"] = label_resized
        return sample


class ChangeDetectionDataset(Dataset):
    """
    Load the Onera Satellite Change Detection dataset entirely into memory as Tensors.

    - In __init__, we:
        • Collect all city‐folders for train/val/test.
        • For each city, verify that img1.png, img2.png (and cm.png if applicable) can be fully decoded.
        • Open, convert, load, tensor‐ify, resize, normalize, and store the resulting Tensors.
        • Skip any city whose images or label cannot be fully decoded.

    - In __getitem__, we simply return the preloaded tensors (no PIL calls at runtime).
    """
    VAL_CITIES   = ["pisa", "rennes", "saclay_e"]
    TRAIN_CITIES = [
        'abudhabi', 'aguasclaras', 'beihai', 'beirut',
        'bercy', 'bordeaux', 'cupertino', 'hongkong',
        'mumbai', 'nantes', 'paris'
    ]

    def __init__(self,
                 root_dir,
                 dataset_subdir="Onera Satellite Change Detection Dataset",
                 mode="train",
                 target_size=(128, 128)):
        """
        root_dir: path to the folder containing “Onera Satellite Change Detection Dataset”
        mode: "train", "val", or "test"
        target_size: (H, W) that we will resize each image & label to
        """
        super().__init__()
        self.root_dir = root_dir
        self.dataset_subdir = dataset_subdir
        self.mode = mode
        self.target_size = target_size

        # ========== Compose transforms ==========
        # 1) ToTensor() converts PIL‐Image to FloatTensor [0..1]
        # 2) ResizeTransform → scale both images & label to target_size
        # 3) NormalizeTransform → map [0..1] → [-1..1] for image1/image2
        self.to_tensor = transforms.ToTensor()
        self.resize_and_norm = transforms.Compose([
            ResizeTransform(target_size),
            NormalizeTransform()
        ])

        # ========== Build “base” paths for images & labels ==========
        dataset_base = os.path.join(root_dir, dataset_subdir)

        image_base = os.path.join(
            dataset_base,
            "images",
            "Onera Satellite Change Detection dataset - Images"
        )
        label_base = os.path.join(
            dataset_base,
            "train_labels",
            "Onera Satellite Change Detection dataset - Train Labels"
        )

        if not os.path.isdir(image_base):
            print(f"Warning: Image base directory not found at:\n    {image_base}")
        if mode != "test" and not os.path.isdir(label_base):
            print(f"Warning: Label base directory not found at:\n    {label_base}")

        # ========== Decide which cities to iterate over ==========
        if mode == "train":
            cities = self.TRAIN_CITIES
            self.has_labels = True
        elif mode == "val":
            cities = self.VAL_CITIES
            self.has_labels = True
        elif mode == "test":
            # Simply list whatever subfolders exist in image_base
            try:
                cities = [
                    d for d in os.listdir(image_base)
                    if os.path.isdir(os.path.join(image_base, d))
                ]
                print(f"Using {len(cities)} cities for TEST: {cities}")
            except FileNotFoundError:
                print(f"Error: Cannot list cities in {image_base}.")
                cities = []
            self.has_labels = False
        else:
            raise ValueError(f"Invalid mode='{mode}'. Must be 'train', 'val', or 'test'.")

        # ========== Preload everything into memory ==========
        self.samples = []
        print(f"Processing cities for mode '{mode}': {cities}")

        for city in cities:
            img1_path = os.path.join(image_base, city, "pair", "img1.png")
            img2_path = os.path.join(image_base, city, "pair", "img2.png")
            label_path = None
            if self.has_labels:
                label_path = os.path.join(label_base, city, "cm", "cm.png")

            # If any expected file is missing → skip city
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"  • Warning: Missing img1.png or img2.png for city='{city}'. Skipping.")
                continue
            if self.has_labels and not os.path.exists(label_path):
                print(f"  • Warning: Missing cm.png for city='{city}'. Skipping.")
                continue

            # --- FULL DECODE & PRELOAD STEP ---
            try:
                # 1) Load img1
                with Image.open(img1_path) as imA:
                    imA_rgb = imA.convert("RGB")
                    imA_rgb.load()  # force PIL to read every pixel
                img1_tensor = self.to_tensor(imA_rgb)

                # 2) Load img2
                with Image.open(img2_path) as imB:
                    imB_rgb = imB.convert("RGB")
                    imB_rgb.load()
                img2_tensor = self.to_tensor(imB_rgb)

                # 3) If labels exist, load and binarize
                if self.has_labels:
                    with Image.open(label_path) as lab:
                        lab_gray = lab.convert("L")
                        lab_gray.load()
                    lab_arr = np.array(lab_gray)
                    lab_mask = (lab_arr > 128).astype(np.uint8)
                    label_tensor = torch.from_numpy(lab_mask).long()  # shape (H, W)
                else:
                    label_tensor = None

                # 4) Apply resizing & normalization now (in‐memory!)
                sample_dict = {
                    "image1": img1_tensor,   # shape (3, H, W)
                    "image2": img2_tensor,   # shape (3, H, W)
                    "label":  label_tensor,  # shape (H, W) or None
                    "city":   city
                }
                sample_dict = self.resize_and_norm(sample_dict)
            except (UnidentifiedImageError, OSError) as e:
                print(f"  • Warning: Failed to fully decode city='{city}'. Skipping. ▶ {e}")
                continue

            # If we reach here, everything is good: store the pre‐loaded, pre‐processed tensors.
            self.samples.append(sample_dict)

        print(f"Found {len(self.samples)} valid samples for mode='{mode}'.\n")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Simply return the preloaded + preprocessed tensors. No PIL calls here!
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.samples[idx]

        # We can (optionally) print to track which city is loaded at batch time:
        print(f"  [DATASET] Returning sample idx={idx} (city='{sample['city']}')")

        # Return a dict with:
        #    "image1" → Tensor (3, H, W)
        #    "image2" → Tensor (3, H, W)
        #    "label"  → Tensor (H, W)  or None
        #    "city"   → string
        return {
            "image1": sample["image1"],
            "image2": sample["image2"],
            "label":  sample["label"],
            "city":   sample["city"]
        }
