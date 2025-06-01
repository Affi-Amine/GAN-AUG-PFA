# Enhanced Satellite Imagery Change Detection with GAN Augmentation

This project implements an enhanced change detection model for satellite imagery, using a Siamese U-Net architecture. It includes:
*   A Pix2Pix-style Generative Adversarial Network (GAN) for generating synthetic post-change images.
*   A script to generate synthetic data using a trained GAN.
*   An updated training pipeline that can combine real and synthetic data.
*   Standard data augmentation (random flips) during training.
*   Learning rate scheduling (ReduceLROnPlateau) during training.

**Note:** This version works with a specific dataset structure where images are RGB PNGs located in `[City]/pair/img1.png` and `[City]/pair/img2.png`, and labels are in `[City]/cm/cm.png`.

## Codebase Files

*   `models.py`: Contains PyTorch implementations of the Siamese U-Net and the GAN components.
*   `dataset.py`: Handles loading and preprocessing of real and synthetic data, including standard augmentation.
*   `train_gan.py`: Script for training the Pix2Pix GAN.
*   `generate_synthetic_data.py`: **New script** to generate synthetic data using a trained GAN generator.
*   `train.py`: **Updated script** for training the Siamese U-Net model, now with options for synthetic data, standard augmentation, and LR scheduling.
*   `evaluate.py`: Script for evaluating a trained Siamese U-Net model.
*   `requirements.txt`: Lists the required Python dependencies.
*   `README.md`: This file (updated instructions).

## Setup Instructions

1.  **Create Project Directory:**
    Create a main directory for your project (e.g., `Change_Detection_Package`).

2.  **Place Codebase:**
    Place the provided Python scripts (`models.py`, `dataset.py`, etc.) and `requirements.txt` directly inside your `Change_Detection_Package` directory.

3.  **Set up Python Environment (Recommended):**
    Use a virtual environment (like `venv` or `conda`).
    ```bash
    cd "Change_Detection_Package"
    python -m venv venv
    # Activate: source venv/bin/activate (Linux/macOS) or venv\Scripts\activate (Windows)
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Check PyTorch website for specific installation commands if needed)*

5.  **Organize Dataset:**
    Ensure your real dataset files are structured correctly within a subdirectory (e.g., `Onera Satellite Change Detection Dataset`) inside your main project directory. The expected structure is:
    ```
    Change_Detection_Package/
    ├── Onera Satellite Change Detection Dataset/  # Real dataset subdirectory
    │   ├── images/
    │   │   └── Onera Satellite Change Detection dataset - Images/
    │   │       ├── [city1]/
    │   │       │   └── pair/
    │   │       │       ├── img1.png
    │   │       │       └── img2.png
    │   │       └── ... (other cities)
    │   └── train_labels/
    │       └── Onera Satellite Change Detection dataset - Train Labels/
    │           ├── [city1]/
    │           │   └── cm/
    │           │       └── cm.png
    │           └── ... (other cities)
    ├── synthetic_data/             # Will be created by generate_synthetic_data.py
    │   ├── images/
    │   │   └── [city1]_synth/
    │   │       ├── img1_synth_*.png
    │   │       └── img2_synth_*.png
    │   └── labels/
    │       └── [city1]_synth/
    │           └── cm_synth_*.png
    ├── models.py
    ├── dataset.py
    ├── train_gan.py
    ├── generate_synthetic_data.py  # New
    ├── train.py                    # Updated
    ├── evaluate.py
    ├── requirements.txt
    ├── README.md                   # Updated
    ├── gan_checkpoints/            # Created by train_gan.py
    ├── gan_samples/                # Created by train_gan.py
    └── siamese_checkpoints/        # Created by train.py
    ```

## Improved Training Workflow

Make sure your virtual environment is activated.

**Step 1: Train the GAN (If not already done)**

*   **Purpose:** Train the Pix2Pix GAN to generate synthetic post-change images.
*   **Command:** `python train_gan.py`
*   **Output:** Checkpoints in `gan_checkpoints/` (e.g., `generator_epoch_100.pth`).

**Step 2: Generate Synthetic Data (New Step)**

*   **Purpose:** Use the trained GAN generator to create synthetic data.
*   **Configuration:** Edit `generate_synthetic_data.py` to set `GENERATOR_CHECKPOINT_NAME` to your trained generator file (e.g., `generator_epoch_100.pth`).
*   **Command:** `python generate_synthetic_data.py`
*   **Output:** Synthetic image pairs and labels saved in the `synthetic_data/` directory, organized by city.

**Step 3: Train the Siamese U-Net (Enhanced)**

*   **Purpose:** Train the main change detection model, optionally using the generated synthetic data and standard augmentations.
*   **Configuration:** The `train.py` script now accepts command-line arguments:
    *   `--use-synthetic`: Add this flag to include the data from `synthetic_data/` during training.
    *   Other arguments like `--num-epochs`, `--batch-size`, `--learning-rate`, `--target-size` can be used to override defaults.
*   **Run Training (Example with Synthetic Data):**
    ```bash
    # Train using both real and synthetic data
    python train.py --use-synthetic --num-epochs 50 --batch-size 4
    ```
*   **Run Training (Example without Synthetic Data):**
    ```bash
    # Train using only real data (standard augmentation still applied)
    python train.py --num-epochs 50 --batch-size 4
    ```
*   **Features:** Includes standard augmentations (flips) and ReduceLROnPlateau learning rate scheduling automatically.
*   **Output:** Trained Siamese U-Net models saved in `siamese_checkpoints/` (e.g., `best_model.pth`).

**Step 4: Evaluate the Siamese U-Net Model**

*   **Purpose:** Evaluate the newly trained change detection model on the validation set.
*   **Configuration:** Edit `evaluate.py` to set `CHECKPOINT_PATH` to your desired trained model (e.g., `siamese_checkpoints/best_model.pth`). Ensure `TARGET_SIZE` matches the size used during training.
*   **Command:** `python evaluate.py`
*   **Output:** Evaluation metrics (Accuracy, Precision, Recall, F1, IoU) printed, and sample visualizations saved to `evaluation_results/`.

## Notes

*   **Experimentation:** Achieving optimal performance often requires experimenting with hyperparameters (learning rates, batch sizes, epochs, loss weights), GAN training duration, and the amount/quality of synthetic data used.
*   **Inference:** The `evaluate.py` script is primarily for validation. Adapting it for easy inference on custom image pairs requires further modification (e.g., adding command-line arguments for input image paths).
*   **Image Registration:** Ensure input images are properly co-registered (spatially aligned).

