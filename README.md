# Enhanced Satellite Imagery Change Detection with GAN Augmentation

This project implements an enhanced change detection model for satellite imagery, using a Siamese U-Net architecture. It also includes a Pix2Pix-style Generative Adversarial Network (GAN) for augmenting the training data with synthetic image pairs.

## Codebase Files

*   `models.py`: Contains PyTorch implementations of the Siamese U-Net and the GAN components (U-Net Generator, PatchGAN Discriminator).
*   `dataset.py`: Handles loading and preprocessing of the OSCD dataset for both Siamese U-Net and GAN training.
*   `train_gan.py`: Script for training the Pix2Pix GAN to generate synthetic post-change images from pre-change images.
*   `train.py`: Script for training the Siamese U-Net change detection model (can potentially be adapted to use GAN-generated data).
*   `evaluate.py`: Script for evaluating a trained Siamese U-Net model or performing inference on new image pairs.
*   `requirements.txt`: Lists the required Python dependencies.
*   `README.md`: This file.

## Setup Instructions

1.  **Create Project Directory:**
    Create a main directory for your project (e.g., `Change Detection Package`).

2.  **Unzip Codebase:**
    Extract the contents of the provided zip file into this main directory. You should now have the Python scripts (`models.py`, `dataset.py`, etc.) and `requirements.txt` directly inside `Change Detection Package`.

3.  **Set up Python Environment (Recommended):**
    It is highly recommended to use a virtual environment (like `venv` or `conda`).
    ```bash
    # Navigate to your project directory
    cd "Change Detection Package"

    # Create a virtual environment (example using venv)
    python -m venv venv

    # Activate the environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Install the required libraries using pip.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing PyTorch (`torch`, `torchvision`, `torchaudio`) might require specific commands depending on your system and CUDA version. Refer to the official PyTorch website ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).*

5.  **Acquire OSCD Dataset:**
    Download the dataset (e.g., using Kaggle CLI or manually) as described in the previous README version.

6.  **Organize Dataset:**
    Ensure the dataset files are structured correctly within a subdirectory named `Onera Satellite Change Detection Dataset` inside your main project directory (`Change Detection Package`).
    ```
    Change Detection Package/
    ├── Onera Satellite Change Detection Dataset/
    │   ├── images/                 # Contains city subfolders (abudhabi, etc.)
    │   │   └── ...                 # -> imgs_1_rect/*.tif, imgs_2_rect/*.tif
    │   └── train_labels/           # Contains city subfolders for labels
    │       └── ...                 # -> cm/*-cm.tif
    ├── models.py
    ├── dataset.py
    ├── train_gan.py
    ├── train.py
    ├── evaluate.py
    ├── requirements.txt
    ├── README.md
    ├── gan_checkpoints/        # Will be created by train_gan.py
    ├── gan_samples/            # Will be created by train_gan.py
    └── siamese_checkpoints/    # Will be created by train.py
    ```
    *The `dataset.py` and `train_gan.py` scripts expect this structure. Ensure the `ROOT_DIR` in the scripts points to `Change Detection Package` and `DATASET_SUBDIR` points to `Onera Satellite Change Detection Dataset`.* 

## Usage Workflow

Make sure your virtual environment is activated before running the scripts.

**Step 1: Train the GAN (Optional but Recommended for Augmentation)**

*   **Purpose:** Train the Pix2Pix GAN to learn how to generate realistic post-change images (Image 2) given pre-change images (Image 1).
*   **Configuration:** Modify settings in `train_gan.py` if needed (e.g., `NUM_EPOCHS`, `LEARNING_RATE_G`, `LEARNING_RATE_D`, `INPUT_BANDS`). Ensure `ROOT_DIR` and `DATASET_SUBDIR` are correct.
*   **Run Training:**
    ```bash
    python train_gan.py
    ```
*   **Output:** Trained Generator and Discriminator models will be saved in `gan_checkpoints/`. Sample generated images will be saved in `gan_samples/`.

**Step 2: Generate Synthetic Data using Trained GAN (If Step 1 was performed)**

*   **Purpose:** Use the trained GAN generator to create synthetic image pairs for augmenting the training data for the Siamese U-Net.
*   **(Implementation Note:** A separate script for *generating* data using the trained GAN is not provided here but would typically involve loading the trained generator checkpoint (`generator_epoch_XXX.pth`), iterating through the real Image 1 data, generating corresponding fake Image 2s, and saving these pairs (potentially along with derived change masks if needed) to a new directory for augmented data.)*

**Step 3: Train the Siamese U-Net Change Detection Model**

*   **Purpose:** Train the main model to detect changes between Image 1 and Image 2.
*   **Configuration:**
    *   Modify settings in `train.py` (e.g., `NUM_EPOCHS`, `LEARNING_RATE`, `INPUT_BANDS`). Ensure `ROOT_DIR` and `DATASET_SUBDIR` are correct.
    *   **(If using augmented data):** You would need to modify `train.py` and potentially `dataset.py` to load *both* the original OSCD data and the synthetic data generated in Step 2. This might involve creating a combined dataset or modifying the training loop to alternate between real and synthetic batches.
*   **Run Training (Standard - without augmentation):**
    ```bash
    python train.py
    ```
*   **Output:** Trained Siamese U-Net models will be saved in `siamese_checkpoints/`.

**Step 4: Evaluate the Siamese U-Net Model / Perform Inference**

*   **Purpose:** Evaluate the trained change detection model or use it to predict changes on new image pairs.
*   **Configuration:**
    *   Modify `evaluate.py`:
        *   Set `CHECKPOINT_PATH` to the desired trained Siamese U-Net model (e.g., `siamese_checkpoints/best_model.pth`).
        *   Ensure `ROOT_DIR`, `DATASET_SUBDIR`, and `INPUT_BANDS` are correct.
        *   Set `MODE` to `"val"` (or `"test"` if test labels are available) for evaluation on OSCD data, or `"inference"` for custom images.
        *   If `MODE = "inference"`, provide paths to your custom input images (`INFERENCE_IMG1_PATH`, `INFERENCE_IMG2_PATH`) and an `OUTPUT_DIR` for the result.
*   **Run Evaluation/Inference:**
    ```bash
    python evaluate.py
    ```
*   **Output:** Evaluation metrics (Precision, Recall, F1, IoU) printed to the console and sample visualizations saved to `evaluation_results/`. For inference mode, the predicted change map (`.png`) is saved to the specified `OUTPUT_DIR`.

## Notes

*   **GAN Integration:** This package provides the separate training script for the GAN (`train_gan.py`). The integration of the GAN-generated data into the main `train.py` script (Step 3) requires further implementation.
*   **Evaluation Metrics:** The `evaluate.py` script calculates metrics based on the OSCD validation/test set structure. Adapting it for custom inference outputs with ground truth requires modification.
*   **Image Registration:** Ensure input images (especially for inference) are properly co-registered (spatially aligned).

