# Enhanced Satellite Imagery Change Detection with GAN Augmentation
This project implements an enhanced change detection model for satellite imagery using a Siamese U-Net architecture. Key features include:

A Pix2Pix-style Generative Adversarial Network (GAN) for generating synthetic post-change images.
A script to generate synthetic data using a trained GAN.
An updated training pipeline that can combine real and synthetic data.
Standard data augmentation (random flips, rotations, color jittering) during training.
Learning rate scheduling (ReduceLROnPlateau) during training.
Hyperparameter tuning using Optuna.
Improved loss functions and class imbalance handling.

Note: This version works with a specific dataset structure where images are RGB PNGs located in [City]/pair/img1.png and [City]/pair/img2.png, and labels are in [City]/cm/cm.png.
Codebase Files

models.py: Contains PyTorch implementations of the Siamese U-Net and the GAN components.
dataset.py: Handles loading and preprocessing of real and synthetic data, including standard augmentation.
train_gan.py: Script for training the Pix2Pix GAN.
generate_synthetic_data.py: Script to generate synthetic data using a trained GAN generator.
train.py: Script for training the Siamese U-Net model, with options for synthetic data, standard augmentation, and hyperparameter tuning.
evaluate.py: Script for evaluating a trained Siamese U-Net model.
requirements.txt: Lists the required Python dependencies.
README.md: This file (updated instructions).

Setup Instructions

Create Project Directory:Create a main directory for your project (e.g., Change_Detection_Package).

Place Codebase:Place the provided Python scripts (models.py, dataset.py, etc.) and requirements.txt directly inside your Change_Detection_Package directory.

Set up Python Environment (Recommended):Use a virtual environment (like venv or conda).
cd "Change_Detection_Package"
python -m venv venv
# Activate: source venv/bin/activate (Linux/macOS) or venv\Scripts\activate (Windows)


Install Dependencies:  
pip install -r requirements.txt

(Check PyTorch website for specific installation commands if needed)

Organize Dataset:Ensure your real dataset files are structured correctly within a subdirectory (e.g., Onera Satellite Change Detection Dataset) inside your main project directory. The expected structure is:
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
├── generate_synthetic_data.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── gan_checkpoints/            # Created by train_gan.py
├── gan_samples/                # Created by train_gan.py
└── siamese_checkpoints/        # Created by train.py



Training the GAN

Purpose: Train the Pix2Pix GAN to generate synthetic post-change images.
Command: python train_gan.py
Output: Checkpoints in gan_checkpoints/ (e.g., generator_epoch_200.pth).
Note: The GAN is trained for 200 epochs to ensure better quality synthetic data.

Generating Synthetic Data

Purpose: Use the trained GAN generator to create synthetic data.
Configuration: Edit generate_synthetic_data.py to set GENERATOR_CHECKPOINT_NAME to your trained generator file (e.g., generator_epoch_200.pth).
Command: python generate_synthetic_data.py
Output: Synthetic image pairs and labels saved in the synthetic_data/ directory, organized by city.

Training the Siamese U-Net

Purpose: Train the main change detection model, optionally using the generated synthetic data, standard augmentations, and hyperparameter tuning.
Configuration: The train.py script accepts command-line arguments:
--use-synthetic: Include synthetic data during training.
--tune: Run hyperparameter tuning using Optuna.
Other arguments like --num-epochs, --batch-size, --learning-rate, --target-size can be used to override defaults.


Run Training (Example with Synthetic Data):python train.py --use-synthetic --num-epochs 50 --batch-size 4


Run Hyperparameter Tuning:python train.py --tune


Features: Includes standard augmentations (flips, rotations, color jittering) and ReduceLROnPlateau learning rate scheduling automatically.
Output: Trained Siamese U-Net models saved in siamese_checkpoints/ (e.g., best_model.pth).

Evaluating the Model

Purpose: Evaluate the trained change detection model on the validation set.
Configuration: Edit evaluate.py to set CHECKPOINT_PATH to your desired trained model (e.g., siamese_checkpoints/best_model.pth). Ensure TARGET_SIZE matches the size used during training.
Command: python evaluate.py
Output: Evaluation metrics (Accuracy, Precision, Recall, F1, IoU) printed, and sample visualizations saved to evaluation_results/.

Improving Model Performance
To enhance the metrics of the change detection model (targeting higher Precision, Recall, F1, and IoU), follow these steps:
Step 1: Enhance Data Augmentation

Why: Increases model robustness by exposing it to varied data.
How: Updated dataset.py with random rotations and color jittering. Run python train.py --use-synthetic --num-epochs 50.

Step 2: Address Class Imbalance

Why: Improves detection of the underrepresented "change" class.
How: Added class weighting to CombinedLoss in train.py. Use a 1:9 weight ratio (adjust based on your data). Train with python train.py --use-synthetic --num-epochs 50.

Step 3: Experiment with Loss Functions

Why: Focal Loss focuses on hard examples, potentially boosting recall.
How: Added Focal Loss in train.py. Comment out CombinedLoss and test with python train.py --use-synthetic --num-epochs 50.

Step 4: Hyperparameter Tuning

Why: Finds optimal training settings.
How: Install Optuna (pip install optuna), then run python train.py --use-synthetic --tune. Use the best lr and batch_size for full training.

Step 5: Improve Synthetic Data Quality

Why: Better synthetic data aids generalization.
How: Increase GAN epochs to 200 in train_gan.py, retrain with python train_gan.py, regenerate data with python generate_synthetic_data.py, then train with python train.py --use-synthetic --num-epochs 50.

Step 6: Post-Processing

Why: Refines predictions to reduce noise.
How: Added morphological operations in evaluate.py. Test with python evaluate.py.

Step 7: Ensemble Methods

Why: Combines multiple models for better accuracy.
How: Train three models (python train.py --use-synthetic --num-epochs 50 thrice, saving as model1.pth, model2.pth, model3.pth), update evaluate.py for ensembling, then run python evaluate.py.

Workflow

Apply Steps 1-3 and train (python train.py --use-synthetic --num-epochs 50).
Evaluate (python evaluate.py) and check metrics.
Tune hyperparameters (Step 4), retrain with best settings, and evaluate.
Improve synthetic data (Step 5), retrain, and evaluate.
Add post-processing (Step 6) and ensembling (Step 7), then evaluate final metrics.

Expect significant improvements in F1 and IoU with these changes.
Notes

Experimentation: Achieving optimal performance often requires experimenting with hyperparameters (learning rates, batch sizes, epochs, loss weights), GAN training duration, and the amount/quality of synthetic data used.
Inference: The evaluate.py script is primarily for validation. Adapting it for easy inference on custom image pairs requires further modification (e.g., adding command-line arguments for input image paths).
Image Registration: Ensure input images are properly co-registered (spatially aligned).
GPU Usage: While a GPU is recommended for faster training, the code can run on a CPU.

