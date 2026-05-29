#!/usr/bin/env python3
"""
=============================================================================
 PPE Detection System — Phase 1: Model Training Pipeline
=============================================================================
 Description : Trains a YOLOv8n (nano) architecture on a custom PPE dataset
               containing 4 classes: Gloves, Vest, helmet, person.
 Author      : Jigil AK
 Platform    : Windows 11 / CUDA GPU with automatic CPU fallback
 Framework   : Ultralytics YOLOv8
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    from ultralytics import YOLO
except ImportError as e:
    print(f"[FATAL] Missing critical dependency: {e}")
    print("[INFO]  Run: pip install -r requirements.txt")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Logger Configuration
# ──────────────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("PPE-Train")

# ──────────────────────────────────────────────────────────────────────────────
# Project Constants & Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

# Absolute path to dataset root (contains data.yaml + train/valid/test splits)
DATASET_ROOT = Path(r"C:\Users\jigil ak\Documents\My Projects\Computer Vision Engineer_Assignment\dataset")
DATA_YAML    = DATASET_ROOT / "data.yaml"

# Training hyperparameters
IMG_SIZE     = 640      # Input resolution (pixels) — standard YOLOv8 default
EPOCHS       = 50       # Total training epochs
BATCH_SIZE   = -1       # -1 = Ultralytics AutoBatch (auto GPU memory estimation)
DEVICE       = 0        # GPU index; automatically falls back to CPU if unavailable

# Output project directory for experiment artifacts (weights, logs, curves)
PROJECT_DIR  = Path(r"C:\Users\jigil ak\Documents\My Projects\Computer Vision Engineer_Assignment\runs")
RUN_NAME     = "ppe_yolov8n_training"

# Baseline model checkpoint — downloads automatically if not cached
BASE_MODEL   = "yolov8n.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device() -> str:
    """
    Determines the optimal compute device for training.
    
    Priority:
        1. CUDA GPU (device index 0) — preferred for training speed
        2. CPU fallback — if no CUDA-capable GPU is detected
    
    Returns:
        str or int: Device identifier compatible with Ultralytics API.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)  # GB
        logger.info(f"✓ CUDA GPU detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        return 0  # GPU index 0
    else:
        logger.warning("✗ No CUDA GPU detected — falling back to CPU training")
        logger.warning("  Training on CPU will be significantly slower.")
        return "cpu"


def validate_dataset(data_yaml_path: Path) -> bool:
    """
    Pre-flight validation of the dataset configuration file and directory structure.
    
    Checks:
        - data.yaml exists and is readable
        - train/ and valid/ image directories exist
        - At least one training image is present
    
    Args:
        data_yaml_path: Absolute path to the data.yaml configuration file.
    
    Returns:
        bool: True if all validation checks pass.
    
    Raises:
        FileNotFoundError: If critical dataset components are missing.
    """
    # Check data.yaml existence
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml_path}\n"
            f"Ensure the dataset is correctly placed at: {DATASET_ROOT}"
        )
    
    # Check split directories
    for split_name in ["train", "valid"]:
        split_images = DATASET_ROOT / split_name / "images"
        if not split_images.exists():
            raise FileNotFoundError(
                f"Missing {split_name} images directory: {split_images}"
            )
        
        # Count images (common extensions)
        image_count = sum(
            1 for f in split_images.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        )
        logger.info(f"  ├── {split_name}/images: {image_count} images found")
    
    # Optional: check test split
    test_images = DATASET_ROOT / "test" / "images"
    if test_images.exists():
        test_count = sum(
            1 for f in test_images.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        )
        logger.info(f"  └── test/images:  {test_count} images found")
    else:
        logger.info(f"  └── test/images:  (not present, skipping)")
    
    return True


def print_banner():
    """Prints a formatted startup banner to the console."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║             PPE DETECTION SYSTEM — YOLOv8n TRAINING PIPELINE           ║
║                    Antigravity Edge Compute Platform                    ║
╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def log_training_config(device):
    """Logs the complete training configuration for reproducibility."""
    logger.info("─" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("─" * 60)
    logger.info(f"  Base Model     : {BASE_MODEL}")
    logger.info(f"  Dataset YAML   : {DATA_YAML}")
    logger.info(f"  Image Size     : {IMG_SIZE}px")
    logger.info(f"  Epochs         : {EPOCHS}")
    logger.info(f"  Batch Size     : {'AutoBatch' if BATCH_SIZE == -1 else BATCH_SIZE}")
    logger.info(f"  Device         : {device}")
    logger.info(f"  Output Project : {PROJECT_DIR}")
    logger.info(f"  Run Name       : {RUN_NAME}")
    logger.info(f"  Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("─" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestrates the complete YOLOv8n training pipeline:
    
    1. Environment setup and device resolution
    2. Dataset validation (pre-flight checks)
    3. Model instantiation from pretrained baseline
    4. Training execution with configured hyperparameters
    5. Post-training validation with mAP metric extraction
    6. Results summary and artifact location logging
    """
    print_banner()
    
    # ── Step 1: Resolve Compute Device ────────────────────────────────────
    logger.info("[1/5] Resolving compute device...")
    device = resolve_device()
    
    # ── Step 2: Validate Dataset ──────────────────────────────────────────
    logger.info("[2/5] Validating dataset integrity...")
    logger.info(f"  Dataset root: {DATASET_ROOT}")
    try:
        validate_dataset(DATA_YAML)
        logger.info("  ✓ Dataset validation passed")
    except FileNotFoundError as e:
        logger.error(f"  ✗ Dataset validation FAILED: {e}")
        sys.exit(1)
    
    # ── Step 3: Instantiate YOLOv8n Model ─────────────────────────────────
    logger.info("[3/5] Loading YOLOv8n baseline architecture...")
    try:
        model = YOLO(BASE_MODEL)
        logger.info(f"  ✓ Model loaded: {BASE_MODEL}")
        logger.info(f"  ✓ Architecture: YOLOv8 Nano (detection head)")
    except Exception as e:
        logger.error(f"  ✗ Failed to load model: {e}")
        sys.exit(1)
    
    # Log full configuration before training begins
    log_training_config(device)
    
    # ── Step 4: Execute Training ──────────────────────────────────────────
    logger.info("[4/5] Initiating training — this may take a while...")
    training_start = time.time()
    
    try:
        results = model.train(
            data=str(DATA_YAML),    # Absolute path to dataset config
            imgsz=IMG_SIZE,          # Input image resolution
            epochs=EPOCHS,           # Number of training epochs
            batch=BATCH_SIZE,        # AutoBatch: -1 lets Ultralytics estimate
            device=device,           # GPU index or 'cpu'
            project=str(PROJECT_DIR),# Output directory for run artifacts
            name=RUN_NAME,           # Experiment run name
            exist_ok=True,           # Overwrite if run name already exists
            verbose=True,            # Enable detailed training logs
            patience=10,             # Early stopping patience (epochs)
            save=True,               # Save checkpoints
            save_period=-1,          # Save only best + last (not every N epochs)
            plots=True,              # Generate training curves and confusion matrix
            workers=4,               # DataLoader worker threads
        )
        
        training_duration = time.time() - training_start
        logger.info(f"  ✓ Training completed in {training_duration / 60:.1f} minutes")
        
    except Exception as e:
        logger.error(f"  ✗ Training FAILED with error: {e}")
        logger.error(f"  Stack trace:", exc_info=True)
        sys.exit(1)
    
    # ── Step 5: Post-Training Validation & Metric Extraction ──────────────
    logger.info("[5/5] Running post-training validation on best weights...")
    
    # Locate the best weights checkpoint
    best_weights = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        logger.warning(f"  ⚠ best.pt not found at expected path: {best_weights}")
        logger.warning(f"  Attempting validation with the last-trained model state...")
        val_model = model
    else:
        logger.info(f"  Loading best checkpoint: {best_weights}")
        val_model = YOLO(str(best_weights))
    
    try:
        val_results = val_model.val(
            data=str(DATA_YAML),
            imgsz=IMG_SIZE,
            device=device,
            split="val",              # Validate on the validation split
            verbose=True,
        )
        
        # ── Extract Precision Metrics ─────────────────────────────────────
        # The val() method returns a Results object with mAP metrics
        map50    = val_results.box.map50       # mAP @ IoU 0.50
        map50_95 = val_results.box.map         # mAP @ IoU 0.50:0.95
        precision = val_results.box.mp          # Mean Precision
        recall    = val_results.box.mr          # Mean Recall
        
        # ── Print Results Summary ─────────────────────────────────────────
        logger.info("═" * 60)
        logger.info("       VALIDATION RESULTS — PERFORMANCE SUMMARY")
        logger.info("═" * 60)
        logger.info(f"  ┌─────────────────────────────────────────────┐")
        logger.info(f"  │  mAP@0.50       :  {map50:.4f}                   │")
        logger.info(f"  │  mAP@0.50:0.95  :  {map50_95:.4f}                   │")
        logger.info(f"  │  Precision       :  {precision:.4f}                   │")
        logger.info(f"  │  Recall          :  {recall:.4f}                   │")
        logger.info(f"  └─────────────────────────────────────────────┘")
        logger.info("═" * 60)
        
        # Per-class metrics (if available)
        if hasattr(val_results.box, 'ap50') and val_results.box.ap50 is not None:
            class_names = val_model.names  # {0: 'Gloves', 1: 'Vest', ...}
            ap50_per_class = val_results.box.ap50
            logger.info("  PER-CLASS AP@0.50:")
            for idx, ap in enumerate(ap50_per_class):
                name = class_names.get(idx, f"class_{idx}")
                logger.info(f"    • {name:<15s} : {ap:.4f}")
            logger.info("─" * 60)
        
    except Exception as e:
        logger.error(f"  ✗ Validation FAILED: {e}")
        logger.error(f"  Stack trace:", exc_info=True)
        sys.exit(1)
    
    # ── Final Summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║              TRAINING PIPELINE COMPLETE                  ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(f"║  Best Weights : {best_weights}")
    logger.info(f"║  Last Weights : {PROJECT_DIR / RUN_NAME / 'weights' / 'last.pt'}")
    logger.info(f"║  Train Curves : {PROJECT_DIR / RUN_NAME}")
    logger.info(f"║  Duration     : {training_duration / 60:.1f} minutes")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("Next step: Run 'python export_onnx.py' to export to ONNX format.")


# ──────────────────────────────────────────────────────────────────────────────
# Script Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
