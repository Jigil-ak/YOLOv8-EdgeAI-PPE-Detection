#!/usr/bin/env python3
"""
=============================================================================
 PPE Detection System — Phase 2: ONNX Export & FP16 Quantization
=============================================================================
 Description : Converts the trained YOLOv8n PyTorch weights (best.pt) into
               an optimized ONNX graph with FP16 half-precision quantization
               for edge deployment on the Antigravity platform.
 Author      : Jigil AK
 Platform    : Windows 11 / ONNX Runtime compatible
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────────────────────────
try:
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
logger = logging.getLogger("PPE-Export")

# ──────────────────────────────────────────────────────────────────────────────
# Export Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Path to the trained FP32 PyTorch weights (output of train.py)
WEIGHTS_DIR = Path(r"C:\Users\jigil ak\Documents\My Projects\Computer Vision Engineer_Assignment\runs\ppe_yolov8n_training\weights")
FP32_WEIGHTS = WEIGHTS_DIR / "best.pt"

# Export output directory
EXPORT_DIR = Path(r"C:\Users\jigil ak\Documents\My Projects\Computer Vision Engineer_Assignment\models")

# ONNX export parameters
ONNX_OPSET    = 12       # ONNX opset version — 12 is broadly compatible
IMG_SIZE      = 640      # Must match training input resolution
HALF_PRECISION = True    # Enable FP16 quantization during export


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_file_size_mb(filepath: Path) -> float:
    """
    Returns the file size in megabytes with 2 decimal precision.
    
    Args:
        filepath: Absolute path to the target file.
    
    Returns:
        float: File size in MB.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    size_bytes = filepath.stat().st_size
    return size_bytes / (1024 * 1024)


def print_banner():
    """Prints a formatted startup banner to the console."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║          PPE DETECTION SYSTEM — ONNX EXPORT & QUANTIZATION             ║
║                    Antigravity Edge Compute Platform                    ║
╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def memory_audit(fp32_path: Path, fp16_path: Path):
    """
    Performs and logs a detailed memory/storage audit comparing the original
    FP32 PyTorch weights against the quantized FP16 ONNX model.
    
    Metrics reported:
        - Original FP32 model size (MB)
        - Quantized FP16 model size (MB)
        - Absolute reduction (MB)
        - Compression ratio (%)
    
    Args:
        fp32_path: Path to the original FP32 .pt weights file.
        fp16_path: Path to the exported FP16 .onnx model file.
    """
    fp32_size = get_file_size_mb(fp32_path)
    fp16_size = get_file_size_mb(fp16_path)
    reduction = fp32_size - fp16_size
    ratio     = (reduction / fp32_size) * 100 if fp32_size > 0 else 0.0
    
    logger.info("═" * 60)
    logger.info("         MEMORY AUDIT — STORAGE REDUCTION ANALYSIS")
    logger.info("═" * 60)
    logger.info(f"  ┌─────────────────────────────────────────────────┐")
    logger.info(f"  │  Original FP32 (.pt)   :  {fp32_size:>8.2f} MB          │")
    logger.info(f"  │  Quantized FP16 (.onnx):  {fp16_size:>8.2f} MB          │")
    logger.info(f"  │  ─────────────────────────────────────────────  │")
    logger.info(f"  │  Absolute Reduction    :  {reduction:>8.2f} MB          │")
    logger.info(f"  │  Compression Ratio     :  {ratio:>8.1f} %           │")
    logger.info(f"  └─────────────────────────────────────────────────┘")
    logger.info("═" * 60)
    
    # Additional context for edge deployment
    if ratio > 40:
        logger.info("  ✓ Excellent compression — well-suited for edge deployment")
    elif ratio > 20:
        logger.info("  ✓ Good compression — acceptable for edge deployment")
    else:
        logger.info("  ⚠ Modest compression — consider INT8 quantization for further gains")


# ──────────────────────────────────────────────────────────────────────────────
# Main Export Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestrates the ONNX export and FP16 quantization pipeline:
    
    1. Validates the trained FP32 weights exist
    2. Loads the YOLOv8n model from the best checkpoint
    3. Exports to ONNX format with FP16 half-precision quantization
    4. Verifies the exported ONNX file integrity
    5. Performs a memory audit comparing FP32 vs FP16 sizes
    6. Copies the final model to the deployment directory
    """
    print_banner()
    
    # ── Step 1: Validate Source Weights ────────────────────────────────────
    logger.info("[1/5] Validating trained FP32 weights...")
    
    if not FP32_WEIGHTS.exists():
        logger.error(f"  ✗ Trained weights not found: {FP32_WEIGHTS}")
        logger.error(f"  ✗ Run 'python train.py' first to generate best.pt")
        sys.exit(1)
    
    fp32_size = get_file_size_mb(FP32_WEIGHTS)
    logger.info(f"  ✓ Found: {FP32_WEIGHTS}")
    logger.info(f"  ✓ Size:  {fp32_size:.2f} MB (FP32 precision)")
    
    # ── Step 2: Load Model ────────────────────────────────────────────────
    logger.info("[2/5] Loading YOLOv8n model from trained checkpoint...")
    
    try:
        model = YOLO(str(FP32_WEIGHTS))
        logger.info(f"  ✓ Model loaded successfully")
        logger.info(f"  ✓ Classes: {model.names}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load model: {e}")
        sys.exit(1)
    
    # ── Step 3: Export to ONNX with FP16 Quantization ─────────────────────
    logger.info("[3/5] Exporting model to ONNX format with FP16 quantization...")
    logger.info(f"  Export parameters:")
    logger.info(f"    • Format      : ONNX")
    logger.info(f"    • Opset       : {ONNX_OPSET}")
    logger.info(f"    • Image Size  : {IMG_SIZE}px")
    logger.info(f"    • Half (FP16) : {HALF_PRECISION}")
    logger.info(f"    • Simplify    : True")
    
    try:
        # The Ultralytics export() method handles the full conversion pipeline:
        #   PyTorch → TorchScript trace → ONNX graph → (optional) ONNX Simplifier
        # Setting half=True converts all FP32 weights to FP16 during serialization
        export_path = model.export(
            format="onnx",          # Target serialization format
            imgsz=IMG_SIZE,          # Input tensor dimensions
            half=HALF_PRECISION,     # FP16 weight quantization
            opset=ONNX_OPSET,       # ONNX operator set version
            simplify=True,           # Run onnx-simplifier to optimize the graph
            dynamic=False,           # Static batch size for edge deployment
        )
        
        logger.info(f"  ✓ ONNX export completed: {export_path}")
        
    except Exception as e:
        logger.error(f"  ✗ ONNX export FAILED: {e}")
        logger.error(f"  Stack trace:", exc_info=True)
        sys.exit(1)
    
    # ── Step 4: Verify Exported Model ─────────────────────────────────────
    logger.info("[4/5] Verifying exported ONNX model integrity...")
    
    onnx_path = Path(export_path)
    if not onnx_path.exists():
        logger.error(f"  ✗ Expected ONNX file not found: {onnx_path}")
        sys.exit(1)
    
    # Optional: Run ONNX checker if onnx package is available
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"  ✓ ONNX model graph validation PASSED")
        
        # Log model metadata
        graph = onnx_model.graph
        logger.info(f"  ✓ Input tensor  : {graph.input[0].name} "
                     f"→ shape {[d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]}")
        logger.info(f"  ✓ Output tensor : {graph.output[0].name} "
                     f"→ shape {[d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]}")
        logger.info(f"  ✓ Graph nodes   : {len(graph.node)}")
        
    except ImportError:
        logger.warning("  ⚠ 'onnx' package not installed — skipping graph validation")
        logger.warning("  Install with: pip install onnx")
    except Exception as e:
        logger.warning(f"  ⚠ ONNX validation warning: {e}")
    
    # ── Step 5: Memory Audit ──────────────────────────────────────────────
    logger.info("[5/5] Performing memory audit...")
    memory_audit(FP32_WEIGHTS, onnx_path)
    
    # ── Copy to Deployment Directory ──────────────────────────────────────
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    deploy_path = EXPORT_DIR / "ppe_yolov8n_fp16.onnx"
    
    try:
        import shutil
        shutil.copy2(str(onnx_path), str(deploy_path))
        logger.info(f"  ✓ Deployment copy saved: {deploy_path}")
    except Exception as e:
        logger.warning(f"  ⚠ Could not copy to deployment dir: {e}")
        logger.warning(f"  Original ONNX file remains at: {onnx_path}")
    
    # ── Final Summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║              ONNX EXPORT PIPELINE COMPLETE              ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(f"║  Source (FP32)  : {FP32_WEIGHTS}")
    logger.info(f"║  Output (FP16)  : {deploy_path}")
    logger.info(f"║  Opset Version  : {ONNX_OPSET}")
    logger.info(f"║  Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("Next step: Run 'python live_inference.py' for real-time detection.")


# ──────────────────────────────────────────────────────────────────────────────
# Script Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
