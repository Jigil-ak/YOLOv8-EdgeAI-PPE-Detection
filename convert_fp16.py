#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 PPE Detection System -- FP16 Post-Export Quantization (CPU-Compatible)
=============================================================================
 Converts the FP32 ONNX model to true FP16 using onnxconverter-common.
 This works on CPU (no CUDA required) unlike Ultralytics half=True export.
=============================================================================
"""
import os, sys, shutil
from pathlib import Path

try:
    import onnx
    from onnxconverter_common import convert_float_to_float16
except ImportError:
    print("[ERROR] Run: pip install onnxconverter-common onnx")
    sys.exit(1)

SRC  = Path(r"runs\ppe_yolov8n_training\weights\best.onnx")   # FP32 ONNX from export
DST  = Path(r"models\ppe_yolov8n_fp16.onnx")                  # True FP16 output

if not SRC.exists():
    print(f"[ERROR] FP32 ONNX not found: {SRC}")
    print("  Run export_onnx.py first.")
    sys.exit(1)

print(f"Loading FP32 ONNX: {SRC}")
model_fp32 = onnx.load(str(SRC))

fp32_size = SRC.stat().st_size / (1024*1024)
print(f"  FP32 size: {fp32_size:.2f} MB")

print("Applying FP16 conversion (onnxconverter-common)...")
# keep_io_types=True keeps input/output as float32 for compatibility
model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)

DST.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model_fp16, str(DST))

fp16_size = DST.stat().st_size / (1024*1024)
reduction = fp32_size - fp16_size
ratio     = (reduction / fp32_size) * 100

print()
print("=" * 52)
print("   MEMORY AUDIT -- TRUE FP16 QUANTIZATION RESULT")
print("=" * 52)
print(f"  FP32 ONNX source   : {fp32_size:>7.2f} MB")
print(f"  FP16 ONNX output   : {fp16_size:>7.2f} MB")
print(f"  Reduction          : {reduction:>7.2f} MB")
print(f"  Compression ratio  : {ratio:>7.1f} %")
print("=" * 52)
print(f"\n[OK] Saved: {DST}")
