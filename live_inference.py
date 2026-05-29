#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 PPE Detection System -- Phase 3: Optimized Live Edge Inference
=============================================================================
 Description : Real-time PPE detection using a quantized ONNX model with
               pure ONNX Runtime + OpenCV. Features custom preprocessing,
               vectorized NMS, and a professional on-screen HUD.
 Author      : Jigil AK
 Platform    : Antigravity Edge Compute Sandbox / Windows 11
 Runtime     : ONNX Runtime (CUDA > CPU execution provider fallback)
=============================================================================
"""
# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    import onnxruntime as ort
except ImportError as e:
    print(f"[FATAL] Missing dependency: {e}")
    print("[INFO]  Run: pip install -r requirements.txt")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Logger Configuration
# ──────────────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("PPE-Inference")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ──────────────────────────────────────────────────────────────────────────────

# Path to the quantized ONNX model (output of export_onnx.py)
ONNX_MODEL_PATH = Path(r"C:\Users\jigil ak\Documents\My Projects\Computer Vision Engineer_Assignment\models\ppe_yolov8n_fp16.onnx")

# Webcam configuration
CAMERA_INDEX    = 0         # Default webcam hardware index
CAMERA_WIDTH    = 1280      # Capture resolution width
CAMERA_HEIGHT   = 720       # Capture resolution height

# Model input dimensions (must match training/export config)
MODEL_INPUT_SIZE = 640

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.45   # Minimum confidence to keep a detection
NMS_IOU_THRESHOLD    = 0.50   # IoU threshold for Non-Maximum Suppression

# Class definitions — matches data.yaml from training
# Index → class name mapping
CLASS_NAMES = {
    0: "Gloves",
    1: "Vest",
    2: "Helmet",
    3: "Person",
}

# Color scheme for bounding boxes (BGR format for OpenCV)
# Protective gear = Green tones, Violations/Person = Red/Orange tones
CLASS_COLORS = {
    0: (0, 200, 100),    # Gloves    → Green
    1: (0, 200, 200),    # Vest      → Yellow-Green
    2: (0, 180, 0),      # Helmet    → Dark Green
    3: (200, 100, 50),   # Person    → Blue-ish (neutral)
}

# HUD styling
HUD_BG_COLOR     = (30, 30, 30)       # Dark background for HUD panel
HUD_TEXT_COLOR    = (220, 220, 220)    # Light gray text
HUD_ACCENT_COLOR = (0, 200, 100)      # Green accent
HUD_WARN_COLOR   = (0, 80, 255)       # Orange-red for warnings
HUD_FONT         = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE   = 0.55
HUD_THICKNESS    = 1


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing Module
# ──────────────────────────────────────────────────────────────────────────────

def letterbox(
    image: np.ndarray,
    target_size: int = 640,
    color: tuple = (114, 114, 114)
) -> tuple:
    """
    Applies letterbox resizing to preserve the original aspect ratio.
    
    The image is scaled to fit within a (target_size × target_size) canvas
    without distortion, with uniform padding applied to fill remaining space.
    
    Args:
        image:       Input BGR image (HWC format) from OpenCV.
        target_size: Target square dimension in pixels.
        color:       Padding fill color (BGR tuple).
    
    Returns:
        tuple: (letterboxed_image, scale_factor, (pad_w, pad_h))
            - letterboxed_image: Padded and resized image (target_size × target_size)
            - scale_factor: The scale applied to the original image
            - (pad_w, pad_h): Padding offsets for coordinate inversion
    """
    h, w = image.shape[:2]
    
    # Compute the scale factor to fit the image within the target canvas
    scale = min(target_size / w, target_size / h)
    
    # Compute new dimensions after scaling
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image using high-quality interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Compute padding to center the resized image on the canvas
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    # Create the letterboxed canvas and place the resized image
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return canvas, scale, (pad_w, pad_h)


def preprocess(image: np.ndarray, target_size: int = 640) -> tuple:
    """
    Complete preprocessing pipeline for ONNX Runtime inference.
    
    Pipeline:
        1. Letterbox resize (aspect-ratio preserving)
        2. BGR → RGB color space conversion
        3. HWC → CHW dimension transposition
        4. Float32 normalization [0, 255] → [0.0, 1.0]
        5. Batch dimension expansion (1, C, H, W)
    
    Args:
        image:       Raw BGR frame from OpenCV VideoCapture.
        target_size: Model input resolution.
    
    Returns:
        tuple: (input_tensor, scale, padding)
            - input_tensor: np.ndarray of shape (1, 3, target_size, target_size), float32
            - scale: Scale factor applied during letterboxing
            - padding: (pad_w, pad_h) padding offsets
    """
    # Step 1: Letterbox resize
    letterboxed, scale, padding = letterbox(image, target_size)
    
    # Step 2: BGR → RGB conversion (OpenCV uses BGR, models expect RGB)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    
    # Step 3: HWC → CHW transposition (Height, Width, Channels → Channels, Height, Width)
    chw = rgb.transpose(2, 0, 1)
    
    # Step 4: Float32 normalization — pixel values from [0, 255] to [0.0, 1.0]
    normalized = chw.astype(np.float32) / 255.0
    
    # Step 5: Add batch dimension → (1, 3, H, W)
    batched = np.expand_dims(normalized, axis=0)
    
    # Ensure contiguous memory layout for ONNX Runtime
    input_tensor = np.ascontiguousarray(batched)
    
    return input_tensor, scale, padding


# ──────────────────────────────────────────────────────────────────────────────
# Custom Non-Maximum Suppression (Vectorized, No PyTorch Dependency)
# ──────────────────────────────────────────────────────────────────────────────

def compute_iou_vectorized(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Computes IoU (Intersection over Union) between a single box and an array of boxes.
    
    Fully vectorized using NumPy — no loops, no PyTorch dependency.
    
    Args:
        box:   Single bounding box [x1, y1, x2, y2] — shape (4,)
        boxes: Array of bounding boxes — shape (N, 4), each row [x1, y1, x2, y2]
    
    Returns:
        np.ndarray: IoU scores — shape (N,)
    """
    # Compute intersection coordinates
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    # Compute intersection area (clamp to 0 for non-overlapping boxes)
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    
    # Compute areas of both boxes
    box_area   = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Compute union area
    union = box_area + boxes_area - intersection
    
    # Compute IoU, avoiding division by zero
    iou = np.where(union > 0, intersection / union, 0.0)
    
    return iou


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.50
) -> np.ndarray:
    """
    Performs class-aware Non-Maximum Suppression using vectorized NumPy operations.
    
    Algorithm:
        1. Sort detections by confidence score (descending)
        2. For each class independently:
           a. Select the highest-scoring box
           b. Compute IoU against all remaining boxes
           c. Suppress boxes with IoU > threshold
           d. Repeat until no boxes remain
    
    Args:
        boxes:         Bounding boxes — shape (N, 4), format [x1, y1, x2, y2]
        scores:        Confidence scores — shape (N,)
        class_ids:     Class indices — shape (N,)
        iou_threshold: IoU threshold above which overlapping boxes are suppressed.
    
    Returns:
        np.ndarray: Indices of surviving detections after NMS.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    keep = []
    unique_classes = np.unique(class_ids)
    
    # Process each class independently (class-aware NMS)
    for cls_id in unique_classes:
        # Filter detections for this class
        cls_mask = class_ids == cls_id
        cls_indices = np.where(cls_mask)[0]
        cls_boxes  = boxes[cls_indices]
        cls_scores = scores[cls_indices]
        
        # Sort by confidence (descending)
        sorted_order = np.argsort(-cls_scores)
        cls_indices = cls_indices[sorted_order]
        cls_boxes   = cls_boxes[sorted_order]
        
        # Greedy NMS loop
        while len(cls_indices) > 0:
            # Keep the highest-scoring box
            keep.append(cls_indices[0])
            
            if len(cls_indices) == 1:
                break
            
            # Compute IoU of the best box against all remaining boxes
            ious = compute_iou_vectorized(cls_boxes[0], cls_boxes[1:])
            
            # Keep only boxes with IoU below the threshold
            surviving = ious < iou_threshold
            cls_indices = cls_indices[1:][surviving]
            cls_boxes   = cls_boxes[1:][surviving]
    
    return np.array(keep, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Post-Processing Module
# ──────────────────────────────────────────────────────────────────────────────

def postprocess(
    output: np.ndarray,
    original_shape: tuple,
    scale: float,
    padding: tuple,
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.50
) -> list:
    """
    Decodes raw ONNX model output tensor into filtered, NMS-applied detections
    mapped back to the original full-resolution camera frame coordinates.
    
    YOLOv8 output format: (1, num_classes + 4, num_predictions)
        - First 4 rows: cx, cy, w, h (center-x, center-y, width, height)
        - Remaining rows: class probabilities (one per class)
    
    Pipeline:
        1. Transpose output to (num_predictions, num_classes + 4)
        2. Extract box coordinates and class scores
        3. Filter by confidence threshold
        4. Convert center-format to corner-format (x1, y1, x2, y2)
        5. Apply custom vectorized NMS
        6. Invert letterbox scaling to map onto original frame
    
    Args:
        output:         Raw model output tensor — shape (1, 4+C, N)
        original_shape: (height, width) of the original camera frame
        scale:          Letterbox scale factor from preprocessing
        padding:        (pad_w, pad_h) letterbox padding offsets
        conf_threshold: Minimum confidence for detection filtering
        iou_threshold:  IoU threshold for NMS
    
    Returns:
        list: List of detections, each as [x1, y1, x2, y2, confidence, class_id]
              where coordinates are in the original frame's pixel space.
    """
    # Squeeze batch dimension: (1, 4+C, N) → (4+C, N)
    predictions = output[0]
    
    # Transpose to (N, 4+C) for easier row-wise processing
    predictions = predictions.T
    
    num_classes = predictions.shape[1] - 4
    
    # Split into box coordinates and class scores
    boxes_cxcywh = predictions[:, :4]      # (N, 4) — cx, cy, w, h
    class_scores = predictions[:, 4:]      # (N, C) — one score per class
    
    # Get the maximum class score and corresponding class ID for each prediction
    max_scores = np.max(class_scores, axis=1)    # (N,)
    class_ids  = np.argmax(class_scores, axis=1) # (N,)
    
    # ── Confidence Filtering ──────────────────────────────────────────────
    conf_mask = max_scores >= conf_threshold
    if not np.any(conf_mask):
        return []
    
    boxes_cxcywh = boxes_cxcywh[conf_mask]
    max_scores   = max_scores[conf_mask]
    class_ids    = class_ids[conf_mask]
    
    # ── Convert Center-Format → Corner-Format ─────────────────────────────
    # (cx, cy, w, h) → (x1, y1, x2, y2)
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    # ── Non-Maximum Suppression ───────────────────────────────────────────
    keep_indices = non_maximum_suppression(
        boxes_xyxy, max_scores, class_ids, iou_threshold
    )
    
    if len(keep_indices) == 0:
        return []
    
    boxes_xyxy = boxes_xyxy[keep_indices]
    max_scores = max_scores[keep_indices]
    class_ids  = class_ids[keep_indices]
    
    # ── Invert Letterbox Scaling → Original Frame Coordinates ─────────────
    pad_w, pad_h = padding
    orig_h, orig_w = original_shape[:2]
    
    # Remove padding offset, then undo the scale
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / scale  # x1
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / scale  # y1
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / scale  # x2
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / scale  # y2
    
    # Clip to frame boundaries
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)
    
    # ── Assemble Final Detections ─────────────────────────────────────────
    detections = []
    for i in range(len(boxes_xyxy)):
        detections.append([
            int(boxes_xyxy[i, 0]),    # x1
            int(boxes_xyxy[i, 1]),    # y1
            int(boxes_xyxy[i, 2]),    # x2
            int(boxes_xyxy[i, 3]),    # y2
            float(max_scores[i]),     # confidence
            int(class_ids[i]),        # class_id
        ])
    
    return detections


# ──────────────────────────────────────────────────────────────────────────────
# HUD (Heads-Up Display) Rendering
# ──────────────────────────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Renders bounding boxes and class labels on the camera frame.
    
    Color coding:
        - Protective gear classes (Gloves, Vest, Helmet) → Green tones
        - Person (neutral detection)                     → Blue tone
    
    Args:
        frame:      Raw camera frame (BGR).
        detections: List of [x1, y1, x2, y2, confidence, class_id].
    
    Returns:
        np.ndarray: Frame with drawn bounding boxes and labels.
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        
        # Look up class name and color
        class_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        color      = CLASS_COLORS.get(cls_id, (200, 200, 200))
        
        # Draw bounding box with rounded corners effect (thick + thin overlay)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        
        # Prepare label text
        label = f"{class_name} {conf:.0%}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, HUD_FONT, HUD_FONT_SCALE, HUD_THICKNESS
        )
        
        # Draw label background (filled rectangle behind text)
        label_y1 = max(y1 - label_h - baseline - 6, 0)
        label_y2 = y1
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + label_w + 8, label_y2),
            color, -1, lineType=cv2.LINE_AA
        )
        
        # Draw label text (black text on colored background)
        cv2.putText(
            frame, label,
            (x1 + 4, label_y2 - baseline - 2),
            HUD_FONT, HUD_FONT_SCALE, (0, 0, 0),
            HUD_THICKNESS, cv2.LINE_AA
        )
    
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    preprocess_ms: float,
    inference_ms: float,
    postprocess_ms: float,
    detection_count: int
) -> np.ndarray:
    """
    Renders a professional on-screen HUD panel with real-time performance metrics.
    
    HUD Elements:
        - Real-time FPS (computed from model inference time only)
        - Pre-processing latency (ms)
        - Model inference latency (ms)
        - Post-processing / NMS latency (ms)
        - Active detection count
    
    Args:
        frame:          Camera frame to overlay the HUD on.
        fps:            Frames per second (model inference only).
        preprocess_ms:  Pre-processing stage latency in milliseconds.
        inference_ms:   ONNX model inference latency in milliseconds.
        postprocess_ms: Post-processing + NMS latency in milliseconds.
        detection_count: Number of active detections in this frame.
    
    Returns:
        np.ndarray: Frame with the HUD overlay.
    """
    h, w = frame.shape[:2]
    
    # ── HUD Panel Background (semi-transparent dark overlay) ──────────────
    panel_w = 320
    panel_h = 200
    panel_x = w - panel_w - 15
    panel_y = 15
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        HUD_BG_COLOR, -1
    )
    # Blend with transparency
    alpha = 0.75
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # ── Panel Border ──────────────────────────────────────────────────────
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        HUD_ACCENT_COLOR, 1, cv2.LINE_AA
    )
    
    # ── Title Bar ─────────────────────────────────────────────────────────
    title = "PPE DETECTION  |  EDGE HUD"
    cv2.putText(
        frame, title,
        (panel_x + 10, panel_y + 25),
        HUD_FONT, 0.50, HUD_ACCENT_COLOR,
        1, cv2.LINE_AA
    )
    
    # Divider line
    cv2.line(
        frame,
        (panel_x + 10, panel_y + 35),
        (panel_x + panel_w - 10, panel_y + 35),
        (80, 80, 80), 1, cv2.LINE_AA
    )
    
    # ── Performance Metrics ───────────────────────────────────────────────
    metrics = [
        ("FPS (Inference)", f"{fps:.1f}", HUD_ACCENT_COLOR if fps > 15 else HUD_WARN_COLOR),
        ("Preprocess",      f"{preprocess_ms:.1f} ms", HUD_TEXT_COLOR),
        ("Inference",       f"{inference_ms:.1f} ms",  HUD_TEXT_COLOR),
        ("NMS / Postproc",  f"{postprocess_ms:.1f} ms", HUD_TEXT_COLOR),
        ("Detections",      f"{detection_count}",       HUD_ACCENT_COLOR),
    ]
    
    y_offset = panel_y + 55
    for label, value, color in metrics:
        # Label (left-aligned)
        cv2.putText(
            frame, f"{label}:",
            (panel_x + 15, y_offset),
            HUD_FONT, HUD_FONT_SCALE, (150, 150, 150),
            HUD_THICKNESS, cv2.LINE_AA
        )
        # Value (right-aligned)
        cv2.putText(
            frame, value,
            (panel_x + 200, y_offset),
            HUD_FONT, HUD_FONT_SCALE, color,
            HUD_THICKNESS, cv2.LINE_AA
        )
        y_offset += 28
    
    # ── Bottom Status Bar ─────────────────────────────────────────────────
    status = "LIVE  |  YOLOv8n  |  ONNX FP16"
    cv2.putText(
        frame, status,
        (15, h - 15),
        HUD_FONT, 0.45, (100, 100, 100),
        1, cv2.LINE_AA
    )
    
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Runtime Session Manager
# ──────────────────────────────────────────────────────────────────────────────

def create_inference_session(model_path: Path) -> ort.InferenceSession:
    """
    Creates an optimized ONNX Runtime InferenceSession with automatic
    execution provider selection (CUDA preferred, CPU fallback).
    
    Session Options:
        - Graph optimization level: ORT_ENABLE_ALL
        - Execution mode: Sequential (for edge deployment)
        - Inter-op thread count: 1 (single-stream inference)
    
    Args:
        model_path: Absolute path to the .onnx model file.
    
    Returns:
        ort.InferenceSession: Ready-to-use inference session.
    
    Raises:
        FileNotFoundError: If the ONNX model file is not found.
        RuntimeError: If session creation fails.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n"
            f"Run 'python export_onnx.py' first to generate the model."
        )
    
    # Configure session options for edge deployment
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.inter_op_num_threads = 1
    
    # Execution provider priority: CUDA GPU → CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers
        )
        
        # Log which execution provider was actually selected
        active_providers = session.get_providers()
        logger.info(f"  ✓ Active execution providers: {active_providers}")
        
        if 'CUDAExecutionProvider' in active_providers:
            logger.info("  ✓ GPU acceleration: ENABLED (CUDA)")
        else:
            logger.info("  ⚠ GPU acceleration: DISABLED (running on CPU)")
        
        # Log model input/output metadata
        input_meta  = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        logger.info(f"  ✓ Input  : {input_meta.name} → {input_meta.shape} ({input_meta.type})")
        logger.info(f"  ✓ Output : {output_meta.name} → {output_meta.shape} ({output_meta.type})")
        
        return session
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create inference session: {e}")
        raise RuntimeError(f"ONNX Runtime session creation failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Inference Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main real-time inference loop:
    
    1. Initialize ONNX Runtime session
    2. Open webcam video stream
    3. For each frame:
       a. Preprocess (letterbox + normalize)
       b. Run ONNX inference
       c. Postprocess (decode + NMS)
       d. Render detections + HUD
       e. Display annotated frame
    4. Clean up resources on exit (ESC or 'q')
    """
    
    # ── Step 1: Initialize ONNX Runtime Session ───────────────────────────
    logger.info("[1/3] Initializing ONNX Runtime inference session...")
    try:
        session = create_inference_session(ONNX_MODEL_PATH)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"  ✗ {e}")
        sys.exit(1)
    
    # Cache input/output names for the inference loop
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # ── Step 2: Open Webcam Stream ────────────────────────────────────────
    logger.info(f"[2/3] Opening webcam (index={CAMERA_INDEX}, "
                f"resolution={CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        logger.error(f"  ✗ Failed to open webcam at index {CAMERA_INDEX}")
        logger.error(f"  ✗ Ensure a camera is connected and not in use by another app.")
        sys.exit(1)
    
    # Set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    # Verify actual resolution (camera may not support requested resolution)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"  ✓ Webcam opened: {actual_w}x{actual_h}")
    
    # ── Step 3: Real-Time Inference Loop ──────────────────────────────────
    logger.info("[3/3] Starting real-time inference — press 'q' or ESC to exit")
    logger.info("─" * 60)
    
    # Performance tracking (exponential moving average for smoother display)
    ema_alpha = 0.1  # Smoothing factor
    ema_fps = 0.0
    ema_preprocess_ms = 0.0
    ema_inference_ms = 0.0
    ema_postprocess_ms = 0.0
    frame_count = 0
    
    try:
        while True:
            # ── Capture Frame ─────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                logger.warning("  ⚠ Failed to read frame from webcam — retrying...")
                continue
            
            original_shape = frame.shape  # (H, W, C)
            
            # ── Stage 1: Preprocessing ────────────────────────────────────
            t_pre_start = time.perf_counter()
            input_tensor, scale, padding = preprocess(frame, MODEL_INPUT_SIZE)
            t_pre_end = time.perf_counter()
            preprocess_ms = (t_pre_end - t_pre_start) * 1000.0
            
            # ── Stage 2: ONNX Inference ───────────────────────────────────
            t_inf_start = time.perf_counter()
            outputs = session.run(
                [output_name],
                {input_name: input_tensor}
            )
            t_inf_end = time.perf_counter()
            inference_ms = (t_inf_end - t_inf_start) * 1000.0
            
            # ── Stage 3: Post-Processing & NMS ────────────────────────────
            t_post_start = time.perf_counter()
            detections = postprocess(
                outputs[0],
                original_shape,
                scale,
                padding,
                conf_threshold=CONFIDENCE_THRESHOLD,
                iou_threshold=NMS_IOU_THRESHOLD
            )
            t_post_end = time.perf_counter()
            postprocess_ms = (t_post_end - t_post_start) * 1000.0
            
            # ── FPS Calculation (pure model inference time) ───────────────
            # FPS is computed strictly from model execution time,
            # isolating it from rendering and I/O overhead
            current_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
            
            # Apply exponential moving average for smooth display
            if frame_count == 0:
                ema_fps = current_fps
                ema_preprocess_ms = preprocess_ms
                ema_inference_ms = inference_ms
                ema_postprocess_ms = postprocess_ms
            else:
                ema_fps = ema_alpha * current_fps + (1 - ema_alpha) * ema_fps
                ema_preprocess_ms = ema_alpha * preprocess_ms + (1 - ema_alpha) * ema_preprocess_ms
                ema_inference_ms = ema_alpha * inference_ms + (1 - ema_alpha) * ema_inference_ms
                ema_postprocess_ms = ema_alpha * postprocess_ms + (1 - ema_alpha) * ema_postprocess_ms
            
            frame_count += 1
            
            # ── Render Detections ─────────────────────────────────────────
            annotated_frame = draw_detections(frame, detections)
            
            # ── Render HUD Overlay ────────────────────────────────────────
            annotated_frame = draw_hud(
                annotated_frame,
                fps=ema_fps,
                preprocess_ms=ema_preprocess_ms,
                inference_ms=ema_inference_ms,
                postprocess_ms=ema_postprocess_ms,
                detection_count=len(detections)
            )
            
            # ── Display Frame ─────────────────────────────────────────────
            cv2.imshow("PPE Detection | Live Inference", annotated_frame)
            
            # ── Key Input Handling ────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to exit
                logger.info("  User requested exit — shutting down...")
                break
    
    except KeyboardInterrupt:
        logger.info("  Keyboard interrupt received — shutting down...")
    
    except Exception as e:
        logger.error(f"  ✗ Runtime error during inference: {e}", exc_info=True)
    
    finally:
        # ── Cleanup Resources ─────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        logger.info("  ✓ Resources released. Inference session terminated.")
        logger.info(f"  ✓ Total frames processed: {frame_count}")
        
        if frame_count > 0:
            logger.info(f"  ✓ Average FPS (inference): {ema_fps:.1f}")
            logger.info(f"  ✓ Average latency: "
                        f"pre={ema_preprocess_ms:.1f}ms | "
                        f"inf={ema_inference_ms:.1f}ms | "
                        f"post={ema_postprocess_ms:.1f}ms")


# ──────────────────────────────────────────────────────────────────────────────
# Script Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
