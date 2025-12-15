"""
Configuration file for Egyptian ALPR System
Contains all system parameters, paths, and domain-specific rules
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolo11m_car_plate_trained.pt"
YOLO_OCR_MODEL_PATH = MODELS_DIR / "yolo11m_car_plate_ocr.pt"  # Character detection model

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# YOLO Detection Parameters
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for plate detection
YOLO_IOU_THRESHOLD = 0.45         # IoU threshold for NMS
YOLO_IMAGE_SIZE = 640             # Input size for YOLO

# OCR Parameters
# Keep OCR focused on Arabic to avoid Latin hallucinations on Arabic plates.
OCR_LANGUAGES = ['ar']
OCR_GPU = False                   # Use CPU (M3 will use MPS automatically)
OCR_BATCH_SIZE = 1                # Process one image at a time
OCR_USE_PADDLE = True             # Enable PaddleOCR primary engine
OCR_PADDLE_USE_ANGLE = False      # Disable angle classifier for speed
OCR_ALLOW_ENGLISH = False         # Exclude English letters from OCR allowlist by default
# Map common Latin misreads to closest Arabic plate letters when English is disallowed
LATIN_TO_ARABIC_FALLBACK = {
    'y': 'ى', 'Y': 'ى',
    'w': 'و', 'W': 'و',
    'n': 'ن', 'N': 'ن',
    'g': 'ج', 'G': 'ج',
    'j': 'ج', 'J': 'ج',
    'h': 'ح', 'H': 'ح',
    's': 'س', 'S': 'س',
    't': 'ت', 'T': 'ت',
}

# ============================================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================================
# Preprocessing
MAX_IMAGE_DIMENSION = 1280        # Resize large images to this max dimension
DENOISE_DIAMETER = 9              # Bilateral filter diameter
DENOISE_SIGMA_COLOR = 75          # Bilateral filter sigma color
DENOISE_SIGMA_SPACE = 75          # Bilateral filter sigma space
PREPROCESS_FORCE_GRAYSCALE = True # Convert to grayscale (keeps 3-channel BGR) for consistent contrast
CLAHE_CLIP_LIMIT = 2.0            # CLAHE clip limit
CLAHE_GRID_SIZE = (8, 8)          # CLAHE tile grid size

# Plate Enhancement
PLATE_MIN_WIDTH = 200             # Minimum plate width for OCR
PLATE_TARGET_HEIGHT = 100         # Target height for plate resize
SHARPEN_KERNEL_SIZE = (5, 5)      # Unsharp mask kernel size
SHARPEN_SIGMA = 1.0               # Gaussian blur sigma for sharpening
SHARPEN_AMOUNT = 1.5              # Sharpening strength
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11  # Block size for adaptive threshold
ADAPTIVE_THRESHOLD_C = 2          # Constant subtracted from mean

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================
HIGH_CONFIDENCE_THRESHOLD = 0.8   # High confidence result
MEDIUM_CONFIDENCE_THRESHOLD = 0.5 # Medium confidence result
# Below MEDIUM_CONFIDENCE_THRESHOLD is considered low confidence

# ============================================================================
# EGYPTIAN LICENSE PLATE RULES
# ============================================================================
# Common Egyptian plate formats:
# Format 1: 3 Arabic letters + 4 digits (most common)
# Format 2: 2 Arabic letters + 4 digits
# Format 3: Special government plates with different patterns

# Valid Arabic letters on Egyptian plates (subset of Arabic alphabet)
EGYPTIAN_ARABIC_LETTERS = set([
    'ا', 'ب', 'ت', 'ج', 'ح', 'د', 'ر', 'س', 'ص', 'ط', 'ع', 'ف', 'ق', 'ل', 'م', 'ن', 'ه', 'و', 'ى'
])

# Valid English letters (sometimes used in special plates)
EGYPTIAN_ENGLISH_LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Valid digits
EGYPTIAN_DIGITS = set('0123456789')

# Arabic-Hindi numerals (used on Egyptian plates) -> Western mapping
ARABIC_HINDI_TO_WESTERN = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

# Western to Arabic-Hindi (for display)
WESTERN_TO_ARABIC_HINDI = {v: k for k, v in ARABIC_HINDI_TO_WESTERN.items()}

# Allowlist for OCR (limits spurious banner text)
if OCR_ALLOW_ENGLISH:
    _allow_letters = EGYPTIAN_ARABIC_LETTERS | EGYPTIAN_ENGLISH_LETTERS
else:
    _allow_letters = EGYPTIAN_ARABIC_LETTERS
OCR_ALLOWLIST = ''.join(sorted(_allow_letters)) \
    + ''.join(sorted(EGYPTIAN_DIGITS)) \
    + ''.join(sorted(ARABIC_HINDI_TO_WESTERN.keys()))

# Common OCR character confusions (for normalization)
CHAR_NORMALIZATION_MAP = {
    'O': '0',  # Letter O to digit 0
    'o': '0',
    'I': '1',  # Letter I to digit 1
    'l': '1',  # Lowercase L to digit 1
    'Z': '2',  # Sometimes confused
    'S': '5',  # Sometimes confused
    'B': '8',  # Sometimes confused
}

# Plate format patterns (regex-like)
PLATE_PATTERNS = [
    # 3 Arabic letters + 4 digits
    {'arabic_letters': 3, 'digits': 4},
    # 2 Arabic letters + 4 digits
    {'arabic_letters': 2, 'digits': 4},
    # 3 Arabic letters + 3 digits (older format)
    {'arabic_letters': 3, 'digits': 3},
]

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
BBOX_COLOR = (0, 255, 0)          # Green for bounding boxes
BBOX_THICKNESS = 2                # Bounding box line thickness
TEXT_COLOR = (0, 255, 0)          # Green for text
TEXT_THICKNESS = 2                # Text thickness
FONT_SCALE = 0.6                  # Font scale for annotations

# ============================================================================
# SYSTEM MESSAGES
# ============================================================================
STATUS_MESSAGES = {
    'high_confidence': "✓ High confidence result",
    'medium_confidence': "⚠ Plate detected but text unclear",
    'low_confidence_detection': "✗ Low confidence detection - plate may not be visible",
    'low_confidence_ocr': "✗ Low confidence OCR - plate image quality insufficient",
    'low_confidence_quality': "✗ Low confidence due to image quality (blur, lighting, angle)",
    'no_plate_detected': "✗ No license plate detected in image",
    'processing_error': "✗ Error during processing",
}

# ============================================================================
# DEVICE CONFIGURATION (M3 Optimization)
# ============================================================================
# PyTorch will automatically use MPS (Metal Performance Shaders) on M3
# This provides GPU-like acceleration on Apple Silicon
DEVICE_PREFERENCE = 'mps'  # Will fall back to CPU if MPS unavailable
