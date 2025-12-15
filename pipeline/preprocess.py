"""
Image Preprocessing Module
Handles image preparation for robust license plate detection

Key operations:
1. Resize to manageable dimensions
2. Denoise while preserving edges
3. Normalize lighting for varying conditions
"""

import cv2
import numpy as np
from typing import Tuple
import config


def resize_image(image: np.ndarray, max_dimension: int = config.MAX_IMAGE_DIMENSION) -> np.ndarray:
    """
    Resize image while preserving aspect ratio
    
    Why: Large images slow down processing without improving accuracy.
    Standardizing size ensures consistent performance.
    
    Args:
        image: Input image (BGR format)
        max_dimension: Maximum width or height
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    # Only resize if image exceeds max dimension
    if max(height, width) <= max_dimension:
        return image
    
    # Calculate scaling factor
    scale = max_dimension / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Use INTER_AREA for downscaling (best quality)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filtering to reduce noise while preserving edges
    
    Why: Real-world images contain noise from camera sensors, compression, etc.
    Bilateral filter smooths noise but keeps plate edges sharp, unlike Gaussian blur.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Denoised image
    """
    # Bilateral filter: reduces noise while preserving edges
    # This is crucial for maintaining plate boundary sharpness
    denoised = cv2.bilateralFilter(
        image,
        d=config.DENOISE_DIAMETER,
        sigmaColor=config.DENOISE_SIGMA_COLOR,
        sigmaSpace=config.DENOISE_SIGMA_SPACE
    )
    
    return denoised


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Why: License plates are often in shadow, backlit, or unevenly lit.
    CLAHE enhances local contrast adaptively, making plates visible
    in challenging lighting without over-brightening already bright areas.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Lighting-normalized image
    """
    # Convert to LAB color space
    # L channel = lightness, A/B = color channels
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel only (preserve colors)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_GRID_SIZE
    )
    l_channel_clahe = clahe.apply(l_channel)
    
    # Merge channels back
    lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
    
    # Convert back to BGR
    normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return normalized


def preprocess_pipeline(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline
    
    Applies all preprocessing steps in optimal order:
    1. Resize (reduce computational load)
    2. Denoise (improve signal quality)
    3. Normalize lighting (handle varying illumination)
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Tuple of (preprocessed_image, metadata_dict)
        metadata contains original dimensions and processing info
    """
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    
    # Step 1: Resize
    resized = resize_image(image)
    
    # Step 2: Denoise
    denoised = denoise_image(resized)
    
    # Step 3: Normalize lighting
    normalized = normalize_lighting(denoised)
    
    # Metadata for tracking
    metadata = {
        'original_size': (original_width, original_height),
        'processed_size': (normalized.shape[1], normalized.shape[0]),
        'resize_applied': (original_height != normalized.shape[0]) or (original_width != normalized.shape[1])
    }
    
    return normalized, metadata


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        test_image = cv2.imread(sys.argv[1])
        if test_image is not None:
            processed, meta = preprocess_pipeline(test_image)
            print(f"Preprocessing successful: {meta}")
            cv2.imwrite("preprocessed_test.jpg", processed)
        else:
            print("Could not load image")
    else:
        print("Preprocessing module loaded successfully")
