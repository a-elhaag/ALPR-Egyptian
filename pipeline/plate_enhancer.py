"""
Plate Enhancement Module
Specialized image processing for license plate crops

Why this stage is critical:
- Detected plates are often small (low resolution)
- May have motion blur, shadows, or poor contrast
- OCR accuracy is highly sensitive to image quality

This module applies targeted enhancements to maximize OCR success.
"""

from typing import Tuple

import cv2
import numpy as np

import config


def resize_plate(
    plate_image: np.ndarray,
    target_height: int = config.PLATE_TARGET_HEIGHT,
    min_width: int = config.PLATE_MIN_WIDTH
) -> np.ndarray:
    """
    Resize plate to optimal OCR resolution
    
    Why: Small plates have insufficient detail for accurate OCR.
    Upscaling to a standard size improves character recognition.
    
    Args:
        plate_image: Cropped plate image
        target_height: Target height in pixels
        min_width: Minimum width to ensure
        
    Returns:
        Resized plate image
    """
    height, width = plate_image.shape[:2]
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Calculate new dimensions
    new_height = target_height
    new_width = int(target_height * aspect_ratio)
    
    # Ensure minimum width
    if new_width < min_width:
        new_width = min_width
        new_height = int(min_width / aspect_ratio)
    
    # Use INTER_CUBIC for upscaling (better quality than INTER_LINEAR)
    resized = cv2.resize(
        plate_image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )
    
    return resized


def denoise_plate(plate_image: np.ndarray) -> np.ndarray:
    """
    Apply advanced denoising while preserving edges
    
    Uses Non-local Means Denoising which is more effective than
    bilateral filter for text images.
    
    Args:
        plate_image: Input plate image
        
    Returns:
        Denoised image
    """
    # For color images, use fastNlMeansDenoisingColored
    denoised = cv2.fastNlMeansDenoisingColored(
        plate_image,
        None,
        h=10,           # Luminance strength
        hColor=10,      # Color component strength
        templateWindowSize=7,
        searchWindowSize=21
    )
    return denoised


def enhance_contrast(plate_image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive histogram equalization to improve contrast
    
    Why: Plates often have poor contrast between characters and background.
    CLAHE enhances local contrast without over-brightening.
    
    Args:
        plate_image: Input plate image
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(plate_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_GRID_SIZE
    )
    l_enhanced = clahe.apply(l_channel)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def sharpen_plate(plate_image: np.ndarray) -> np.ndarray:
    """
    Apply unsharp masking to enhance character edges
    
    Why: Motion blur and camera defocus soften character edges.
    Sharpening makes characters more distinct for OCR.
    
    Technique: Unsharp masking
    1. Create blurred version
    2. Subtract from original to get high-frequency details
    3. Add scaled details back to original
    
    Args:
        plate_image: Input plate image
        
    Returns:
        Sharpened image
    """
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(
        plate_image,
        config.SHARPEN_KERNEL_SIZE,
        config.SHARPEN_SIGMA
    )
    
    # Unsharp mask: original + amount * (original - blurred)
    sharpened = cv2.addWeighted(
        plate_image, 1.0 + config.SHARPEN_AMOUNT,
        blurred, -config.SHARPEN_AMOUNT,
        0
    )
    
    return sharpened


def binarize_plate(plate_image: np.ndarray) -> np.ndarray:
    """
    Convert to binary (black/white) using adaptive thresholding
    
    Why: For very challenging cases, binary images can improve OCR.
    Adaptive thresholding handles varying lighting across the plate.
    
    Note: This is applied as an alternative enhancement, not always used.
    
    Args:
        plate_image: Input plate image
        
    Returns:
        Binary plate image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    # ADAPTIVE_THRESH_GAUSSIAN_C: threshold value is weighted sum of neighborhood
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
        config.ADAPTIVE_THRESHOLD_C
    )
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to BGR for consistency
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return binary_bgr


def deskew_plate(plate_image: np.ndarray) -> np.ndarray:
    """
    Correct plate rotation/skew for better OCR
    
    Uses Hough line detection to find dominant angle and rotate.
    
    Args:
        plate_image: Input plate image
        
    Returns:
        Deskewed plate image
    """
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return plate_image
    
    # Calculate angles of lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider small angles (likely plate edges)
            if abs(angle) < 15:
                angles.append(angle)
    
    if not angles:
        return plate_image
    
    # Use median angle for robustness
    median_angle = np.median(angles)
    
    # Only correct if angle is significant
    if abs(median_angle) < 0.5:
        return plate_image
    
    # Rotate image to correct skew
    h, w = plate_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        plate_image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def enhance_pipeline(plate_image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Complete plate enhancement pipeline
    
    Applies enhancements in optimal order:
    1. Resize to standard size
    2. Deskew (correct rotation)
    3. Denoise
    4. Enhance contrast
    5. Sharpen edges
    
    Args:
        plate_image: Cropped plate image from detector
        
    Returns:
        Tuple of (enhanced_plate, metadata)
    """
    # Store original size
    original_height, original_width = plate_image.shape[:2]
    
    # Step 1: Resize to optimal OCR size
    resized = resize_plate(plate_image)
    
    # Step 2: Deskew (correct rotation)
    deskewed = deskew_plate(resized)
    
    # Step 3: Denoise
    denoised = denoise_plate(deskewed)
    
    # Step 4: Enhance contrast
    contrast_enhanced = enhance_contrast(denoised)
    
    # Step 5: Sharpen
    sharpened = sharpen_plate(contrast_enhanced)
    
    # Metadata
    metadata = {
        'original_size': (original_width, original_height),
        'enhanced_size': (sharpened.shape[1], sharpened.shape[0]),
        'upscaled': original_height < config.PLATE_TARGET_HEIGHT,
        'deskewed': True,
        'denoised': True
    }
    
    return sharpened, metadata


def enhance_pipeline_with_binary(plate_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Enhancement pipeline that also produces a binary version
    
    Useful for trying multiple OCR attempts with different preprocessing.
    
    Args:
        plate_image: Cropped plate image
        
    Returns:
        Tuple of (enhanced_plate, binary_plate, metadata)
    """
    enhanced, metadata = enhance_pipeline(plate_image)
    binary = binarize_plate(enhanced)
    
    metadata['binary_created'] = True
    
    return enhanced, binary, metadata


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_plate = cv2.imread(sys.argv[1])
        if test_plate is not None:
            enhanced, meta = enhance_pipeline(test_plate)
            print(f"Enhancement successful: {meta}")
            cv2.imwrite("enhanced_plate.jpg", enhanced)
        else:
            print("Could not load image")
    else:
        print("Plate enhancement module loaded successfully")
