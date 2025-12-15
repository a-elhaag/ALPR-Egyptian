"""
Visualization Utilities
Drawing and display functions for pipeline stages

Provides professional visualization for:
- Bounding boxes with labels
- Stage-by-stage pipeline outputs
- Confidence indicators
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    confidence: Optional[float] = None,
    color: Tuple[int, int, int] = config.BBOX_COLOR,
    thickness: int = config.BBOX_THICKNESS
) -> np.ndarray:
    """
    Draw bounding box with optional label and confidence
    
    Args:
        image: Input image
        bbox: (x1, y1, x2, y2) coordinates
        label: Text label
        confidence: Optional confidence score
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with bounding box drawn
    """
    img_copy = image.copy()
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text
    if confidence is not None:
        text = f"{label} {confidence:.2%}" if label else f"{confidence:.2%}"
    else:
        text = label
    
    # Draw label background and text
    if text:
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = config.FONT_SCALE
        font_thickness = config.TEXT_THICKNESS
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            img_copy,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled
        )
        
        # Draw text
        cv2.putText(
            img_copy,
            text,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            font_thickness
        )
    
    return img_copy


def annotate_image(
    image: np.ndarray,
    text: str,
    position: str = 'top',
    color: Tuple[int, int, int] = config.TEXT_COLOR
) -> np.ndarray:
    """
    Add text annotation to image
    
    Args:
        image: Input image
        text: Text to add
        position: 'top' or 'bottom'
        color: Text color (BGR)
        
    Returns:
        Annotated image
    """
    img_copy = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.FONT_SCALE
    font_thickness = config.TEXT_THICKNESS
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    
    # Calculate position
    if position == 'top':
        x = 10
        y = text_height + 10
    else:  # bottom
        x = 10
        y = image.shape[0] - 10
    
    # Draw background
    cv2.rectangle(
        img_copy,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        img_copy,
        text,
        (x, y),
        font,
        font_scale,
        color,
        font_thickness
    )
    
    return img_copy


def create_stage_grid(
    stages: List[Tuple[str, np.ndarray]],
    grid_cols: int = 2
) -> np.ndarray:
    """
    Create a grid visualization of pipeline stages
    
    Args:
        stages: List of (stage_name, image) tuples
        grid_cols: Number of columns in grid
        
    Returns:
        Grid image with all stages
    """
    if not stages:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Calculate grid dimensions
    num_stages = len(stages)
    grid_rows = (num_stages + grid_cols - 1) // grid_cols
    
    # Find max dimensions for each cell
    max_height = max(img.shape[0] for _, img in stages)
    max_width = max(img.shape[1] for _, img in stages)
    
    # Create grid
    grid_height = grid_rows * (max_height + 40)  # +40 for labels
    grid_width = grid_cols * (max_width + 20)    # +20 for spacing
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Place images in grid
    for idx, (stage_name, img) in enumerate(stages):
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Calculate position
        y_offset = row * (max_height + 40) + 30
        x_offset = col * (max_width + 20) + 10
        
        # Ensure image is 3-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Place image
        h, w = img.shape[:2]
        grid[y_offset:y_offset+h, x_offset:x_offset+w] = img
        
        # Add label
        cv2.putText(
            grid,
            stage_name,
            (x_offset, y_offset - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return grid


def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """
    Get color based on confidence level
    
    Args:
        confidence: Confidence score [0, 1]
        
    Returns:
        BGR color tuple
    """
    # Use blue tones instead of green/orange/red
    if confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
        return (250, 165, 96)   # Light blue (BGR for #60a5fa)
    elif confidence >= config.MEDIUM_CONFIDENCE_THRESHOLD:
        return (214, 143, 87)   # Mid blue
    else:
        return (170, 120, 80)   # Darker blue


def create_confidence_bar(
    confidence: float,
    width: int = 400,
    height: int = 40
) -> np.ndarray:
    """
    Create a visual confidence bar
    
    Args:
        confidence: Confidence score [0, 1]
        width: Bar width in pixels
        height: Bar height in pixels
        
    Returns:
        Confidence bar image
    """
    # Create dark background
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (59, 41, 30)  # BGR for #1e293b
    
    # Calculate fill width
    fill_width = int(width * confidence)
    
    # Get color
    color = get_confidence_color(confidence)
    
    # Draw filled portion
    cv2.rectangle(bar, (0, 0), (fill_width, height), color, -1)
    
    # Draw border
    cv2.rectangle(bar, (0, 0), (width-1, height-1), (200, 200, 200), 2)
    
    # Add text
    text = f"{confidence:.1%}"
    cv2.putText(
        bar,
        text,
        (width // 2 - 30, height // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    return bar


if __name__ == "__main__":
    # Test visualization functions
    test_img = np.ones((300, 400, 3), dtype=np.uint8) * 200
    
    # Test bbox
    bbox_img = draw_bbox(test_img, (50, 50, 350, 250), "Test Plate", 0.95)
    cv2.imwrite("test_bbox.jpg", bbox_img)
    
    # Test confidence bar
    conf_bar = create_confidence_bar(0.87)
    cv2.imwrite("test_confidence.jpg", conf_bar)
    
    print("Visualization utilities test complete")
