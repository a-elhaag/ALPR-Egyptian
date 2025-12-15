"""
Vehicle Detector Module (Optional Stage)
Isolates vehicle region to reduce background clutter

Why this stage exists:
- Reduces false positives from background text/signs
- Improves plate detection accuracy by narrowing search space
- Falls back gracefully if no vehicle detected
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from ultralytics import YOLO
import config


class VehicleDetector:
    """
    Optional vehicle detection stage
    
    Uses YOLO to detect cars and crop to vehicle region.
    This is beneficial when images contain significant background clutter.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to YOLO model (can be same as plate detector)
                       If None, uses the plate detection model
        """
        self.model_path = model_path or str(config.YOLO_MODEL_PATH)
        self.model = None
        
    def load_model(self) -> bool:
        """
        Load YOLO model for vehicle detection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = YOLO(self.model_path)
            print("✓ Vehicle detector loaded")
            return True
        except Exception as e:
            print(f"✗ Error loading vehicle detector: {e}")
            return False
    
    def detect_vehicle(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.3
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Detect vehicle and return cropped region
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Minimum confidence for vehicle detection
            
        Returns:
            Tuple of (cropped_image, metadata) if vehicle detected
            None if no vehicle detected (caller should use original image)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run inference
        # Note: This assumes the model can detect vehicles (class 'car', 'truck', etc.)
        # If using a plate-only model, this will not find vehicles
        results = self.model.predict(
            image,
            conf=conf_threshold,
            verbose=False
        )
        
        # Look for vehicle classes (car, truck, bus, motorcycle)
        # COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
        vehicle_classes = {2, 3, 5, 7}
        
        best_vehicle = None
        best_confidence = 0.0
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # Check if this is a vehicle class
                if class_id in vehicle_classes and confidence > best_confidence:
                    best_confidence = confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    best_vehicle = (int(x1), int(y1), int(x2), int(y2))
        
        # If vehicle detected, crop to that region
        if best_vehicle is not None:
            x1, y1, x2, y2 = best_vehicle
            
            # Add small margin around vehicle
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.shape[1], x2 + margin)
            y2 = min(image.shape[0], y2 + margin)
            
            cropped = image[y1:y2, x1:x2]
            
            metadata = {
                'vehicle_detected': True,
                'confidence': best_confidence,
                'bbox': (x1, y1, x2, y2)
            }
            
            return cropped, metadata
        
        # No vehicle detected - return None (caller uses original image)
        return None


def isolate_vehicle(
    image: np.ndarray,
    use_vehicle_detection: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to optionally isolate vehicle region
    
    Args:
        image: Input image
        use_vehicle_detection: Whether to attempt vehicle detection
        
    Returns:
        Tuple of (image_to_use, metadata)
        If vehicle detection disabled or fails, returns original image
    """
    if not use_vehicle_detection:
        return image, {'vehicle_detected': False}
    
    try:
        detector = VehicleDetector()
        if detector.load_model():
            result = detector.detect_vehicle(image)
            if result is not None:
                return result
    except Exception as e:
        print(f"Vehicle detection failed: {e}")
    
    # Fallback to original image
    return image, {'vehicle_detected': False}


if __name__ == "__main__":
    print("Vehicle detector module loaded")
    print("Note: This stage is optional and may not work if YOLO model")
    print("is trained only for license plates (not general object detection)")
