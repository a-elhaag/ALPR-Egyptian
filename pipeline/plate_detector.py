"""
License Plate Detector Module
Wrapper around YOLOv11 for license plate detection

Responsibilities:
- Load pretrained YOLO model
- Run inference on preprocessed images
- Extract plate bounding boxes with confidence scores
- Handle multiple detections and select best candidate
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch
from ultralytics import YOLO
import config


class PlateDetector:
    """
    YOLOv11-based license plate detector
    
    This class wraps the YOLO model and provides a clean interface
    for plate detection with confidence scoring.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the plate detector
        
        Args:
            model_path: Path to YOLO model file (.pt)
                       If None, uses config.YOLO_MODEL_PATH
        """
        self.model_path = model_path or str(config.YOLO_MODEL_PATH)
        self.model = None
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """
        Determine best available device (MPS for M3, CPU fallback)
        
        Why: M3's Metal Performance Shaders provide GPU-like acceleration.
        PyTorch automatically uses MPS when available.
        
        Returns:
            Device string ('mps', 'cuda', or 'cpu')
        """
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def load_model(self) -> bool:
        """
        Load the YOLO model from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load YOLO model
            self.model = YOLO(self.model_path)
            
            # Move model to appropriate device
            # Note: Ultralytics handles device placement automatically
            print(f"✓ YOLO model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            return False
    
    def detect_plates(
        self, 
        image: np.ndarray,
        conf_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD
    ) -> List[dict]:
        """
        Detect license plates in image
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Minimum confidence threshold
            
        Returns:
            List of detection dictionaries, each containing:
            - bbox: (x1, y1, x2, y2) coordinates
            - confidence: detection confidence score
            - crop: cropped plate image
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run YOLO inference
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=config.YOLO_IOU_THRESHOLD,
            imgsz=config.YOLO_IMAGE_SIZE,
            verbose=False  # Suppress YOLO output
        )
        
        detections = []
        
        # Process results
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
                
                # Crop plate region
                plate_crop = image[y1:y2, x1:x2]
                
                # Only include valid crops
                if plate_crop.size > 0:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'crop': plate_crop
                    })
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def get_best_plate(
        self,
        image: np.ndarray,
        conf_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD
    ) -> Optional[dict]:
        """
        Detect and return the highest-confidence plate
        
        Why: In most cases, we want the single best detection.
        This simplifies downstream processing.
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Minimum confidence threshold
            
        Returns:
            Best detection dict or None if no plates detected
        """
        detections = self.detect_plates(image, conf_threshold)
        
        if len(detections) > 0:
            return detections[0]  # Already sorted by confidence
        else:
            return None


def test_detector():
    """Test function for development"""
    detector = PlateDetector()
    
    if detector.load_model():
        print("Detector ready for inference")
        return detector
    else:
        print("Failed to load detector")
        return None


if __name__ == "__main__":
    # Test the detector
    detector = test_detector()
    
    import sys
    if len(sys.argv) > 1 and detector is not None:
        test_image = cv2.imread(sys.argv[1])
        if test_image is not None:
            result = detector.get_best_plate(test_image)
            if result:
                print(f"Plate detected with confidence: {result['confidence']:.2f}")
                cv2.imwrite("detected_plate.jpg", result['crop'])
            else:
                print("No plate detected")
