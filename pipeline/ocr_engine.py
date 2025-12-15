"""
OCR Engine Module
Hybrid OCR approach combining YOLO character detection with EasyOCR

Key features:
- YOLO-based character detection for accurate localization
- EasyOCR for Arabic + English text recognition
- Character-level confidence tracking
- Multiple recognition attempts with different preprocessing
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config
from pipeline import postprocess

# Optional imports (PaddleOCR, EasyOCR)
try:
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None

try:
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None


class YOLOCharacterDetector:
    """
    YOLO-based character detector for license plates
    
    Uses a fine-tuned YOLO model to detect individual characters,
    providing more accurate localization than end-to-end OCR.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize character detector"""
        self.model_path = model_path or str(config.YOLO_OCR_MODEL_PATH)
        self.model = None
        self.class_names = []
        
    def load_model(self) -> bool:
        """Load YOLO character detection model"""
        try:
            if not Path(self.model_path).exists():
                print(f"⚠ YOLO OCR model not found at {self.model_path}")
                return False
                
            self.model = YOLO(self.model_path)
            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            print("✓ YOLO character detector loaded")
            return True
        except Exception as e:
            print(f"⚠ Error loading YOLO OCR model: {e}")
            return False
    
    def detect_characters(
        self,
        plate_image: np.ndarray,
        conf_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Detect individual characters in plate image
        
        Returns list of detections sorted left-to-right (for LTR) or 
        right-to-left (for Arabic text).
        """
        if self.model is None:
            return []
            
        results = self.model.predict(
            plate_image,
            conf=conf_threshold,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get character label from class names
                char = self.class_names.get(class_id, str(class_id)) if isinstance(self.class_names, dict) else (
                    self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                )
                
                detections.append({
                    'char': char,
                    'confidence': confidence,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center_x': (x1 + x2) / 2
                })
        
        # Sort by x-position (handles both LTR and RTL by position)
        detections.sort(key=lambda d: d['center_x'])
        
        return detections
    
    def get_plate_text(
        self,
        plate_image: np.ndarray,
        conf_threshold: float = 0.3
    ) -> Tuple[str, float, List[Dict]]:
        """
        Get complete plate text from character detections
        
        Returns:
            Tuple of (text, average_confidence, character_details)
        """
        detections = self.detect_characters(plate_image, conf_threshold)
        
        if not detections:
            return "", 0.0, []
        
        # Combine characters
        text = ''.join(d['char'] for d in detections)
        avg_confidence = np.mean([d['confidence'] for d in detections])
        
        return text, avg_confidence, detections


class PaddleOCREngine:
    """
    PaddleOCR wrapper for Arabic plates (faster than EasyOCR)
    """

    def __init__(self):
        self.ocr = None
        self.available = False

    def initialize(self) -> bool:
        if PaddleOCR is None:
            print("ℹ PaddleOCR not installed")
            return False
        try:
            self.ocr = PaddleOCR(
                lang='ar',
                use_gpu=config.OCR_GPU,
                use_angle_cls=config.OCR_PADDLE_USE_ANGLE
            )
            self.available = True
            print("✓ PaddleOCR initialized")
            return True
        except Exception as e:
            print(f"⚠ PaddleOCR init failed: {e}")
            return False

    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float, List[dict]]:
        if not self.available or self.ocr is None:
            return "", 0.0, []
        try:
            # Paddle expects RGB
            rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            results = self.ocr.ocr(rgb, det=False, rec=True)
            if not results:
                return "", 0.0, []

            texts = []
            confs = []
            details = []
            for item in results[0]:
                text, score = item
                texts.append(text)
                confs.append(score)
                details.append({'text': text, 'confidence': score})

            combined = ''.join(texts)
            overall = float(np.mean(confs)) if confs else 0.0
            return combined, overall, details
        except Exception as e:
            print(f"⚠ PaddleOCR inference error: {e}")
            return "", 0.0, []


class OCREngine:
    """
    Hybrid OCR engine combining YOLO detection with EasyOCR
    
    Strategy:
    1. Try YOLO character detection first (if model available)
    2. Fall back to EasyOCR for full text recognition
    3. Combine results for best accuracy
    """
    
    def __init__(self):
        """Initialize OCR engine"""
        self.reader = None
        self.paddle_engine = PaddleOCREngine()
        self.yolo_detector = YOLOCharacterDetector()
        self.use_yolo_ocr = False
        self.use_paddle = False

    @staticmethod
    def _filter_allowlist(text: str) -> str:
        """
        Keep only allowed characters and convert Latin misreads to Arabic.
        
        Steps:
        1. Convert Arabic-Hindi numerals to Western
        2. Map common Latin misreads to Arabic letters (if English not allowed)
        3. Filter to allowlist
        """
        result = []
        allowed = set(config.EGYPTIAN_ARABIC_LETTERS) | set(config.EGYPTIAN_DIGITS) | set(config.ARABIC_HINDI_TO_WESTERN.keys())
        if config.OCR_ALLOW_ENGLISH:
            allowed |= set(config.EGYPTIAN_ENGLISH_LETTERS)
        
        for ch in text:
            # Convert Arabic-Hindi numerals
            if ch in config.ARABIC_HINDI_TO_WESTERN:
                result.append(config.ARABIC_HINDI_TO_WESTERN[ch])
            # Map Latin misreads to Arabic (when English disabled)
            elif not config.OCR_ALLOW_ENGLISH and ch in config.LATIN_TO_ARABIC_FALLBACK:
                result.append(config.LATIN_TO_ARABIC_FALLBACK[ch])
            # Keep if in allowlist
            elif ch in allowed:
                result.append(ch)
            # Drop digits that are already Western
            elif ch.isdigit():
                result.append(ch)
        
        return ''.join(result)
        
    def initialize_reader(self) -> bool:
        """
        Load PaddleOCR, EasyOCR, and YOLO character detector
        
        Returns:
            True if at least one OCR backend loaded successfully
        """
        loaded_any = False

        # PaddleOCR (primary if enabled)
        if config.OCR_USE_PADDLE:
            self.use_paddle = self.paddle_engine.initialize()
            loaded_any = loaded_any or self.use_paddle

        # EasyOCR as secondary
        if easyocr is not None:
            try:
                print("Loading EasyOCR (this may take a moment on first run)...")
                self.reader = easyocr.Reader(
                    config.OCR_LANGUAGES,
                    gpu=config.OCR_GPU
                )
                loaded_any = True
                print("✓ EasyOCR initialized")
            except Exception as e:
                print(f"⚠ EasyOCR init failed: {e}")
        else:
            print("ℹ EasyOCR not installed")

        # YOLO character detector
        if self.yolo_detector.load_model():
            self.use_yolo_ocr = True
            loaded_any = True
            print("✓ YOLO character detector enabled")
        else:
            print("ℹ YOLO OCR model not available")

        if not loaded_any:
            print("✗ No OCR backend loaded")
        return loaded_any
    
    def recognize_text(
        self,
        plate_image: np.ndarray,
        detail: int = 1
    ) -> Tuple[str, float, List[dict]]:
        """
        Extract text from plate image using EasyOCR
        
        Args:
            plate_image: Enhanced plate image
            detail: EasyOCR detail level
            
        Returns:
            Tuple of (text, overall_confidence, character_details)
        """
        if self.reader is None:
            raise RuntimeError("OCR reader not initialized")
        
        results = self.reader.readtext(
            plate_image,
            detail=detail,
            paragraph=False,
            allowlist=config.OCR_ALLOWLIST  # Limit to valid plate characters to avoid headers/banners
        )
        
        if len(results) == 0:
            return "", 0.0, []
        
        all_text = []
        all_confidences = []
        character_details = []
        
        for bbox, text, confidence in results:
            all_text.append(text)
            all_confidences.append(confidence)
            character_details.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        combined_text = ''.join(all_text)
        combined_text = self._filter_allowlist(combined_text)
        overall_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return combined_text, overall_confidence, character_details
    
    def recognize_with_yolo(
        self,
        plate_image: np.ndarray
    ) -> Tuple[str, float, List[dict]]:
        """
        Use YOLO character detection for OCR
        
        Returns:
            Tuple of (text, confidence, details)
        """
        if not self.use_yolo_ocr:
            return "", 0.0, []

        text, conf, details = self.yolo_detector.get_plate_text(plate_image)
        if text:
            text = self._filter_allowlist(text)
        return text, conf, details
    
    def recognize_with_fallback(
        self,
        enhanced_plate: np.ndarray,
        binary_plate: Optional[np.ndarray] = None
    ) -> Tuple[str, float, List[dict]]:
        """
        Try multiple OCR approaches and return best result
        
        Strategy:
        1. YOLO character detection (fast, localized)
        2. PaddleOCR (primary OCR if available)
        3. EasyOCR on enhanced plate (fallback)
        4. EasyOCR on binary plate (optional fallback)
        
        Args:
            enhanced_plate: Standard enhanced plate
            binary_plate: Optional binary version
            
        Returns:
            Best (text, confidence, details) tuple
        """
        results = []
        
        # Attempt 1: YOLO character detection
        if self.use_yolo_ocr:
            text, conf, details = self.recognize_with_yolo(enhanced_plate)
            if text:
                results.append(('yolo', text, conf, details))
                # If YOLO confidence is high, use it directly
                if conf > 0.7:
                    return text, conf, details

        # Attempt 2: PaddleOCR (primary OCR if available)
        if self.use_paddle:
            text_p, conf_p, details_p = self.paddle_engine.recognize(enhanced_plate)
            if text_p:
                text_p = self._filter_allowlist(text_p)
                results.append(('paddle', text_p, conf_p, details_p))
                if conf_p > 0.8:
                    return text_p, conf_p, details_p
            # Try Paddle on binary plate if available
            if binary_plate is not None:
                text_pb, conf_pb, details_pb = self.paddle_engine.recognize(binary_plate)
                if text_pb:
                    text_pb = self._filter_allowlist(text_pb)
                    results.append(('paddle_binary', text_pb, conf_pb, details_pb))
                    if conf_pb > 0.8:
                        return text_pb, conf_pb, details_pb
        
        # Attempt 3: EasyOCR on enhanced plate
        if self.reader is not None:
            text1, conf1, details1 = self.recognize_text(enhanced_plate)
            if text1:
                results.append(('easyocr_enhanced', text1, conf1, details1))
        
        # Attempt 4: EasyOCR on binary plate
        if binary_plate is not None and self.reader is not None:
            text2, conf2, details2 = self.recognize_text(binary_plate)
            if text2:
                results.append(('easyocr_binary', text2, conf2, details2))
        
        # Select best result
        if not results:
            return "", 0.0, []

        def result_score(entry: Tuple[str, str, float, List[dict]]) -> float:
            """Score results with bias toward valid Egyptian format and presence of letters+digits."""
            _, text_val, conf_val, _ = entry
            is_valid, _ = postprocess.validate_egyptian_format(text_val)
            letters, digits = postprocess.extract_components(text_val)
            bonus = 0.0
            if is_valid:
                bonus += 0.1
            if letters and digits:
                bonus += 0.05
            return conf_val + bonus

        results.sort(key=result_score, reverse=True)
        best = results[0]

        return best[1], best[2], best[3]


def calculate_overall_confidence(
    detection_confidence: float,
    ocr_confidence: float,
    weights: Tuple[float, float] = (0.4, 0.6)
) -> float:
    """
    Combine detection and OCR confidence into overall confidence
    
    Why: Both stages contribute to final reliability.
    OCR confidence is weighted higher because text accuracy is the end goal.
    
    Args:
        detection_confidence: YOLO plate detection confidence
        ocr_confidence: OCR text recognition confidence
        weights: (detection_weight, ocr_weight) - should sum to 1.0
        
    Returns:
        Overall confidence score [0, 1]
    """
    det_weight, ocr_weight = weights
    overall = (det_weight * detection_confidence) + (ocr_weight * ocr_confidence)
    return overall


if __name__ == "__main__":
    # Test OCR engine initialization
    engine = OCREngine()
    if engine.initialize_reader():
        print("OCR engine ready for text recognition")
        
        import sys
        if len(sys.argv) > 1:
            test_image = cv2.imread(sys.argv[1])
            if test_image is not None:
                text, conf, details = engine.recognize_text(test_image)
                print(f"Recognized: '{text}' (confidence: {conf:.2f})")
            else:
                print("Could not load test image")
    else:
        print("Failed to initialize OCR engine")
