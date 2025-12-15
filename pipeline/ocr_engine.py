"""
OCR Engine Module - PaddleOCR only
Character-level detection for Egyptian license plates

Key approach:
- Use PaddleOCR with det=True to find text regions
- Split plate into letters (right) and digits (left) areas
- Recognize each region and concatenate
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

# PaddleOCR import
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


class OCREngine:
    """
    PaddleOCR-based engine with character-level detection
    """
    
    def __init__(self):
        self.ocr = None
        self.available = False

    def initialize_reader(self) -> bool:
        """Initialize PaddleOCR with Arabic support"""
        if PaddleOCR is None:
            print("✗ PaddleOCR not installed. Run: pip install paddlepaddle paddleocr")
            return False
        try:
            self.ocr = PaddleOCR(lang='ar')
            self.available = True
            print("✓ PaddleOCR initialized")
            return True
        except Exception as e:
            print(f"✗ PaddleOCR init failed: {e}")
            return False

    def _clean_char(self, ch: str) -> str:
        """
        Clean a single character:
        - Convert Arabic-Hindi numerals to Western
        - Map Latin misreads to Arabic equivalents
        """
        # Arabic-Hindi to Western numerals
        if ch in config.ARABIC_HINDI_TO_WESTERN:
            return config.ARABIC_HINDI_TO_WESTERN[ch]
        # Latin to Arabic mapping
        if ch in config.LATIN_TO_ARABIC_FALLBACK:
            return config.LATIN_TO_ARABIC_FALLBACK[ch]
        return ch
    
    def _clean_text(self, text: str) -> str:
        """
        Clean OCR output:
        1. Convert Arabic-Hindi numerals to Western
        2. Map Latin misreads to Arabic
        3. Keep only valid plate characters
        """
        result = []
        allowed_arabic = set(config.EGYPTIAN_ARABIC_LETTERS)
        
        for ch in text:
            cleaned = self._clean_char(ch)
            # Keep Arabic letters
            if cleaned in allowed_arabic:
                result.append(cleaned)
            # Keep Western digits
            elif cleaned.isdigit():
                result.append(cleaned)
            # Skip everything else (spaces, punctuation, etc.)
        
        return ''.join(result)

    def _split_plate_regions(self, plate_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split plate into letters region (right) and digits region (left).
        Egyptian plates: [digits on left] | [letters on right]
        """
        h, w = plate_image.shape[:2]
        
        # Approximate split: 45% left for digits, 55% right for letters
        split_x = int(w * 0.45)
        
        digits_region = plate_image[:, :split_x]
        letters_region = plate_image[:, split_x:]
        
        return letters_region, digits_region

    def _recognize_region(self, region: np.ndarray) -> Tuple[str, float, List[dict]]:
        """
        Recognize text in a single region using PaddleOCR predict method.
        """
        if not self.available or self.ocr is None:
            return "", 0.0, []
        
        if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
            return "", 0.0, []
            
        try:
            rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Use predict() method which returns list of OCRResult objects
            results = self.ocr.predict(rgb)
            
            if not results or len(results) == 0:
                return "", 0.0, []
            
            # Get first result (for single image input)
            ocr_result = results[0]
            
            # Extract rec_texts, rec_scores, and rec_polys from OCRResult
            rec_texts = ocr_result.get('rec_texts', []) if isinstance(ocr_result, dict) else getattr(ocr_result, 'rec_texts', [])
            rec_scores = ocr_result.get('rec_scores', []) if isinstance(ocr_result, dict) else getattr(ocr_result, 'rec_scores', [])
            rec_polys = ocr_result.get('rec_polys', []) if isinstance(ocr_result, dict) else getattr(ocr_result, 'rec_polys', [])
            
            if not rec_texts or len(rec_texts) == 0:
                return "", 0.0, []
            
            # Build detections with text, confidence, and position
            detections = []
            for i, text in enumerate(rec_texts):
                score = rec_scores[i] if i < len(rec_scores) else 0.0
                poly = rec_polys[i] if i < len(rec_polys) else None
                
                # Calculate center x for sorting
                center_x = 0
                if poly is not None and len(poly) > 0:
                    # poly might be array of points [[x1,y1], [x2,y2], ...]
                    try:
                        if hasattr(poly, '__iter__'):
                            xs = [p[0] if hasattr(p, '__getitem__') else 0 for p in poly]
                            center_x = float(np.mean(xs)) if xs else 0
                    except:
                        center_x = i * 100  # fallback: use index
                
                detections.append({
                    'text': text,
                    'confidence': float(score),
                    'center_x': center_x,
                    'bbox': poly
                })
            
            if not detections:
                return "", 0.0, []
            
            # Sort left to right
            detections.sort(key=lambda d: d['center_x'])
            
            all_text = ''.join(d['text'] for d in detections)
            avg_conf = float(np.mean([d['confidence'] for d in detections]))
            
            return all_text, avg_conf, detections
            
        except Exception as e:
            print(f"⚠ OCR error on region: {e}")
            import traceback
            traceback.print_exc()
            return "", 0.0, []

    def _recognize_whole_plate(self, plate_image: np.ndarray) -> Tuple[str, float, List[dict]]:
        """
        Recognize the whole plate at once.
        """
        return self._recognize_region(plate_image)

    def _recognize_split_plate(self, plate_image: np.ndarray) -> Tuple[str, float, List[dict]]:
        """
        Recognize plate by splitting into letters and digits regions.
        Returns: combined text as "letters digits" format
        """
        letters_region, digits_region = self._split_plate_regions(plate_image)
        
        # Recognize letters region (Arabic letters on right side)
        letters_text, letters_conf, letters_det = self._recognize_region(letters_region)
        letters_clean = self._clean_text(letters_text)
        # Keep only Arabic letters
        letters_only = ''.join(c for c in letters_clean if c in config.EGYPTIAN_ARABIC_LETTERS)
        
        # Recognize digits region (numbers on left side)
        digits_text, digits_conf, digits_det = self._recognize_region(digits_region)
        digits_clean = self._clean_text(digits_text)
        # Keep only digits
        digits_only = ''.join(c for c in digits_clean if c.isdigit())
        
        # Combine: letters + space + digits
        combined = f"{letters_only} {digits_only}".strip() if letters_only or digits_only else ""
        
        # Average confidence
        confs = []
        if letters_conf > 0:
            confs.append(letters_conf)
        if digits_conf > 0:
            confs.append(digits_conf)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        
        details = {
            'letters': letters_only,
            'digits': digits_only,
            'letters_raw': letters_text,
            'digits_raw': digits_text,
            'letters_conf': letters_conf,
            'digits_conf': digits_conf
        }
        
        return combined, avg_conf, [details]

    def recognize_with_fallback(
        self,
        enhanced_plate: np.ndarray,
        binary_plate: Optional[np.ndarray] = None
    ) -> Tuple[str, float, List[dict]]:
        """
        Try multiple approaches and return best result.
        
        1. Split plate recognition (letters + digits separately)
        2. Whole plate recognition
        3. Binary plate recognition (if provided)
        """
        results = []
        
        # Approach 1: Split recognition (more accurate for Egyptian plates)
        text1, conf1, det1 = self._recognize_split_plate(enhanced_plate)
        if text1:
            text1_clean = self._clean_text(text1.replace(' ', ''))
            # Re-format with space
            letters = ''.join(c for c in text1_clean if c in config.EGYPTIAN_ARABIC_LETTERS)
            digits = ''.join(c for c in text1_clean if c.isdigit())
            formatted = f"{letters} {digits}".strip()
            results.append(('split', formatted, conf1, det1))
        
        # Approach 2: Whole plate recognition
        text2, conf2, det2 = self._recognize_whole_plate(enhanced_plate)
        if text2:
            text2_clean = self._clean_text(text2)
            letters = ''.join(c for c in text2_clean if c in config.EGYPTIAN_ARABIC_LETTERS)
            digits = ''.join(c for c in text2_clean if c.isdigit())
            formatted = f"{letters} {digits}".strip()
            results.append(('whole', formatted, conf2, det2))
        
        # Approach 3: Binary plate (if provided)
        if binary_plate is not None:
            text3, conf3, det3 = self._recognize_whole_plate(binary_plate)
            if text3:
                text3_clean = self._clean_text(text3)
                letters = ''.join(c for c in text3_clean if c in config.EGYPTIAN_ARABIC_LETTERS)
                digits = ''.join(c for c in text3_clean if c.isdigit())
                formatted = f"{letters} {digits}".strip()
                results.append(('binary', formatted, conf3, det3))
        
        if not results:
            return "", 0.0, []
        
        # Score and sort results
        def score_result(entry):
            _, text_val, conf_val, _ = entry
            letters = ''.join(c for c in text_val if c in config.EGYPTIAN_ARABIC_LETTERS)
            digits = ''.join(c for c in text_val if c.isdigit())
            
            score = conf_val
            # Bonus for having both letters and digits
            if letters and digits:
                score += 0.15
            # Bonus for typical Egyptian format (2-3 letters + 2-4 digits)
            if 1 <= len(letters) <= 3 and 1 <= len(digits) <= 4:
                score += 0.1
            return score
        
        results.sort(key=score_result, reverse=True)
        best = results[0]
        
        return best[1], best[2], best[3]


def calculate_overall_confidence(
    detection_confidence: float,
    ocr_confidence: float,
    weights: Tuple[float, float] = (0.4, 0.6)
) -> float:
    """Combine detection and OCR confidence."""
    det_weight, ocr_weight = weights
    return (det_weight * detection_confidence) + (ocr_weight * ocr_confidence)


if __name__ == "__main__":
    engine = OCREngine()
    if engine.initialize_reader():
        print("OCR engine ready")
        
        import sys
        if len(sys.argv) > 1:
            test_image = cv2.imread(sys.argv[1])
            if test_image is not None:
                text, conf, _ = engine.recognize_with_fallback(test_image)
                print(f"Recognized: '{text}' (confidence: {conf:.2f})")
    else:
        print("Failed to initialize OCR engine")
