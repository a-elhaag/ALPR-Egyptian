"""
Confidence Estimation Utilities
Functions for assessing and reporting system confidence

Provides:
- Overall confidence calculation
- Failure mode diagnosis
- Human-readable status messages
"""

from typing import Tuple
import config


def estimate_overall_confidence(
    detection_conf: float,
    ocr_conf: float,
    format_valid: bool,
    text_length: int
) -> float:
    """
    Estimate overall system confidence
    
    Combines multiple factors:
    1. Plate detection confidence (40% weight)
    2. OCR confidence (60% weight)
    3. Format validity (penalty if invalid)
    4. Text length (penalty if too short)
    
    Args:
        detection_conf: YOLO detection confidence
        ocr_conf: OCR recognition confidence
        format_valid: Whether plate format is valid
        text_length: Length of recognized text
        
    Returns:
        Overall confidence [0, 1]
    """
    # Base confidence (weighted average)
    base_conf = (0.4 * detection_conf) + (0.6 * ocr_conf)
    
    # Apply penalties
    confidence = base_conf
    
    # Penalize invalid format
    if not format_valid:
        confidence *= 0.8
    
    # Penalize very short text (likely incomplete)
    if text_length < 5:
        confidence *= 0.85
    
    # Ensure in valid range
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence


def classify_failure_mode(
    detection_conf: float,
    ocr_conf: float,
    format_valid: bool,
    plate_detected: bool
) -> str:
    """
    Diagnose why confidence is low
    
    Helps users understand what went wrong.
    
    Args:
        detection_conf: Plate detection confidence
        ocr_conf: OCR confidence
        format_valid: Format validity
        plate_detected: Whether any plate was detected
        
    Returns:
        Failure mode description
    """
    if not plate_detected:
        return "no_plate_detected"
    
    if detection_conf < 0.5:
        return "low_confidence_detection"
    
    if ocr_conf < 0.5:
        return "low_confidence_ocr"
    
    if not format_valid:
        return "invalid_format"
    
    return "low_confidence_quality"


def generate_status_message(
    confidence: float,
    failure_mode: str = None
) -> str:
    """
    Generate human-readable status message
    
    Args:
        confidence: Overall confidence score
        failure_mode: Optional failure mode from classify_failure_mode()
        
    Returns:
        Status message string
    """
    if confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
        return config.STATUS_MESSAGES['high_confidence']
    
    elif confidence >= config.MEDIUM_CONFIDENCE_THRESHOLD:
        return config.STATUS_MESSAGES['medium_confidence']
    
    else:
        # Low confidence - use failure mode if provided
        if failure_mode and failure_mode in config.STATUS_MESSAGES:
            return config.STATUS_MESSAGES[failure_mode]
        else:
            return config.STATUS_MESSAGES['low_confidence_quality']


def get_confidence_level(confidence: float) -> str:
    """
    Get confidence level category
    
    Args:
        confidence: Confidence score [0, 1]
        
    Returns:
        'high', 'medium', or 'low'
    """
    if confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
        return 'high'
    elif confidence >= config.MEDIUM_CONFIDENCE_THRESHOLD:
        return 'medium'
    else:
        return 'low'


def format_confidence_report(
    overall_conf: float,
    detection_conf: float,
    ocr_conf: float,
    format_valid: bool
) -> str:
    """
    Create detailed confidence report
    
    Args:
        overall_conf: Overall confidence
        detection_conf: Detection confidence
        ocr_conf: OCR confidence
        format_valid: Format validity
        
    Returns:
        Multi-line report string
    """
    level = get_confidence_level(overall_conf)
    
    report = f"""
Confidence Report
{'='*50}
Overall Confidence: {overall_conf:.1%} ({level.upper()})
{'='*50}
Detection Confidence: {detection_conf:.1%}
OCR Confidence:       {ocr_conf:.1%}
Format Valid:         {'Yes' if format_valid else 'No'}
{'='*50}
"""
    
    return report


if __name__ == "__main__":
    # Test confidence utilities
    print("Testing confidence estimation...")
    
    # Test case 1: High confidence
    conf1 = estimate_overall_confidence(0.95, 0.92, True, 7)
    msg1 = generate_status_message(conf1)
    print(f"Test 1: {conf1:.2%} - {msg1}")
    
    # Test case 2: Medium confidence
    conf2 = estimate_overall_confidence(0.75, 0.65, True, 7)
    msg2 = generate_status_message(conf2)
    print(f"Test 2: {conf2:.2%} - {msg2}")
    
    # Test case 3: Low confidence
    conf3 = estimate_overall_confidence(0.45, 0.40, False, 3)
    msg3 = generate_status_message(conf3)
    print(f"Test 3: {conf3:.2%} - {msg3}")
    
    # Test report
    print(format_confidence_report(conf1, 0.95, 0.92, True))
