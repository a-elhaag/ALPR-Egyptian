"""
Post-Processing Module
Apply Egyptian license plate domain rules

This is the "WOW FACTOR" module that demonstrates domain expertise.

Key functions:
1. Validate against Egyptian plate formats
2. Normalize common OCR errors
3. Convert Arabic-Hindi numerals to Western
4. Remove invalid characters
5. Apply format-specific rules
"""

import re
from typing import Optional, Tuple

import config


def is_arabic_char(char: str) -> bool:
    """Check if character is Arabic"""
    return char in config.EGYPTIAN_ARABIC_LETTERS


def is_english_char(char: str) -> bool:
    """Check if character is English letter"""
    return char.upper() in config.EGYPTIAN_ENGLISH_LETTERS


def is_digit(char: str) -> bool:
    """Check if character is a digit (Western or Arabic-Hindi)"""
    return char in config.EGYPTIAN_DIGITS or char in config.ARABIC_HINDI_TO_WESTERN


def is_arabic_hindi_digit(char: str) -> bool:
    """Check if character is an Arabic-Hindi numeral"""
    return char in config.ARABIC_HINDI_TO_WESTERN


def convert_arabic_hindi_numerals(text: str) -> str:
    """
    Convert Arabic-Hindi numerals (٠١٢٣٤٥٦٧٨٩) to Western (0123456789)
    
    Egyptian plates use Arabic-Hindi numerals, but OCR may output either format.
    Standardizing to Western digits simplifies downstream processing.
    
    Args:
        text: Input text with potential Arabic-Hindi numerals
        
    Returns:
        Text with Western numerals only
    """
    result = []
    for char in text:
        if char in config.ARABIC_HINDI_TO_WESTERN:
            result.append(config.ARABIC_HINDI_TO_WESTERN[char])
        else:
            result.append(char)
    return ''.join(result)


def convert_western_to_arabic_hindi(text: str) -> str:
    """
    Convert Western numerals (0123456789) to Arabic-Hindi (٠١٢٣٤٥٦٧٨٩)
    
    Used for display so the UI shows native numerals while internal
    validation stays in Western digits.
    """
    result = []
    for char in text:
        if char in config.WESTERN_TO_ARABIC_HINDI:
            result.append(config.WESTERN_TO_ARABIC_HINDI[char])
        else:
            result.append(char)
    return ''.join(result)


def normalize_characters(text: str) -> str:
    """
    Fix common OCR character confusions and normalize numerals
    
    Why: OCR frequently confuses similar-looking characters:
    - Letter O vs digit 0
    - Letter I vs digit 1
    - Arabic-Hindi numerals vs Western numerals
    
    Egyptian plates use specific character sets, so we can
    intelligently correct these errors.
    
    Args:
        text: Raw OCR output
        
    Returns:
        Normalized text
    """
    # First convert Arabic-Hindi numerals to Western
    text = convert_arabic_hindi_numerals(text)
    
    normalized = []
    for char in text:
        # Convert common Latin misreads to Arabic if English is not allowed
        if not config.OCR_ALLOW_ENGLISH and char in config.LATIN_TO_ARABIC_FALLBACK:
            normalized.append(config.LATIN_TO_ARABIC_FALLBACK[char])
            continue

        # Apply normalization map
        if char in config.CHAR_NORMALIZATION_MAP:
            normalized.append(config.CHAR_NORMALIZATION_MAP[char])
        else:
            normalized.append(char)
    
    return ''.join(normalized)


def remove_invalid_chars(text: str) -> str:
    """
    Remove characters that cannot appear on Egyptian plates
    
    Valid characters:
    - Arabic letters (subset defined in config)
    - Digits 0-9
    - Spaces and hyphens (separators)
    
    Args:
        text: Input text
        
    Returns:
        Filtered text
    """
    valid_chars = []
    
    for char in text:
        if (is_arabic_char(char) or 
            is_digit(char) or 
            char in ' -'):
            valid_chars.append(char)
        # Optionally keep English letters for special plates
        elif config.OCR_ALLOW_ENGLISH and is_english_char(char):
            valid_chars.append(char)
    
    return ''.join(valid_chars)


def extract_components(text: str) -> Tuple[str, str]:
    """
    Separate Arabic letters from digits
    
    Egyptian plates typically have:
    - Arabic letters (prefix)
    - Digits (suffix)
    
    Args:
        text: Plate text
        
    Returns:
        Tuple of (letters, digits)
    """
    letters = []
    digits = []
    
    for char in text:
        if is_arabic_char(char) or is_english_char(char):
            letters.append(char)
        elif is_digit(char):
            digits.append(char)
    
    return ''.join(letters), ''.join(digits)


def validate_egyptian_format(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate text against known Egyptian plate formats
    
    Common formats:
    1. 3 Arabic letters + 4 digits (most common)
    2. 2 Arabic letters + 4 digits
    3. 3 Arabic letters + 3 digits (older)
    
    Args:
        text: Processed plate text
        
    Returns:
        Tuple of (is_valid, format_description)
    """
    letters, digits = extract_components(text)
    
    num_letters = len(letters)
    num_digits = len(digits)
    
    # Check against known patterns
    for pattern in config.PLATE_PATTERNS:
        if (num_letters == pattern.get('arabic_letters', 0) and
            num_digits == pattern.get('digits', 0)):
            format_desc = f"{num_letters} letters + {num_digits} digits"
            return True, format_desc
    
    # Not a recognized format
    return False, None


def apply_format_rules(text: str) -> Tuple[str, dict]:
    """
    Apply all Egyptian plate rules and formatting
    
    Complete post-processing pipeline:
    1. Normalize character confusions
    2. Remove invalid characters
    3. Validate format
    4. Format output
    
    Args:
        text: Raw OCR output
        
    Returns:
        Tuple of (processed_text, metadata)
        metadata contains validation info
    """
    # Step 1: Normalize characters
    normalized = normalize_characters(text)
    
    # Step 2: Remove invalid characters
    filtered = remove_invalid_chars(normalized)
    
    # Step 3: Validate format
    is_valid, format_desc = validate_egyptian_format(filtered)
    
    # Step 4: Format output (add space between letters and digits)
    letters, digits = extract_components(filtered)
    if letters and digits:
        formatted = f"{letters} {digits}"
    else:
        formatted = filtered

    # Create display form with Arabic-Hindi numerals for UI
    display_formatted = convert_western_to_arabic_hindi(formatted)
    
    # Metadata
    metadata = {
        'original_text': text,
        'normalized': normalized != text,
        'filtered': filtered != normalized,
        'valid_format': is_valid,
        'format_description': format_desc,
        'letter_count': len(letters),
        'digit_count': len(digits),
        'canonical_text': formatted  # Western digits for downstream checks
    }
    
    return display_formatted, metadata


def assess_plate_quality(
    text: str,
    ocr_confidence: float,
    format_valid: bool
) -> Tuple[str, float]:
    """
    Assess overall plate recognition quality
    
    Combines multiple factors:
    - OCR confidence
    - Format validity
    - Text length
    
    Returns adjusted confidence and quality assessment.
    
    Args:
        text: Recognized text
        ocr_confidence: Raw OCR confidence
        format_valid: Whether format matches Egyptian patterns
        
    Returns:
        Tuple of (quality_message, adjusted_confidence)
    """
    # Penalize invalid formats
    if not format_valid:
        adjusted_confidence = ocr_confidence * 0.7
        quality = "Non-standard format"
    else:
        adjusted_confidence = ocr_confidence
        quality = "Valid format"
    
    # Penalize very short text (likely incomplete)
    if len(text.replace(' ', '')) < 5:
        adjusted_confidence *= 0.8
        quality += ", possibly incomplete"
    
    return quality, adjusted_confidence


if __name__ == "__main__":
    # Test post-processing
    test_cases = [
        "abc1234",  # Should normalize to Arabic if possible
        "ابج 1234",  # Valid 3 letters + 4 digits
        "اب 1234",   # Valid 2 letters + 4 digits
        "O12I34",    # Should normalize O->0, I->1
    ]
    
    print("Post-processing module test:")
    for test in test_cases:
        processed, meta = apply_format_rules(test)
        print(f"  '{test}' -> '{processed}' (valid: {meta['valid_format']})")
