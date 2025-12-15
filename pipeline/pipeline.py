"""
Main ALPR Pipeline Orchestrator
Coordinates all processing stages and provides unified interface

This is the heart of the system - it ties together all components
and manages the flow of data through the pipeline.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import time

# Import pipeline stages
from pipeline.preprocess import preprocess_pipeline
from pipeline.vehicle_detector import isolate_vehicle
from pipeline.plate_detector import PlateDetector
from pipeline.plate_enhancer import enhance_pipeline, enhance_pipeline_with_binary
from pipeline.ocr_engine import OCREngine, calculate_overall_confidence
from pipeline.postprocess import apply_format_rules, assess_plate_quality

import config


class ALPRPipeline:
    """
    Complete ALPR pipeline orchestrator
    
    Manages the entire processing flow from raw image to recognized plate text.
    Provides interpretability through stage-by-stage outputs.
    """
    
    def __init__(self, use_vehicle_detection: bool = False):
        """
        Initialize pipeline
        
        Args:
            use_vehicle_detection: Whether to use optional vehicle detection stage
        """
        self.use_vehicle_detection = use_vehicle_detection
        
        # Initialize components
        self.plate_detector = PlateDetector()
        self.ocr_engine = OCREngine()
        
        # Track initialization status
        self.initialized = False
        
    def initialize(self) -> bool:
        """
        Load all models and initialize components
        
        Returns:
            True if successful, False otherwise
        """
        print("Initializing ALPR pipeline...")
        
        # Load plate detector
        if not self.plate_detector.load_model():
            print("✗ Failed to load plate detector")
            return False
        
        # Initialize OCR engine
        if not self.ocr_engine.initialize_reader():
            print("✗ Failed to initialize OCR engine")
            return False
        
        self.initialized = True
        print("✓ Pipeline initialized successfully")
        return True
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Process a single image through the complete pipeline
        
        This is the main entry point for the system.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing:
            - success: bool
            - plate_text: str (recognized text)
            - confidence: float (overall confidence)
            - status_message: str (human-readable status)
            - processing_time: float (seconds)
            - stages: dict (intermediate outputs for visualization)
            - metadata: dict (detailed processing information)
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # Initialize result structure
        result = {
            'success': False,
            'plate_text': '',
            'confidence': 0.0,
            'status_message': '',
            'processing_time': 0.0,
            'stages': {},
            'metadata': {}
        }
        
        try:
            # ================================================================
            # STAGE 1: Preprocessing
            # ================================================================
            preprocessed, preprocess_meta = preprocess_pipeline(image)
            result['stages']['original'] = image.copy()
            result['stages']['preprocessed'] = preprocessed
            result['metadata']['preprocessing'] = preprocess_meta
            
            # ================================================================
            # STAGE 2: Vehicle Detection (Optional)
            # ================================================================
            vehicle_crop, vehicle_meta = isolate_vehicle(
                preprocessed,
                use_vehicle_detection=self.use_vehicle_detection
            )
            result['stages']['vehicle_crop'] = vehicle_crop
            result['metadata']['vehicle_detection'] = vehicle_meta
            
            # ================================================================
            # STAGE 3: Plate Detection
            # ================================================================
            plate_detection = self.plate_detector.get_best_plate(vehicle_crop)
            
            if plate_detection is None:
                # No plate detected
                result['status_message'] = config.STATUS_MESSAGES['no_plate_detected']
                result['confidence'] = 0.0
                result['processing_time'] = time.time() - start_time
                return result
            
            # Store detection results
            result['stages']['plate_bbox'] = plate_detection['bbox']
            result['stages']['plate_crop'] = plate_detection['crop']
            result['metadata']['plate_detection'] = {
                'confidence': plate_detection['confidence'],
                'bbox': plate_detection['bbox']
            }
            
            detection_confidence = plate_detection['confidence']
            
            # ================================================================
            # STAGE 4: Plate Enhancement
            # ================================================================
            enhanced_plate, binary_plate, enhance_meta = enhance_pipeline_with_binary(
                plate_detection['crop']
            )
            result['stages']['enhanced_plate'] = enhanced_plate
            result['stages']['binary_plate'] = binary_plate
            result['metadata']['enhancement'] = enhance_meta
            
            # ================================================================
            # STAGE 5: OCR
            # ================================================================
            raw_text, ocr_confidence, ocr_details = self.ocr_engine.recognize_with_fallback(
                enhanced_plate,
                binary_plate
            )
            
            result['metadata']['ocr'] = {
                'raw_text': raw_text,
                'confidence': ocr_confidence,
                'details': ocr_details
            }
            
            # ================================================================
            # STAGE 6: Post-Processing
            # ================================================================
            processed_text, postprocess_meta = apply_format_rules(raw_text)
            result['metadata']['postprocessing'] = postprocess_meta
            
            # ================================================================
            # STAGE 7: Confidence Assessment
            # ================================================================
            overall_confidence = calculate_overall_confidence(
                detection_confidence,
                ocr_confidence
            )
            
            quality_msg, adjusted_confidence = assess_plate_quality(
                processed_text,
                overall_confidence,
                postprocess_meta['valid_format']
            )
            
            # ================================================================
            # Generate Status Message
            # ================================================================
            if adjusted_confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
                status = config.STATUS_MESSAGES['high_confidence']
            elif adjusted_confidence >= config.MEDIUM_CONFIDENCE_THRESHOLD:
                status = config.STATUS_MESSAGES['medium_confidence']
            else:
                # Diagnose low confidence reason
                if detection_confidence < 0.5:
                    status = config.STATUS_MESSAGES['low_confidence_detection']
                elif ocr_confidence < 0.5:
                    status = config.STATUS_MESSAGES['low_confidence_ocr']
                else:
                    status = config.STATUS_MESSAGES['low_confidence_quality']
            
            # ================================================================
            # Finalize Result
            # ================================================================
            result['success'] = True
            result['plate_text'] = processed_text
            result['confidence'] = adjusted_confidence
            result['status_message'] = status
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            # Handle any processing errors
            result['status_message'] = f"{config.STATUS_MESSAGES['processing_error']}: {str(e)}"
            result['processing_time'] = time.time() - start_time
            result['metadata']['error'] = str(e)
            return result
    
    def get_stage_outputs(self, result: Dict) -> Dict:
        """
        Extract stage outputs for visualization
        
        Args:
            result: Result dictionary from process_image()
            
        Returns:
            Dictionary of stage name -> image
        """
        return result.get('stages', {})


def create_pipeline(use_vehicle_detection: bool = False) -> Optional[ALPRPipeline]:
    """
    Factory function to create and initialize pipeline
    
    Args:
        use_vehicle_detection: Whether to enable vehicle detection stage
        
    Returns:
        Initialized pipeline or None if initialization fails
    """
    pipeline = ALPRPipeline(use_vehicle_detection=use_vehicle_detection)
    
    if pipeline.initialize():
        return pipeline
    else:
        return None


if __name__ == "__main__":
    # Test pipeline
    import sys
    
    print("Creating ALPR pipeline...")
    pipeline = create_pipeline(use_vehicle_detection=False)
    
    if pipeline is None:
        print("Failed to create pipeline")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        # Test with provided image
        test_image = cv2.imread(sys.argv[1])
        if test_image is not None:
            print(f"\nProcessing image: {sys.argv[1]}")
            result = pipeline.process_image(test_image)
            
            print(f"\n{'='*60}")
            print(f"Result: {result['status_message']}")
            print(f"Plate Text: {result['plate_text']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            print(f"{'='*60}")
        else:
            print(f"Could not load image: {sys.argv[1]}")
    else:
        print("\nPipeline ready. Usage: python pipeline.py <image_path>")
