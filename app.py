"""
Egyptian License Plate Recognition System
Streamlit Web Interface

A production-grade ALPR system with step-by-step visualization
and confidence-aware results.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Import pipeline
from pipeline.pipeline import create_pipeline
from utils.visualization import draw_bbox, create_confidence_bar, get_confidence_color
import config


# Page configuration
st.set_page_config(
    page_title="Egyptian ALPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stage-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .result-text {
        font-size: 3rem;
        font-weight: bold;
        letter-spacing: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #00ff00;
    }
    .confidence-medium {
        color: #ffa500;
    }
    .confidence-low {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load and cache the ALPR pipeline"""
    with st.spinner("Initializing ALPR system... This may take a moment."):
        pipeline = create_pipeline(use_vehicle_detection=False)
    return pipeline


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🚗 Egyptian License Plate Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production-Grade Multi-Stage ALPR with Interpretability</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system recognizes Egyptian license plates through a sophisticated multi-stage pipeline:
        
        **Pipeline Stages:**
        1. 🔧 Preprocessing (denoising, lighting normalization)
        2. 🎯 Plate Detection (YOLOv11)
        3. ✨ Plate Enhancement (contrast, sharpening)
        4. 📝 OCR (Arabic + English)
        5. ✅ Post-Processing (Egyptian plate rules)
        
        **Key Features:**
        - Confidence-aware results
        - Step-by-step visualization
        - Domain-specific validation
        """)
        
        st.header("⚙️ Settings")
        show_all_stages = st.checkbox("Show All Pipeline Stages", value=True)
        show_metadata = st.checkbox("Show Technical Metadata", value=False)
    
    # Initialize pipeline
    try:
        pipeline = load_pipeline()
        
        if pipeline is None:
            st.error("❌ Failed to initialize ALPR pipeline. Please check that the YOLO model file exists.")
            st.info(f"Expected model path: `{config.YOLO_MODEL_PATH}`")
            return
        
        st.success("✅ ALPR pipeline initialized successfully")
        
    except Exception as e:
        st.error(f"❌ Error initializing pipeline: {str(e)}")
        st.info("Please ensure all dependencies are installed and the YOLO model file is available.")
        return
    
    # File uploader
    st.markdown("---")
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image containing a vehicle with Egyptian license plate",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Load image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Display original image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="stage-header">📷 Original Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        
        # Process button
        with col2:
            st.markdown('<div class="stage-header">⚡ Processing</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Recognize License Plate", type="primary", use_container_width=True):
                # Process image
                with st.spinner("Processing image through pipeline..."):
                    result = pipeline.process_image(image_np)
                
                # Display results
                st.markdown("---")
                
                if result['success']:
                    # Main result display
                    confidence_level = 'high' if result['confidence'] >= config.HIGH_CONFIDENCE_THRESHOLD else 'medium' if result['confidence'] >= config.MEDIUM_CONFIDENCE_THRESHOLD else 'low'
                    
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="font-size: 1.5rem;">Recognized Plate Number</div>
                        <div class="result-text">{result['plate_text']}</div>
                        <div style="font-size: 1.2rem; margin-top: 1rem;">
                            {result['status_message']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence display
                    st.markdown('<div class="stage-header">📊 Confidence Analysis</div>', unsafe_allow_html=True)
                    
                    conf_col1, conf_col2, conf_col3 = st.columns(3)
                    
                    with conf_col1:
                        st.metric("Overall Confidence", f"{result['confidence']:.1%}")
                    
                    with conf_col2:
                        detection_conf = result['metadata']['plate_detection']['confidence']
                        st.metric("Detection Confidence", f"{detection_conf:.1%}")
                    
                    with conf_col3:
                        ocr_conf = result['metadata']['ocr']['confidence']
                        st.metric("OCR Confidence", f"{ocr_conf:.1%}")
                    
                    # Confidence bar
                    conf_bar = create_confidence_bar(result['confidence'], width=800, height=50)
                    conf_bar_rgb = cv2.cvtColor(conf_bar, cv2.COLOR_BGR2RGB)
                    st.image(conf_bar_rgb, use_container_width=True)
                    
                    # Processing time
                    st.info(f"⏱️ Processing completed in {result['processing_time']:.2f} seconds")
                    
                    # Pipeline stages visualization
                    if show_all_stages:
                        st.markdown("---")
                        st.markdown('<div class="stage-header">🔍 Pipeline Stages</div>', unsafe_allow_html=True)
                        
                        stages = result['stages']
                        
                        # Stage 1: Preprocessing
                        with st.expander("1️⃣ Preprocessing", expanded=False):
                            st.markdown("**Applied:** Denoising, Lighting Normalization (CLAHE)")
                            stage_col1, stage_col2 = st.columns(2)
                            with stage_col1:
                                st.markdown("**Original**")
                                st.image(cv2.cvtColor(stages['original'], cv2.COLOR_BGR2RGB), use_container_width=True)
                            with stage_col2:
                                st.markdown("**Preprocessed**")
                                st.image(cv2.cvtColor(stages['preprocessed'], cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Stage 2: Plate Detection
                        with st.expander("2️⃣ Plate Detection (YOLOv11)", expanded=True):
                            st.markdown(f"**Detection Confidence:** {detection_conf:.1%}")
                            
                            # Draw bounding box on image
                            bbox = stages['plate_bbox']
                            img_with_bbox = draw_bbox(
                                stages['vehicle_crop'],
                                bbox,
                                "License Plate",
                                detection_conf
                            )
                            
                            det_col1, det_col2 = st.columns(2)
                            with det_col1:
                                st.markdown("**Detection Result**")
                                st.image(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB), use_container_width=True)
                            with det_col2:
                                st.markdown("**Cropped Plate**")
                                st.image(cv2.cvtColor(stages['plate_crop'], cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Stage 3: Plate Enhancement
                        with st.expander("3️⃣ Plate Enhancement", expanded=False):
                            st.markdown("**Applied:** Resize, Contrast Enhancement, Sharpening")
                            enh_col1, enh_col2 = st.columns(2)
                            with enh_col1:
                                st.markdown("**Original Crop**")
                                st.image(cv2.cvtColor(stages['plate_crop'], cv2.COLOR_BGR2RGB), use_container_width=True)
                            with enh_col2:
                                st.markdown("**Enhanced**")
                                st.image(cv2.cvtColor(stages['enhanced_plate'], cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Stage 4: OCR
                        with st.expander("4️⃣ OCR (EasyOCR - Arabic + English)", expanded=False):
                            st.markdown(f"**OCR Confidence:** {ocr_conf:.1%}")
                            st.markdown(f"**Raw Text:** `{result['metadata']['ocr']['raw_text']}`")
                            st.markdown(f"**Processed Text:** `{result['plate_text']}`")
                        
                        # Stage 5: Post-Processing
                        with st.expander("5️⃣ Post-Processing (Egyptian Plate Rules)", expanded=False):
                            post_meta = result['metadata']['postprocessing']
                            st.markdown(f"**Format Valid:** {'✅ Yes' if post_meta['valid_format'] else '❌ No'}")
                            if post_meta['format_description']:
                                st.markdown(f"**Format:** {post_meta['format_description']}")
                            st.markdown(f"**Letters:** {post_meta['letter_count']}")
                            st.markdown(f"**Digits:** {post_meta['digit_count']}")
                            st.markdown(f"**Normalization Applied:** {'Yes' if post_meta['normalized'] else 'No'}")
                    
                    # Technical metadata
                    if show_metadata:
                        st.markdown("---")
                        st.markdown('<div class="stage-header">🔧 Technical Metadata</div>', unsafe_allow_html=True)
                        st.json(result['metadata'])
                
                else:
                    # Processing failed
                    st.error(f"❌ {result['status_message']}")
                    st.info("Try uploading a different image with a clearer view of the license plate.")
    
    else:
        # No file uploaded
        st.info("👆 Please upload an image to begin")
        
        # Example instructions
        with st.expander("📋 Usage Instructions"):
            st.markdown("""
            ### How to Use
            
            1. **Upload an Image**: Click the upload button and select an image containing a vehicle with an Egyptian license plate
            2. **Process**: Click the "Recognize License Plate" button
            3. **View Results**: The system will display:
               - Recognized plate number
               - Confidence scores
               - Step-by-step pipeline visualization
            
            ### Best Results
            
            For optimal recognition accuracy:
            - ✅ Clear, well-lit images
            - ✅ Plate is visible and not obscured
            - ✅ Minimal motion blur
            - ✅ Frontal or near-frontal view
            
            ### System Limitations
            
            - ⚠️ Requires YOLO model file to be present
            - ⚠️ Processing time: 1-3 seconds per image (M3 optimized)
            - ⚠️ Works best with standard Egyptian plate formats
            - ⚠️ May struggle with heavily damaged or obscured plates
            """)


if __name__ == "__main__":
    main()
