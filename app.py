"""
Egyptian License Plate Recognition System
Streamlit Web Interface

A production-grade ALPR system with step-by-step visualization
and confidence-aware results.
"""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import config
# Import pipeline
from pipeline.pipeline import create_pipeline
from utils.visualization import (create_confidence_bar, draw_bbox,
                                 get_confidence_color)

# Page configuration
st.set_page_config(
    page_title="Egyptian ALPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean modern UI skin
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --bg: #0f172a;          /* Dark blue background */
    --card: #1e293b;        /* Slightly lighter blue for cards */
    --panel: #334155;       /* Panel color */
    --border: rgba(255,255,255,0.1);
    --text: #ffffff;        /* White text */
    --muted: #94a3b8;       /* Muted text color */
    --accent: #60a5fa;      /* Light blue accent */
    --accent-hover: #93c5fd;
    --radius: 16px;
    --radius-sm: 10px;
    --shadow: 0 4px 24px rgba(0,0,0,0.4);
}
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
}
.block-container { padding: 2rem 2.5rem 3rem 2.5rem !important; max-width: 1200px; }
.full-width-tabs {
    width: 100vw;
    margin-left: calc(50% - 50vw);
    padding: 0 2.5rem;
    box-sizing: border-box;
}
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
}
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}
.drop-box {
    border: 2px dashed rgba(255,255,255,0.15);
    border-radius: var(--radius);
    background: var(--panel);
    padding: 2.5rem 2rem;
    text-align: center;
    transition: all 0.25s ease;
    cursor: pointer;
}
.drop-box:hover {
    border-color: var(--accent);
    background: rgba(59,130,246,0.05);
}
.drop-box-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.6;
}
.drop-box-title {
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.25rem;
}
.drop-box-hint {
    color: var(--muted);
    font-size: 0.9rem;
}
.result-card {
    background: linear-gradient(145deg, #1e3a5f 0%, #0f172a 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
    box-shadow: var(--shadow);
    margin: 1.5rem 0;
}
.result-label {
    color: var(--text);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.result-text {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 0.4rem;
    margin: 0.5rem 0;
    color: var(--text);
}
.result-status {
    color: var(--text);
    font-size: 0.95rem;
    margin-top: 0.75rem;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.25rem 0;
}
@media (max-width: 768px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
}
.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 1.25rem;
    text-align: center;
    color: var(--text);
}
.metric-label {
    color: #cbd5e1;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 1.5rem 0 0.75rem 0;
}
.divider {
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
}
.tips-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
}
.tips-card h3 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
}
.tips-card ul {
    color: var(--muted);
    padding-left: 1.25rem;
    margin: 0;
    line-height: 1.8;
}
.fade-in { animation: fadeIn 0.5s ease forwards; opacity: 0; }
@keyframes fadeIn { to { opacity: 1; } }

/* Style native Streamlit file uploader to fill width and match theme */
.stFileUploader > label {
    color: var(--text);
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.35rem;
}
.stFileUploader > div {
    border: 2px dashed rgba(255,255,255,0.18);
    border-radius: var(--radius);
    background: var(--panel);
    padding: 1.75rem 1.5rem;
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
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### Control Panel")
        st.markdown("Fine-tune what you see while keeping the heavy lifting in the main canvas.")
        show_all_stages = st.checkbox("Show All Pipeline Stages", value=True)
        show_metadata = st.checkbox("Show Technical Metadata", value=False)
        st.markdown("---")
        st.markdown("**Pipeline**")
        st.markdown("- YOLOv11 for plates\n- PaddleOCR (Arabic)\n- Domain rules for Egypt\n- Confidence fusion")
    
    # Initialize pipeline
    try:
        pipeline = load_pipeline()
        
        if pipeline is None:
            st.error("Failed to initialize ALPR pipeline. Please check that the YOLO model file exists.")
            st.info(f"Expected model path: `{config.YOLO_MODEL_PATH}`")
            return
        
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        st.info("Please ensure all dependencies are installed and the YOLO model file is available.")
        return

    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>Egyptian License Plate Recognition</h1>
        <p>Upload a vehicle image to detect and read the plate</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader (styled via CSS above)
    uploaded_file = st.file_uploader(
        "Upload a vehicle image to detect and read the plate",
        type=['jpg', 'jpeg', 'png'],
        help="Drag and drop or click to browse (JPG, JPEG, PNG)",
        label_visibility="visible"
    )
    
    if uploaded_file is not None:
        # Load image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Workspace
        col1, col2 = st.columns([1.05, 1])
        with col1:
            st.markdown('<div class="section-title">Input Preview</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-title">Recognition</div>', unsafe_allow_html=True)
            if st.button("Run Recognition", type="primary", use_container_width=True):
                with st.spinner("Processing image through pipeline..."):
                    result = pipeline.process_image(image_np)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                if result['success']:
                    detection_conf = result['metadata']['plate_detection']['confidence']
                    ocr_conf = result['metadata']['ocr']['confidence']
                    status_clean = result['status_message'].replace('✓', '').replace('⚠', '').replace('✗', '').strip()
                    
                    st.markdown(f"""
                    <div class="result-card fade-in">
                        <div class="result-label">Recognized Plate</div>
                        <div class="result-text">{result['plate_text']}</div>
                        <div class="result-status">{status_clean}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence metrics
                    st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Overall</div>
                            <div class="metric-value">{result['confidence']*100:.0f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Detection</div>
                            <div class="metric-value">{detection_conf*100:.0f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">OCR</div>
                            <div class="metric-value">{ocr_conf*100:.0f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Latency</div>
                            <div class="metric-value">{result['processing_time']:.2f}s</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Confidence bar
                    conf_bar = create_confidence_bar(result['confidence'], width=900, height=42)
                    conf_bar_rgb = cv2.cvtColor(conf_bar, cv2.COLOR_BGR2RGB)
                    st.image(conf_bar_rgb, use_container_width=True, caption="Composite confidence")
                    
                    # Pipeline stages visualization
                    if show_all_stages:
                        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                        st.markdown('<div class="section-title">Pipeline Stages</div>', unsafe_allow_html=True)
                        stages = result['stages']

                        st.markdown('<div class="full-width-tabs">', unsafe_allow_html=True)
                        tabs = st.tabs(["Preprocess", "Plate Detection", "Enhancement", "OCR", "Post-Process"])

                        with tabs[0]:
                            st.markdown("**Denoise + Grayscale + CLAHE**")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("Input")
                                st.image(cv2.cvtColor(stages['original'], cv2.COLOR_BGR2RGB), use_container_width=True)
                            with c2:
                                st.markdown("Preprocessed")
                                st.image(cv2.cvtColor(stages['preprocessed'], cv2.COLOR_BGR2RGB), use_container_width=True)

                        with tabs[1]:
                            st.markdown(f"**YOLOv11 detection** · {detection_conf:.1%}")
                            bbox = stages['plate_bbox']
                            img_with_bbox = draw_bbox(
                                stages['vehicle_crop'],
                                bbox,
                                "License Plate",
                                detection_conf
                            )
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Detection overlay")
                            with c2:
                                st.image(cv2.cvtColor(stages['plate_crop'], cv2.COLOR_BGR2RGB), use_container_width=True, caption="Plate crop")

                        with tabs[2]:
                            st.markdown("**Resize · Deskew · Denoise · Contrast · Sharpen**")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(cv2.cvtColor(stages['plate_crop'], cv2.COLOR_BGR2RGB), use_container_width=True, caption="Before")
                            with c2:
                                st.image(cv2.cvtColor(stages['enhanced_plate'], cv2.COLOR_BGR2RGB), use_container_width=True, caption="Enhanced")
                            st.image(cv2.cvtColor(stages['binary_plate'], cv2.COLOR_BGR2RGB), use_container_width=True, caption="Binary variant")

                        with tabs[3]:
                            st.markdown(f"**PaddleOCR Arabic** · {ocr_conf:.1%}")
                            st.markdown(f"`Raw:` `{result['metadata']['ocr']['raw_text']}`")
                            st.markdown(f"`Processed:` `{result['plate_text']}`")

                        with tabs[4]:
                            post_meta = result['metadata']['postprocessing']
                            st.markdown(f"**Format Valid:** {'✅' if post_meta['valid_format'] else '❌'}")
                            if post_meta['format_description']:
                                st.markdown(f"**Format:** {post_meta['format_description']}")
                            st.markdown(f"**Letters:** {post_meta['letter_count']} · **Digits:** {post_meta['digit_count']}")
                            st.markdown(f"**Normalized:** {'Yes' if post_meta['normalized'] else 'No'}")
                            if 'canonical_text' in post_meta:
                                st.markdown(f"**Canonical (Western digits):** `{post_meta['canonical_text']}`")

                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Technical metadata
                    if show_metadata:
                        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                        st.markdown('<div class="section-title">Technical Metadata</div>', unsafe_allow_html=True)
                        st.json(result['metadata'])
                
                else:
                    st.error(result['status_message'].replace('✓', '').replace('⚠', '').replace('✗', '').strip())
                    st.info("Try uploading a different image with a clearer view of the license plate.")
    
    else:
        st.markdown("""
        <div class="tips-card fade-in">
            <h3>Tips for best results</h3>
            <ul>
                <li>Plate fully visible, not occluded</li>
                <li>Minimal motion blur</li>
                <li>Near-frontal angle</li>
                <li>Even lighting without glare</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
