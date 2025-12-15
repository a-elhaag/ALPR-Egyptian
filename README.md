# Egyptian License Plate Recognition System

A production-grade, multi-stage Automatic License Plate Recognition (ALPR) system specifically designed for Egyptian license plates.

## 🎯 System Overview

This project implements a sophisticated ALPR pipeline that processes vehicle images through multiple stages to detect and recognize Egyptian license plates with high accuracy and interpretability.

**Key Differentiators:**
- **Multi-stage architecture** with interpretability at each step
- **Egyptian plate domain expertise** with format validation and error correction
- **Confidence-aware results** with failure mode diagnosis
- **M3 MacBook Air optimized** for efficient local inference
- **Production-grade code** with modular design and comprehensive documentation

## 🏗️ Architecture

### Pipeline Stages

```
Input Image
    ↓
[1] Preprocessing
    ├─ Resize to standard dimensions
    ├─ Bilateral filtering (edge-preserving denoising)
    └─ CLAHE (lighting normalization)
    ↓
[2] Vehicle Detection (Optional)
    └─ Isolate vehicle region to reduce background clutter
    ↓
[3] Plate Detection
    └─ YOLOv11-based license plate localization
    ↓
[4] Plate Enhancement
    ├─ Upscale to optimal OCR resolution
    ├─ Contrast enhancement (CLAHE)
    ├─ Unsharp masking (edge sharpening)
    └─ Optional binarization
    ↓
[5] OCR
    └─ EasyOCR (Arabic + English)
    ↓
[6] Post-Processing
    ├─ Character normalization (O→0, I→1, etc.)
    ├─ Invalid character removal
    ├─ Egyptian format validation
    └─ Format-specific rules
    ↓
Output: Plate Text + Confidence + Status
```

### Project Structure

```
ALPR-Egyptian/
├── app.py                       # Streamlit web interface
├── config.py                    # Configuration and constants
├── requirements.txt             # Python dependencies
├── pipeline/
│   ├── preprocess.py           # Image preprocessing
│   ├── vehicle_detector.py     # Optional vehicle isolation
│   ├── plate_detector.py       # YOLO plate detection
│   ├── plate_enhancer.py       # Plate-specific enhancement
│   ├── ocr_engine.py           # EasyOCR wrapper
│   ├── postprocess.py          # Egyptian plate rules
│   └── pipeline.py             # Main orchestrator
├── utils/
│   ├── visualization.py        # Drawing utilities
│   └── confidence.py           # Confidence estimation
├── models/
│   └── yolo11m_car_plate_trained.pt  # YOLO model (user-provided)
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- macOS (M3 optimized) or any system with CPU/GPU support

### Setup

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   cd ALPR-Egyptian
   pip install -r requirements.txt
   ```

3. **Add YOLO model:**
   - Place your pretrained YOLOv11 model file in `models/yolo11m_car_plate_trained.pt`
   - If you don't have a model, you can:
     - Train your own using Ultralytics YOLO
     - Use a pretrained license plate detection model
     - Download from a model repository

## 💻 Usage

### Web Interface (Recommended)

Run the Streamlit app:

```bash
streamlit run app.py
```

Then:
1. Open your browser to the provided URL (typically `http://localhost:8501`)
2. Upload an image containing a vehicle with Egyptian license plate
3. Click "Recognize License Plate"
4. View results with step-by-step visualization

### Command Line

Process a single image:

```bash
python pipeline/pipeline.py path/to/image.jpg
```

### Python API

```python
from pipeline.pipeline import create_pipeline
import cv2

# Initialize pipeline
pipeline = create_pipeline()

# Process image
image = cv2.imread("car_image.jpg")
result = pipeline.process_image(image)

# Access results
print(f"Plate: {result['plate_text']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Status: {result['status_message']}")
```

## 🔍 Pipeline Stages Explained

### 1. Preprocessing

**Purpose:** Improve image quality for downstream processing

**Techniques:**
- **Bilateral Filtering:** Reduces noise while preserving edges (crucial for plate boundaries)
- **CLAHE:** Adaptive histogram equalization handles shadows and uneven lighting

**Why it matters:** Real-world images suffer from varying lighting, blur, and noise. Preprocessing significantly improves detection and OCR accuracy.

### 2. Vehicle Detection (Optional)

**Purpose:** Reduce background clutter and false positives

**Approach:** Uses YOLO to detect vehicles and crop to vehicle region

**Fallback:** If no vehicle detected, uses full image

### 3. Plate Detection

**Purpose:** Localize license plate in image

**Model:** YOLOv11 (pretrained on license plate dataset)

**Output:** Bounding box coordinates + confidence score

**Optimization:** Automatically uses MPS (Metal Performance Shaders) on M3 for GPU-like acceleration

### 4. Plate Enhancement

**Purpose:** Maximize OCR accuracy through targeted image processing

**Techniques:**
- **Upscaling:** Small plates are resized to optimal OCR resolution
- **Contrast Enhancement:** CLAHE improves character visibility
- **Unsharp Masking:** Enhances character edges
- **Adaptive Binarization:** Optional black/white conversion for challenging cases

**Why it matters:** Detected plates are often small, blurry, or low-contrast. These enhancements dramatically improve OCR performance.

### 5. OCR

**Purpose:** Extract text from plate image

**Engine:** EasyOCR with Arabic and English support

**Features:**
- Character-level confidence tracking
- Multiple recognition attempts with different preprocessing
- Fallback strategy for low-confidence results

**Why EasyOCR:** Native Arabic support (critical for Egyptian plates) with good accuracy out-of-the-box.

### 6. Post-Processing (🌟 Key Differentiator)

**Purpose:** Apply Egyptian license plate domain knowledge

**Egyptian Plate Rules:**
- **Common formats:**
  - 3 Arabic letters + 4 digits (most common)
  - 2 Arabic letters + 4 digits
  - 3 Arabic letters + 3 digits (older format)

**Processing:**
- **Character Normalization:** Fix common OCR errors (O→0, I→1, l→1, etc.)
- **Invalid Character Removal:** Filter characters that cannot appear on Egyptian plates
- **Format Validation:** Check against known Egyptian plate patterns
- **Confidence Adjustment:** Penalize results that don't match expected formats

**Why it matters:** Domain-specific rules dramatically reduce false positives and correct common OCR mistakes. This is what separates a generic OCR system from a specialized ALPR solution.

## 📊 Confidence System

The system provides transparent confidence estimation:

### Confidence Levels

- **High (≥80%):** Reliable result, ready for use
- **Medium (50-80%):** Plate detected but text may need verification
- **Low (<50%):** Uncertain result, manual review recommended

### Confidence Components

1. **Detection Confidence (40% weight):** How confident YOLO is that it found a plate
2. **OCR Confidence (60% weight):** How confident EasyOCR is in the recognized text
3. **Format Validity:** Penalty if plate doesn't match Egyptian patterns
4. **Text Length:** Penalty if text is suspiciously short

### Status Messages

- ✓ "High confidence result"
- ⚠ "Plate detected but text unclear"
- ✗ "Low confidence detection - plate may not be visible"
- ✗ "Low confidence OCR - plate image quality insufficient"
- ✗ "Low confidence due to image quality (blur, lighting, angle)"

## ⚙️ Configuration

All system parameters are centralized in `config.py`:

- Model paths and parameters
- Image processing constants
- Egyptian plate validation rules
- Confidence thresholds
- Visualization settings

Modify `config.py` to adapt the system to different requirements or regions.

## 🎨 Features

### Interpretability

- **Step-by-step visualization** of all pipeline stages
- **Intermediate outputs** available for debugging
- **Confidence breakdown** by component
- **Failure mode diagnosis** when results are uncertain

### Performance

- **M3 Optimized:** Automatically uses Metal Performance Shaders
- **Processing Time:** 1-3 seconds per image on M3 MacBook Air
- **Efficient Pipeline:** Each stage optimized for speed/accuracy balance

### Robustness

- **Handles challenging conditions:** shadows, blur, poor lighting, angled plates
- **Graceful failure:** Informative error messages when recognition fails
- **Multiple OCR attempts:** Tries different preprocessing if initial attempt has low confidence

## ⚠️ Limitations

### Current Constraints

1. **Model Dependency:** Requires pretrained YOLO model (not included)
2. **Processing Speed:** 1-3 seconds per image (not real-time)
3. **Egyptian Plates Only:** Post-processing rules are Egypt-specific
4. **Image Quality:** Performance degrades with heavily damaged or obscured plates

### Known Issues

- Very small or distant plates may not be detected
- Heavily tilted plates (>45° angle) may have reduced accuracy
- Dirty or damaged plates may produce incomplete results

## 🔮 Future Work

### Potential Improvements

1. **Model Fine-tuning:**
   - Fine-tune YOLO on Egyptian-specific dataset
   - Train custom OCR model for Egyptian plates

2. **Additional Features:**
   - Batch processing mode
   - Video stream support
   - Database integration for plate tracking
   - Multi-plate detection in single image

3. **Performance Optimization:**
   - Model quantization for faster inference
   - Parallel processing for batch operations
   - Caching strategies for repeated processing

4. **Robustness Enhancements:**
   - Perspective correction for angled plates
   - Super-resolution for very small plates
   - Ensemble OCR (multiple engines)

5. **Regional Adaptation:**
   - Configurable rules for other countries
   - Multi-country plate recognition
   - Automatic format detection

## 🤝 Contributions

This project integrates several pretrained components:

- **YOLOv11:** Ultralytics (pretrained object detection)
- **EasyOCR:** JaidedAI (pretrained OCR with Arabic support)

**My Contributions:**
- Multi-stage pipeline architecture
- Preprocessing strategies (bilateral filtering, CLAHE)
- Plate enhancement pipeline (upscaling, sharpening, binarization)
- Egyptian plate domain rules and validation
- Confidence estimation and failure awareness
- Streamlit interface with interpretability
- Comprehensive documentation

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv11
- **JaidedAI** for EasyOCR
- **OpenCV** community for image processing tools
- **Streamlit** for the web framework

---

**Built with academic rigor and production-grade standards.**
