# Quick Start Guide - Egyptian ALPR System

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
cd /Users/anas/Projects/Computer-Vision/ALPR-Egyptian
pip install -r requirements.txt
```

**Note:** First run will download EasyOCR language models (~200MB total)

---

### Step 2: Add YOLO Model

You need a pretrained YOLOv11 license plate detection model.

**Quick Option - Download Pretrained:**
```bash
# Example using a public model (you'll need to find an actual source)
# Place the .pt file in: models/yolo11m_car_plate_trained.pt
```

**See [models/README.md](file:///Users/anas/Projects/Computer-Vision/ALPR-Egyptian/models/README.md) for detailed instructions**

---

### Step 3: Run the System

**Launch Web Interface:**
```bash
streamlit run app.py
```

Then open browser to `http://localhost:8501`

**Or test from command line:**
```bash
python pipeline/pipeline.py path/to/car_image.jpg
```

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `config.py` | All system parameters |
| `pipeline/pipeline.py` | Main orchestrator |
| `pipeline/preprocess.py` | Image preprocessing |
| `pipeline/plate_detector.py` | YOLO detection |
| `pipeline/plate_enhancer.py` | Plate enhancement |
| `pipeline/ocr_engine.py` | EasyOCR wrapper |
| `pipeline/postprocess.py` | Egyptian plate rules |
| `utils/visualization.py` | Drawing utilities |
| `utils/confidence.py` | Confidence estimation |

---

## 🎯 System Pipeline

```
Image → Preprocess → Detect Plate → Enhance → OCR → Post-process → Result
         (CLAHE)      (YOLOv11)    (Sharpen)  (Arabic)  (Validate)
```

---

## ⚙️ Configuration

Edit `config.py` to customize:
- Confidence thresholds
- Image processing parameters
- Egyptian plate rules
- OCR languages

---

## 🔍 Testing Individual Modules

```bash
# Test preprocessing
python pipeline/preprocess.py image.jpg

# Test plate detection
python pipeline/plate_detector.py image.jpg

# Test enhancement
python pipeline/plate_enhancer.py plate_crop.jpg

# Test post-processing
python pipeline/postprocess.py
```

---

## 📊 Understanding Results

### Confidence Levels
- **High (≥80%):** ✅ Reliable result
- **Medium (50-80%):** ⚠️ May need verification
- **Low (<50%):** ❌ Manual review recommended

### Status Messages
- "High confidence result" - Ready to use
- "Plate detected but text unclear" - OCR struggled
- "Low confidence detection" - Plate may not be visible
- "No license plate detected" - No plate found

---

## 🐛 Troubleshooting

**"Model not found"**
- Add YOLO model to `models/yolo11m_car_plate_trained.pt`
- See models/README.md for instructions

**"EasyOCR downloading models"**
- Normal on first run
- Downloads Arabic + English models (~200MB)
- Only happens once

**"Low confidence results"**
- Try better quality images
- Ensure plate is clearly visible
- Check lighting conditions

---

## 📚 Documentation

- **Full README:** [README.md](file:///Users/anas/Projects/Computer-Vision/ALPR-Egyptian/README.md)
- **Implementation Details:** See walkthrough.md in artifacts
- **Model Setup:** [models/README.md](file:///Users/anas/Projects/Computer-Vision/ALPR-Egyptian/models/README.md)

---

## 🎨 Key Features

✅ Multi-stage pipeline with interpretability  
✅ Egyptian plate domain expertise  
✅ M3 MacBook Air optimized (MPS acceleration)  
✅ Confidence-aware results  
✅ Professional Streamlit UI  
✅ Step-by-step visualization  
✅ Comprehensive error handling  

---

**Ready to recognize Egyptian license plates! 🚗✨**
