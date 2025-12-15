# YOLO Model Placeholder

This directory should contain your pretrained YOLOv11 license plate detection model.

## Required File

- **Filename:** `yolo11m_car_plate_trained.pt`
- **Type:** PyTorch model file (.pt)
- **Purpose:** License plate detection

## How to Obtain a Model

### Option 1: Use a Pretrained Model

Download a pretrained license plate detection model from:
- Ultralytics Model Zoo
- Roboflow Universe
- GitHub repositories with ALPR models

### Option 2: Train Your Own

1. Collect/download a license plate dataset
2. Annotate images with bounding boxes (if not already annotated)
3. Train using Ultralytics YOLO:

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolo11m.pt')

# Train on your dataset
results = model.train(
    data='path/to/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Save trained model
model.save('yolo11m_car_plate_trained.pt')
```

### Option 3: Download from Roboflow

Example using a public license plate dataset:

```bash
# Install roboflow
pip install roboflow

# Download dataset (example)
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace().project('license-plate-detection')
dataset = project.version(1).download('yolov8')
"
```

## Model Requirements

- **Input:** RGB images (any size, will be resized to 640x640)
- **Output:** Bounding boxes for license plates
- **Classes:** Should detect license plates (class name doesn't matter)

## Testing Your Model

Once you have the model file in this directory, test it:

```bash
cd ..
python pipeline/plate_detector.py path/to/test/image.jpg
```

## Note

This system is designed to work with any YOLOv11 (or YOLOv8) license plate detection model. The model architecture is handled automatically by Ultralytics.
