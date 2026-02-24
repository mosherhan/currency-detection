# 💵 Real-Time Currency Detection System

AI-powered currency note detection using Python, OpenCV, and Deep Learning.

---

## 📌 Overview

This project is a real-time currency detection system that uses a connected camera to identify and classify currency notes using a deep learning model.

### Features

- Detects currency notes from live camera feed
- Classifies denomination (10, 20, 50, 100, etc.)
- Displays confidence score
- Draws bounding boxes
- Supports multiple denominations
- Easily extendable to multiple countries

---

## 🧠 System Architecture

```
Camera Feed
     ↓
Frame Preprocessing
     ↓
Trained Deep Learning Model
     ↓
Prediction (Denomination + Confidence)
     ↓
Bounding Box Overlay
     ↓
GUI Display
```

---

## 📁 Project Structure

```
currency-detection/
│
├── dataset/
│   ├── train/
│   │   ├── 10/
│   │   ├── 20/
│   │   ├── 50/
│   │   └── 100/
│   ├── val/
│   │   ├── 10/
│   │   ├── 20/
│   │   ├── 50/
│   │   └── 100/
│
├── models/
│   └── best_model.pth
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── gui.py
│   ├── preprocessing.py
│   ├── config.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/currency-detection.git
cd currency-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Mac/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

Organize your dataset like this:

```
dataset/
    train/
        10/
        20/
        50/
        100/
    val/
        10/
        20/
        50/
        100/
```

### Recommended Dataset Guidelines

- Minimum 500+ images per class
- Different lighting conditions
- Various angles and orientations
- Multiple backgrounds
- Real camera-captured images
- Partial occlusions

---

## 🏋️ Training the Model

Run:

```bash
python src/train.py
```

Training includes:

- Transfer Learning using MobileNetV2
- Frozen base layers
- Custom classification head
- CrossEntropy Loss
- Early stopping
- Model checkpoint saving
- Validation accuracy tracking

Trained model will be saved in:

```
models/best_model.pth
```

---

## 🎥 Running Real-Time Detection

```bash
python src/inference.py
```

Features:

- Live camera capture
- Frame preprocessing
- Real-time prediction
- Bounding box drawing
- Confidence score display
- Press `q` to quit

---

## 🖥 Running GUI Application

```bash
python src/gui.py
```

GUI Features:

- Start Camera Button
- Stop Camera Button
- Live Video Preview
- Prediction Display
- Confidence Percentage

---

## ⚙️ Model Details

- Base Model: MobileNetV2 (Pretrained on ImageNet)
- Input Size: 224x224
- Output Layer: Softmax
- Loss Function: CrossEntropy
- Optimizer: Adam
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 🚀 Performance Optimization

- Use `model.eval()` for inference
- Resize frames for speed
- Add FPS counter
- Enable GPU acceleration
- Convert model to ONNX for deployment (optional)

---

## 🌍 Extending to New Currencies

To support new denominations:

1. Add new folders inside `dataset/train` and `dataset/val`
2. Update labels inside `config.py`
3. Retrain the model
4. Update GUI label mapping if needed

---

## 🧪 Future Improvements

- YOLOv8 object detection integration
- Multi-note detection in single frame
- Counterfeit detection module
- TensorFlow Lite mobile deployment
- Raspberry Pi / Jetson Nano deployment
- REST API backend
- Web dashboard version

---

## 🛠 Troubleshooting

### Camera Not Opening

Check camera index:

```python
cv2.VideoCapture(0)
```

If it fails, try:

```python
cv2.VideoCapture(1)
```

---

### Low Accuracy

- Increase dataset size
- Add stronger augmentation
- Fine-tune more layers
- Improve lighting diversity

---

### Slow Inference

- Use GPU
- Reduce frame resolution
- Convert model to ONNX

---

## 📦 Example requirements.txt

```
opencv-python
numpy
torch
torchvision
matplotlib
scikit-learn
pillow
tk
```

---

## 📜 License

MIT License

---

## ⭐ Project Vision

This project can evolve into:

- Retail automation system
- Banking security assistant
- Fintech integration tool
- Embedded AI vision product
