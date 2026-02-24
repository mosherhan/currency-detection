This project is a real-time currency detection system that uses a connected camera to identify and classify currency notes using a deep learning model.

The system:

Detects currency notes from live camera feed

Classifies denomination

Displays confidence score

Draws bounding boxes

Supports multiple denominations

Can be extended to multiple countries

Built using:

Python 3.10+

OpenCV

TensorFlow / PyTorch

MobileNetV2 (Transfer Learning)

Tkinter / PyQt (GUI)

🧠 System Architecture
Camera Feed → Frame Preprocessing → Trained Model → Prediction
        ↓
 Bounding Box + Confidence Overlay → GUI Display
📁 Project Structure
currency-detection/
│
├── dataset/
│   ├── train/
│   │   ├── 10/
│   │   ├── 20/
│   │   ├── 50/
│   │   └── 100/
│   ├── val/
│
├── models/
│   └── best_model.pth / best_model.h5
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
🔧 Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/currency-detection.git
cd currency-detection
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
📊 Dataset Preparation

Organize your dataset like this:

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

Each folder should contain images of that denomination.

Recommended:

500+ images per class

Different lighting

Different angles

Background variations

Partial occlusion examples

🏋️ Training the Model
python src/train.py

Training features:

Transfer Learning (MobileNetV2)

Frozen base layers

Custom classification head

Early stopping

Model checkpoint saving

Validation accuracy tracking

Training graph generation

Model will be saved in:

models/best_model.pth
🎥 Running Real-Time Detection
python src/inference.py

Features:

Live camera capture

Frame preprocessing

Real-time prediction

Bounding box drawing

Confidence score display

Press q to quit

🖥 GUI Application

To launch the graphical interface:

python src/gui.py

GUI includes:

Start Camera Button

Stop Camera Button

Live Preview

Prediction Label

Confidence Display

⚙️ Model Details

Base Model: MobileNetV2 (Pretrained on ImageNet)

Input Size: 224x224

Output Layer: Softmax

Loss Function: CrossEntropy

Optimizer: Adam

Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

🚀 Performance Optimization

Uses model.eval() for inference

Frame resizing for speed

FPS counter display

GPU acceleration supported

Optional ONNX export

🌍 Extending to Other Currencies

To support new currency:

Add new denomination folders in dataset

Add label in config.py

Retrain model

Update GUI label mapping

🧪 Future Improvements

YOLOv8 object detection integration

Fake currency detection module

Multi-note detection in single frame

Mobile deployment (TensorFlow Lite)

Raspberry Pi / Jetson Nano deployment

Web dashboard version

REST API backend

📉 Common Issues
Camera Not Opening

Check device index in OpenCV:

cv2.VideoCapture(0)
Low Accuracy

Increase dataset size

Add augmentation

Improve lighting conditions

Fine-tune more layers

Slow Inference

Use GPU

Reduce frame size

Convert model to ONNX

📦 Requirements

Example requirements.txt:

opencv-python
numpy
torch
torchvision
matplotlib
scikit-learn
pillow
tk
📜 License

This project is open-source and available under the MIT License.

🤝 Contribution

Pull requests are welcome.
For major changes, open an issue first.

⭐ Acknowledgements

ImageNet pretrained models

OpenCV community

PyTorch / TensorFlow ecosystem
