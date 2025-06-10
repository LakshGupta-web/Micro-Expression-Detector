
# 🎭 Facial Micro-expression Recognition System

This project is built in **two phases** to detect and classify human emotions from facial expressions:

- **Phase 1**: Real-time Face Detection with Image Capture and Gallery GUI
- **Phase 2**: Emotion Classification using CNN (Trained on FER2013 + CK+)

## 📌 Project Overview

| Phase | Description |
|-------|-------------|
| **1** | A real-time face detection system using OpenCV's DNN face detector. It features a Tkinter-based GUI to capture, crop, and store images. |
| **2** | A CNN model is trained to recognize 7 micro-expressions by combining FER2013 and CK+ datasets. Real-time webcam-based emotion recognition is also supported. |

## 📁 Folder Structure

```
archive/
├── fer2013/
│   └── fer2013/
│       └── Training/
│           └── [emotion]/
├── ck+/
│   └── ck+/
│       └── [emotion]/
├── MicroExpression_Split/
│   ├── training/
│   └── test/
samples/
├── test1.jpg
├── test2.jpg
```

## ⚙️ Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/LakshGupta-web/Micro-Expression-Detector.git
cd micro-expression-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` should contain:

```
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
Pillow
```

## 🧠 Phase 1: Real-time Face Detection GUI

### Features

- Uses OpenCV DNN for accurate face detection
- GUI built with Tkinter
- Real-time camera feed, image capture
- Auto-cropping of face regions
- Captured images saved into a gallery

### How to Run

```bash
python face_detector_gui.py
```

## 🤖 Phase 2: Micro-expression Recognition

### Dataset Merging & Splitting

You can combine **FER2013** and **CK+** datasets and split them like so:

```bash
python split_datasets.py
```

It creates:
```
archive/MicroExpression_Split/
├── training/
└── test/
```

### Model Training

Trains a CNN model from scratch using the merged dataset.

```bash
python train_model.py
```

- Input size: 64x64
- CNN with Conv2D, MaxPooling, Dropout, Dense layers
- EarlyStopping & ReduceLROnPlateau for stability
- Class weighting for imbalance handling

### Model Saving

After training, the model is saved as:

```
micro_expression_model.h5
```

### Predict on Static Images

```bash
python predict_images.py
```

- Input images from `samples/`
- Preprocessing: Resize, grayscale, normalize
- Outputs predicted emotion + confidence

### Real-Time Webcam Inference

```bash
python realtime_webcam.py
```

- Uses OpenCV webcam capture
- Detects face → crops → predicts → overlays emotion label
- Press **'q'** to exit

## 🎯 Supported Emotions

- Anger
- Contempt
- Disgust
- Fear
- Happy
- Sad
- Surprised

## 🔍 Sample Results

| Image | Predicted Emotion |
|-------|-------------------|
| `test1.jpg` | Happy (92.4%) |
| `test2.jpg` | Fear (81.3%) |

## 📈 Accuracy and Notes

- Final CNN reached ~80–85% validation accuracy (varies per run).
- Better performance possible using MobileNetV2 with fine-tuning.
- Balanced data across FER2013 and CK+ is key.

## 🙋‍♂️ Author

**Laksh Gupta**  
🗓️ June 2025 

## 💡 Future Work

- Add support for expression timeline tracking
- Upgrade model to MobileNetV2 or EfficientNet
- Export model to TensorFlow Lite for mobile deployment
