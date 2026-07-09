# Palmprint Locker Authentication System

AI-powered biometric locker authentication system using palmprint recognition, realtime server communication, and embedded hardware control.

## 🎥 Demo

▶️ [Watch Demo Video](https://drive.google.com/file/d/1XvmbyGqb-8U7NaBYBQBwI1W84gFA-sKH/view?usp=sharing)

---

# Overview

This project is a complete smart locker authentication system based on palmprint biometrics.

The system combines:

* Deep learning palmprint verification
* Realtime inference server
* WebSocket communication
* Embedded hardware control
* Automatic locker unlocking

The main objective is to provide a secure, contactless, and realtime authentication solution for smart locker systems.

---

# System Architecture

```text
ESP32-CAM
     │
     ▼
Palm Image Capture
     │
     ▼
Flask Inference Server
     │
     ├── Image Preprocessing
     ├── Embedding Extraction
     ├── Cosine Similarity Matching
     └── Authentication Decision
     │
     ▼
ESP32-C3 Controller
     │
     ▼
Servo Motor
     │
     ▼
Locker Unlock
```

---

# Features

* Palmprint biometric authentication
* Deep learning embedding extraction
* Cosine similarity verification
* Threshold-based authentication
* Open-set biometric verification
* Realtime WebSocket communication
* Multi-process inference pipeline
* Embedded locker control
* Training and evaluation framework
* Automatic locker unlocking

---

# Repository Structure

```text
├── 📁 flask_server
│   ├── 📁 common
│   ├── 📁 server
│   └── 📁 worker
│
├── 📁 hardware
│   ├── 📁 esp32_c3mini
│   └── 📁 palmcam
│
├── 📁 src
│   ├── 📁 datasets
│   ├── 📁 model
│   ├── 📁 training
│   └── 📁 transforms
│
├── 📁 models
├── 📁 data
├── 📁 results
├── 📁 runs
├── 📁 scripts
├── 📁 storage
│
├── ⚙️ requirements.txt
└── 📝 README.md
```

---

# AI Training Pipeline

The training framework supports:

* ArcFace Loss
* Triplet Loss
* Cosine Similarity Verification
* Data Augmentation
* Mixed Precision Training

## Training

```bash
python .\src\training\train.py \
    --loss arcface \
    --train_path datasets \
    --val_path datasets \
    --model_name palmnet_arcface \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4
```

---

# Evaluation

The system is evaluated using biometric verification metrics:

* Verification Accuracy
* FAR (False Acceptance Rate)
* FRR (False Rejection Rate)
* ROC-AUC
* EER (Equal Error Rate)

## FAR / FRR Curve

![Metric](docs/images/metric.png)

The intersection point between FAR and FRR represents the Equal Error Rate (EER).

## Confusion Matrix

![Confusion](docs/images/confusion.png)

## Similarity Distribution

![Distance](docs/images/distance.png)

## Training Loss

![Loss](docs/images/loss.png)

## Evaluation Example

```bash
python .\src\training\compare.py \
    --model1 models/best_1.pth \
    --model2 models/best_2.pth \
    --val_path dataset
```

---

# Inference Server

The backend server is implemented using Flask and Flask-Sock for realtime communication.

The server architecture includes:

* REST API handling
* WebSocket communication
* Multiprocessing inference workers
* Queue-based task processing
* Background synchronization threads

## Start Server

```bash
cd flask_server

python .\server\main.py
```

---

# Hardware

The hardware subsystem controls the locker unlocking mechanism.

## Components

* ESP32-CAM
* ESP32-C3 Mini
* Servo motor
* TFT display
* Power supply

## Hardware Workflow

1. Capture palm image
2. Send image to inference server
3. Receive authentication result
4. Activate servo motor
5. Unlock locker
6. Auto lock after timeout

---

# Communication Flow

```text
ESP32-CAM
    │
    ▼
Capture Palm Image
    │
    ▼
Flask Server
    │
    ├── Preprocessing
    ├── Feature Extraction
    ├── Similarity Matching
    └── Verification
    │
    ▼
Authentication Result
    │
    ▼
ESP32-C3
    │
    ▼
Servo Unlock
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/tienson05/palm-locker.git

cd palm-locker
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

Datasets used in this project:

* Tongji Palmprint Dataset (training)
* IITD Palmprint Dataset (validation and testing)

---

# Results

## Triplet Loss Model

```json
{
    "accuracy": 0.8787,
    "eer": 0.1205,
    "far": 0.1212,
    "frr": 0.1198,
    "roc_auc": 0.9555
}
```

## ArcFace Model

```json
{
    "accuracy": 0.9778,
    "eer": 0.0213,
    "far": 0.0221,
    "frr": 0.0206,
    "roc_auc": 0.9971
}
```

The ArcFace-based model significantly outperforms the Triplet Loss model in verification performance and achieves a much lower Equal Error Rate.

---

# Technologies Used

## Backend & Server

* Python
* Flask
* Flask-Sock

## AI & Computer Vision

* PyTorch
* OpenCV
* NumPy

## Concurrency & Processing

* threading
* multiprocessing
* Queue-based worker architecture

## Communication

* REST API
* WebSocket realtime communication

## Embedded Systems

* ESP32-CAM
* ESP32-C3 Mini
* Servo motor