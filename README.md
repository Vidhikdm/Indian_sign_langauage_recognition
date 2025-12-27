# 🇮🇳 Indian Sign Language Recognition System
A real-time Indian Sign Language (ISL) recognition system built with MediaPipe, OpenCV, and Machine Learning. This project recognizes ISL alphabets and common words/phrases with 90%+ accuracy, enabling communication for the deaf and hard-of-hearing community.


# Project Motivation
According to the 2011 Census of India, over 5 million people in India have hearing disabilities. Indian Sign Language is their primary mode of communication, but most people don't understand ISL, creating a significant communication barrier.
This project aims to bridge that gap by providing:

1. Real-time ISL recognition on commodity hardware
2. Educational tool for learning ISL
3. Accessibility technology for ISL users


# Features

**Core Functionality**

Real-time Recognition: 20-30 FPS ISL sign detection from webcam
High Accuracy: 90-95% classification accuracy
ISL Alphabets: A-Z letter recognition (26 classes)
Common Words: Hello, Thank You, Please, etc. (expandable)
Sentence Building: Construct sentences by combining signs
Confidence Display: Shows prediction confidence for each sign


**Technical Features**

Features: 57-dimensional feature vector (translation/scale/rotation invariant)
Temporal Smoothing: Reduces jitter for stable predictions
Production-Ready: Clean, modular, well-documented code
Dual Datasets: Supports both Kaggle datasets and custom collection




**hand-gesture-recognition/**
│
├── **notebooks/**                     # Development notebooks
│   ├── 01-05_*.ipynb                  # Basic gesture recognition
│   ├── 06_ISL_data_preparation.ipynb  # ISL dataset setup
│   ├── 07_ISL_model_training.ipynb    # ISL model training
│   └── 08_ISL_real_time.ipynb         # Real-time ISL testing
│
├── **src/**                           # Core modules
│   ├── hand_tracker.py                # MediaPipe wrapper
│   ├── feature_extractor.py           # Feature engineering
│   ├── gesture_classifier.py          # ML classifier
│   └── utils.py                       # Helper functions
│
├── **app/**                           # Applications
│   ├── gesture_recognition_app.py     # Basic gestures
│   └── isl_recognition_app.py       **# ISL recognition**
│
├── **data/**                          # Datasets
│   ├── raw/
│   │   ├── gesture_dataset.npz        # Basic gestures
│   │   └── isl_custom_dataset.npz     # ISL dataset
│   └── processed/
│
├── **models/**                        # Trained models
│   ├── gesture_classifier.pkl         # Basic model
│   ├── isl_gesture_classifier.pkl   **# ISL model** 
│   ├── isl_scaler.pkl
│   └── isl_model_metadata.json
│
├── assets/                            # Documentation
└── requirements.txt




# Download ISL Dataset (Option A: Kaggle)
**Download dataset**

Dataset link : https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset?select=ISL_Dataset


# Collect Your Own ISL Data (Option B: Custom)

jupyter notebook notebooks/06_ISL_data_preparation.ipynb

Instructions:

Run the ISL Data Collector cell
Type gesture name (e.g., "A", "hello")
Press SPACE to collect 50+ samples
Repeat for all ISL signs you want to recognize
Save dataset



# ISL Signs Supported

# Default Configuration (Alphabets)

| Sign | Description           
|------|----------------------
| A-Z  | ISL alphabet letters  

# Extended Configuration (Words)

| Sign      | 
|-----------|
| hello     | 
| thank_you | 
| please    |
| sorry     | 
| yes       | 
| no        | 
| help      | 
| family    | 
| friend    |
| water     | 
| food      |

---

# Performance Metrics

# Model Performance

| Metric          | Value      |
|-----------------|------------|
| Accuracy        | 90-95%     |
| F1-Score        | 0.92+      |
| Inference Speed | 20-30 FPS  |
| Latency         | <50 ms     |




# Resources
**ISL Learning**

ISLRTC Resources : http://www.islrtc.nic.in/

**Technical References**

MediaPipe Hands : https://ai.google.dev/edge/mediapipe/solutions/guide
ISL Research Papers : https://scholar.google.com/scholar?q=indian+sign+language+recognition


License
MIT License - Free for educational and commercial use



**This project aims to make communication more accessible for the 5+ million deaf and hard-of-hearing individuals in India. Every contribution, star, or share helps raise awareness about ISL and accessibility technology.
Together, we can break communication barriers!**



