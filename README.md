# 😊 Real-Time Facial Emotion Detector

An end-to-end deep learning and computer vision application that detects human emotions from facial expressions in real-time (via webcam) or through uploaded images (via a Streamlit web app). Built completely from scratch using Python, TensorFlow/Keras, OpenCV, and Streamlit.

## 🚀 Features
- **Live Webcam Detection**: Uses OpenCV to capture real-time video, detect faces, and overlay the predicted emotion and confidence score.
- **Image Upload Web App**: A clean, interactive Streamlit interface where users can upload a photo and instantly receive an emotion prediction.
- **Custom Convolutional Neural Network (CNN)**: Trained from scratch on the FER2013 dataset (35,000+ grayscale 48x48 face images) without relying on pre-trained weights.
- **7 Emotion Classes**: Predicts **Angry** 😠, **Disgust** 🤢, **Fear** 😨, **Happy** 😊, **Sad** 😢, **Surprise** 😲, and **Neutral** 😐.

## 🛠️ Tech Stack
- **Python** (3.9+) 
- **TensorFlow / Keras** (Deep Learning framework)
- **OpenCV** (Face detection using Haar Cascades and image processing)
- **Streamlit** (Front-end web interface)
- **NumPy / Pandas** (Data manipulation and preprocessing)
- **Scikit-learn** (Data splitting and evaluation)

## 📁 Project Structure
```text
emotion_detector/
├── data/
│   └── fer2013.csv              # FER2013 dataset
├── models/
│   └── emotion_model.h5         # Trained CNN model weights
├── notebooks/
│   └── exploration.ipynb        # EDA and model testing notebook
├── src/                         # Source code
│   ├── train.py                 # Model training script
│   ├── realtime.py              # Real-time webcam detection script
│   └── app.py                   # Streamlit web application
├── README.md                    
└── requirements.txt             # All project dependencies

