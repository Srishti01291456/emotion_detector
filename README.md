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

## 📷 Demo
<img width="560" height="482" alt="WhatsApp Image 2026-07-08 at 02 19 54" src="https://github.com/user-attachments/assets/b8d98ed6-dc83-4614-a9a4-eac66d0588fd" />

<img width="631" height="506" alt="WhatsApp Image 2026-07-08 at 02 19 27" src="https://github.com/user-attachments/assets/f061fccf-249f-42b5-ad9c-feb205460d34" />

<img width="1102" height="649" alt="WhatsApp Image 2026-07-08 at 02 23 40" src="https://github.com/user-attachments/assets/b5d3863a-29a0-42ac-ac39-ea6ed2dc82ae" />

<img width="938" height="606" alt="WhatsApp Image 2026-07-08 at 02 24 08" src="https://github.com/user-attachments/assets/3970af01-c63c-4350-9f23-7ca38b58441a" />

<img width="1040" height="257" alt="WhatsApp Image 2026-07-08 at 02 24 37" src="https://github.com/user-attachments/assets/cede098e-52ba-410c-8309-6b4753265b0b" />
