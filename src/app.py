import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# ----------------------------
# PATH SETUP (works from any location)
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # go up one level from src/
model_path = os.path.join(project_root, 'models', 'emotion_model.h5')

# ----------------------------
# LOAD MODEL AND CASCADE
# ----------------------------
model = load_model(model_path)

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    st.error("Face cascade file not found! Please check OpenCV installation.")
    st.stop()
face_cascade = cv2.CascadeClassifier(cascade_path)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="😊")
st.title("😊 Emotion Detector")
st.write("Upload a clear face image and I'll predict the emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Optional: resize if too small
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        img = cv2.resize(img, (300, 300))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with more sensitive parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    st.write(f"👤 Found {len(faces)} face(s)")  # debug info
    
    if len(faces) == 0:
        st.error("❌ No face detected! Try a clearer, front‑facing photo.")
        st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)
    else:
        # Show original image
        st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)
        
        # Process each face
        for (x, y, w_face, h_face) in faces:
            face = gray[y:y+h_face, x:x+w_face]
            face = cv2.resize(face, (48, 48)) / 255.0
            face = np.expand_dims(face, axis=(0, -1))   # (1,48,48,1)
            
            pred = model.predict(face, verbose=0)
            emotion = EMOTIONS[np.argmax(pred)]
            conf = np.max(pred)
            
            # Draw box on a copy
            img_with_box = img.copy()
            cv2.rectangle(img_with_box, (x, y), (x+w_face, y+h_face), (0, 255, 0), 2)
            cv2.putText(
                img_with_box, f"{emotion} ({conf:.2f})", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
            
            st.image(img_with_box, channels="BGR", caption="Detection Result", use_container_width=True)
            st.success(f"**Predicted Emotion:** {emotion}  (confidence: {conf:.2f})")