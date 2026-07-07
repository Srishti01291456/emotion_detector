import cv2
import numpy as np
from tensorflow.keras.models import load_model
'''import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'models', 'emotion_model.h5')
model = load_model(model_path)'''

# ----------------------------
# 1. LOAD MODEL & CASCADE
# ----------------------------
model = load_model(r'C:\Users\kapta\OneDrive\Desktop\emotion_detector\models\emotion_model.h5')
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)   # (48,48,1)
    face = np.expand_dims(face, axis=0)    # (1,48,48,1)
    return face

# ----------------------------
# 2. START CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48)
    )

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        input_face = preprocess_face(face_roi)

        pred = model.predict(input_face, verbose=0)
        emotion_idx = np.argmax(pred)
        emotion = EMOTIONS[emotion_idx]
        confidence = np.max(pred)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{emotion} ({confidence:.2f})", (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    cv2.imshow('Emotion Detector - Press q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()