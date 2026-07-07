import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split          # <-- this fixes your error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------------
# 1. LOAD DATA WITH ROBUST PATH
# ----------------------------
# Get the directory of this script (train.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (emotion_detector)
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, 'data', 'fer2013.csv')

print("Loading data from:", csv_path)
df = pd.read_csv(csv_path)

X = []
y = []
for _, row in df.iterrows():
    pixels = list(map(int, row['pixels'].split()))
    X.append(np.array(pixels).reshape(48, 48, 1))
    y.append(row['emotion'])

X = np.array(X) / 255.0
y = np.array(y)

# Split into train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# ----------------------------
# 2. BUILD MODEL
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------------
# 3. TRAIN & SAVE
# ----------------------------
# Create models folder if it doesn't exist
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
model_path = os.path.join(project_root, 'models', 'emotion_model.h5')

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True)
]

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

print(f"✅ Training complete! Model saved to {model_path}")