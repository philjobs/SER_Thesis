
#%pip install datasets
#%pip install pydub
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import tensorflow as tf

import librosa
import librosa.display
import IPython.display as ipd

from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.style.use('ggplot')

# Define the path to the TESS dataset
TESS = '/TESS Toronto emotional speech set data/'

# Ensure the directory exists
if not os.path.exists(TESS):
    raise FileNotFoundError(f"The specified TESS dataset path does not exist: {TESS}")

# Load TESS dataset files
tess_files = glob(os.path.join(TESS, '**', '*.wav'), recursive=True)
if not tess_files:
    raise FileNotFoundError("No .wav files found in the specified TESS directory.")

print(f"Found {len(tess_files)} audio files in the TESS dataset.")

# Extract labels from file names
labels = [os.path.basename(file).split('_')[2][:-4] for file in tess_files]  # Assumes filenames have emotion labels

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Feature extraction function
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Extract features from all audio files
print("Extracting features...")
features = np.array([extract_features(file) for file in tess_files])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

# Build a simple neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Example prediction
example_idx = 0
example_feature = features[example_idx].reshape(1, -1)
example_label = labels[example_idx]

prediction = model.predict(example_feature)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

print(f"Actual Label: {example_label}, Predicted Label: {predicted_class[0]}")
