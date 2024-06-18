import os
import zipfile
import requests
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Step 1: Data Preparation
url = 'https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip'
local_zip_file = 'ravdess.zip'

if not os.path.exists(local_zip_file):
    print("Downloading dataset...")
    r = requests.get(url, stream=True)
    with open(local_zip_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Download complete!")

# Unzip the dataset if not already unzipped
if not os.path.exists('ravdess_data'):
    print("Unzipping dataset...")
    with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
        zip_ref.extractall('ravdess_data')
    print("Unzipping complete!")

# Ensure dataset path is correct
dataset_path = 'ravdess_data/Audio_Speech_Actors_01-24'
if not os.path.exists(dataset_path):
    dataset_path = 'ravdess_data'
    print("Updated dataset path to:", dataset_path)

# Step 2: Feature Extraction
def extract_features(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate).T, axis=0)
    features = np.hstack([mfccs, chroma, mel, contrast])
    return features

# Extract features for all audio files
features = []
labels = []

print("Extracting features from audio files...")
for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file)
            emotion = int(file.split('-')[2]) - 1  # Emotions are labeled from 1 to 8
            features.append(extract_features(file_path))
            labels.append(emotion)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)
print("Feature extraction complete!")

# Step 3: Data Preprocessing
# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Model Building and Training
# Use k-fold cross-validation for better validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

val_accuracies = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Reshape for CNN input
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # Build the model
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128)))

    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))  # 8 emotions

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model with callbacks for early stopping and learning rate reduction
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr])
    print("Model training complete!")

    # Store validation accuracy
    val_accuracies.append(history.history['val_accuracy'][-1])

# Calculate average validation accuracy
average_val_accuracy = np.mean(val_accuracies)
print(f"Average Validation Accuracy: {average_val_accuracy}")

# Step 5: Model Evaluation and Saving
# Save the model
model.save('improved_emotion_recognition_model.h5')
print("Model saved as improved_emotion_recognition_model.h5")

# Optional: Streamlit App for Deployment
# Streamlit app
st.title('Emotion Recognition from Speech')
st.write("Upload an audio file to predict the emotion.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    X, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate).T, axis=0)
    feature_vector = np.hstack([mfccs, chroma, mel, contrast]).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)

    # Reshape for CNN input
    feature_vector = feature_vector[..., np.newaxis]

    # Load the model
    model = tf.keras.models.load_model('improved_emotion_recognition_model.h5')

    # Predict the emotion
    prediction = model.predict(feature_vector)
    emotion = np.argmax(prediction)

    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    st.write(f"Predicted emotion: {emotions[emotion]}")
