import os
import numpy as np
import streamlit as st
import tensorflow as tf
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_model('improved_emotion_recognition_model.h5')

# Function to extract features from audio file
def extract_features(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate).T, axis=0)
    feature_vector = np.hstack([mfccs, chroma, mel, contrast]).reshape(1, -1)
    return feature_vector

# Normalize the feature vector
scaler = StandardScaler()

# Streamlit app
st.title('Emotion Recognition from Speech')
st.write("Upload an audio file to predict the emotion.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Extract features from uploaded file
    X = extract_features(uploaded_file)
    X = scaler.fit_transform(X)

    # Reshape for CNN input
    X = X[..., np.newaxis]

    # Predict the emotion
    prediction = model.predict(X)
    emotion = np.argmax(prediction)

    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    st.write(f"Predicted emotion: {emotions[emotion]}")
