# Emotion Recognition from Speech

## Overview

This project aims to recognize emotions from speech using machine learning techniques. It utilizes audio data from the RAVDESS dataset and builds a deep learning model to classify emotions such as neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

## Features

- **Audio Feature Extraction**: MFCCs, chroma, mel spectrogram, and spectral contrast features are extracted from audio files to capture emotion-related information.
  
- **Model Architecture**: A deep neural network consisting of convolutional layers followed by bidirectional LSTM layers is employed for emotion classification.

- **Web Application**: Integration with Streamlit allows users to upload audio files and receive real-time emotion predictions.

## Setup

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

Before getting started, ensure you have Python 3.7+ and pip (Python package installer) installed on your system.

#### Required Libraries and Versions

- **numpy** (Version 1.26.4)
- **tensorflow** (Version 2.8.0)
- **librosa** (Version 0.10.2)
- **scikit-learn** (Version 1.5.0)
- **streamlit** (Version 1.9.0)
- **requests** (Version 2.32.3)
- **toml** (Version 0.10.2)
- **pydub** (Version 0.26.7)
- **soundfile** (Version 0.12.1)
- **matplotlib** (Version 3.5.1)

#### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Raufjatoi/Emotion-Recognition-from-Speech.git
   cd Emotion-Recognition-from-Speech
