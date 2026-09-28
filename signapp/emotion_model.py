# signapp/emotion_model.py
import pickle
import librosa
import numpy as np
from pathlib import Path

# Load the emotion detection model from the correct path in the ml_models directory
def load_emotion_model():
    model_path = Path(__file__).resolve().parent.parent / "ml_models" / "bangla_full_emotion_cpu_last.pkl"  # Correct path to model
    with open(model_path, 'rb') as file:
        emotion_model = pickle.load(file)
    return emotion_model

# Extract Audio Features (MFCC)
def extract_features(audio_file_path):
    audio, sr = librosa.load(audio_file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)  # Extracting mean of MFCCs

# Emotion Prediction
def predict_emotion(audio_features):
    model = load_emotion_model()
    emotion = model.predict([audio_features])  # Assuming the model uses predict method
    return emotion[0]  # Return the predicted emotion label