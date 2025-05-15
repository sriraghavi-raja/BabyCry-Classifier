import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from collections import Counter

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "donateacry_corpus_cleaned_and_updated_data"
CLASSES = ["hungry", "belly_pain", "burping", "discomfort", "tired"]
SAMPLE_RATE = 22050
DURATION = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

def extract_features(file_path):
    """Feature extraction function must match production exactly"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Standardize length
        target_length = SAMPLE_RATE * DURATION
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        # Must match audio_processing.py exactly
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        return np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1),
            [librosa.feature.rms(y=y).mean()],
            [librosa.feature.zero_crossing_rate(y).mean()]
        ])

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Load dataset
features = []
labels = []

print("Loading dataset...")
for class_name in CLASSES:
    class_path = DATA_DIR / class_name
    if not class_path.exists():
        raise FileNotFoundError(f"Missing class directory: {class_path}")
    
    for file in class_path.glob("*.*"):
        if file.suffix.lower() in (".wav", ".mp3", ".ogg"):
            feat = extract_features(file)
            if feat is not None:
                features.append(feat)
                labels.append(class_name)

# Verify feature dimensions
feature_lengths = [len(f) for f in features]
if len(set(feature_lengths)) != 1:
    raise ValueError(f"Inconsistent feature lengths: {set(feature_lengths)}")

print(f"Loaded {len(features)} samples")
print("Class distribution:", Counter(labels))

# Prepare data
X = np.array(features)
y = np.array(labels)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Save artifacts
MODEL_DIR = BASE_DIR / "app" / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_DIR / "model_v1.joblib")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder_v1.joblib")

print(f"Model saved with {X.shape[1]} features")