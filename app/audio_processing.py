import librosa
import numpy as np
from pathlib import Path
import noisereduce as nr
import pywt

async def process_audio_file(file_path: Path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=5, res_type='kaiser_best')
        
        if len(y) > 2048:
            y = nr.reduce_noise(y=y, sr=sr, stationary=True)
        
        return extract_audio_features(y, sr)
        
    except Exception as e:
        raise Exception(f"Audio processing failed: {str(e)}")

def extract_audio_features(y, sr):
    """Consistent feature extraction matching training"""
    if len(y) == 0:
        raise ValueError("Empty audio file")
    
    # Standard features (must match training exactly)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    return np.concatenate([
        np.mean(mfcc, axis=1),        # 13
        np.std(mfcc, axis=1),         # 13
        np.mean(chroma, axis=1),      # 12
        np.mean(contrast, axis=1),    # 7
        np.mean(tonnetz, axis=1),     # 6
        [librosa.feature.rms(y=y).mean()],        # 1
        [librosa.feature.zero_crossing_rate(y).mean()]  # 1
    ]).reshape(1, -1)  # Total: 53 features