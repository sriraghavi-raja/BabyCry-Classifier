import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.preprocessing import LabelEncoder

class CryClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.expected_features = 53  # Must match feature extraction
        self._load_model()

    def _load_model(self):
        """Load model with feature dimension validation"""
        try:
            model_path = Path(__file__).parent / "model_v1.joblib"
            encoder_path = Path(__file__).parent / "label_encoder_v1.joblib"
            
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            
            # Verify feature dimensions
            if hasattr(self.model, "n_features_in_"):
                self.expected_features = self.model.n_features_in_
            
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def validate_features(self, features: np.ndarray):
        """Ensure correct feature dimensions"""
        if features.shape[1] != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features, got {features.shape[1]}"
            )

    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Prediction with feature validation"""
        try:
            features = np.array(features)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            self.validate_features(features)
            
            proba = self.model.predict_proba(features)[0]
            pred_idx = np.argmax(proba)
            
            return (
                self.label_encoder.classes_[pred_idx],
                float(proba[pred_idx]),
                {cls: float(p) for cls, p in zip(self.label_encoder.classes_, proba)}
            )
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

# Singleton instance
classifier = CryClassifier()

def predict_cry(features: np.ndarray) -> Dict[str, Any]:
    try:
        pred, conf, probs = classifier.predict(features)
        return {
            "prediction": pred,
            "confidence": conf,
            "probabilities": probs,
            "status": "success"
        }
    except Exception as e:
        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "status": "error",
            "message": str(e)
        }