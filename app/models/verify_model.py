# Create a test script verify_model.py in your models folder
import pickle
from pathlib import Path

try:
    with open(Path(__file__).parent / "model.pkl", 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")