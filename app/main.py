from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
from datetime import datetime
from pathlib import Path
import shutil

# Local imports
from .audio_processing import process_audio_file
from .models.ml_model import predict_cry
from .schemas import PredictionResponse, CryLogCreate
from .database import SessionLocal, CryModel

app = FastAPI(
    title="Enhanced Baby Cry Classifier API",
    version="2.0.0",
    description="Improved version with better handling of class imbalance"
)

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:19006",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static upload directory with cleanup
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file with unique filename
        file_suffix = Path(file.filename).suffix
        unique_filename = f"{datetime.now().timestamp()}{file_suffix}"
        file_location = UPLOAD_DIR / unique_filename
        
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        # Process audio and make prediction
        features = await process_audio_file(file_location)
        result = predict_cry(features)

        if result["status"] != "success":
            raise HTTPException(
                status_code=400, 
                detail=f"Prediction failed: {result['message']}"
            )

        # Only log confident predictions
        if result["confidence"] > 0.5:
            db = SessionLocal()
            try:
                db_prediction = CryModel(
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    timestamp=datetime.utcnow(),
                    audio_file_path=str(file_location)
                )
                db.add(db_prediction)
                db.commit()
            except Exception as db_error:
                db.rollback()
                raise HTTPException(
                    status_code=500, 
                    detail=f"Database error: {str(db_error)}"
                )
            finally:
                db.close()

        return result

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/history", response_model=List[CryLogCreate])
async def get_history(limit: int = 10):
    db = SessionLocal()
    try:
        history = db.query(CryModel)\
            .order_by(CryModel.timestamp.desc())\
            .limit(limit)\
            .all()
        return history
    finally:
        db.close()