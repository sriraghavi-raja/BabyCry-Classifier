from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Dict, Optional, List

class PredictionResponse(BaseModel):
    prediction: str = Field(..., example="hungry")
    confidence: float = Field(..., ge=0, le=1, example=0.85)
    probabilities: Dict[str, float] = Field(
        ...,
        example={"hungry": 0.85, "belly_pain": 0.10, "burping": 0.05}
    )

    @validator('probabilities')
    def check_probabilities_sum(cls, v):
        if abs(sum(v.values()) - 1.0) > 0.01:  # 1% tolerance
            raise ValueError("Probabilities must sum to 1")
        return v

class CryLogBase(BaseModel):
    prediction: str = Field(..., example="hungry")
    confidence: float = Field(..., ge=0, le=1, example=0.85)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    audio_file_path: Optional[str] = Field(None, example="uploads/cry_123.wav")

class CryLogCreate(CryLogBase):
    pass

class CryLog(CryLogBase):
    id: int = Field(..., example=1)
    
    class Config:
        from_attributes = True  # Changed from orm_mode in Pydantic V2
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }