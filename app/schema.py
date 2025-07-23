from pydantic import BaseModel, Field
from typing import List

class IrisInput(BaseModel):
    """Input schema for Iris prediction"""
    sepal_length: float = Field(..., gt=0, lt=20, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, lt=20, description="Sepal width in cm") 
    petal_length: float = Field(..., gt=0, lt=20, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, lt=20, description="Petal width in cm")
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class IrisOutput(BaseModel):
    """Output schema for Iris prediction"""
    prediction: str
    confidence: float
    probabilities: dict
    input_features: dict

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    version: str
    model_loaded: bool
