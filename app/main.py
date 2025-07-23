from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.schema import IrisInput, IrisOutput, HealthResponse
import joblib
import numpy as np
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Iris Species Classification API",
    description="ML API for predicting Iris species using sepal and petal measurements",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
species_names = ["setosa", "versicolor", "virginica"]

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        model_path = "model/iris_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Iris Species Classification API",
        "description": "This API predicts the species of Iris flowers based on sepal and petal measurements",
        "endpoints": {
            "POST /predict": "Make predictions",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=IrisOutput)
async def predict_species(
    input_data: IrisInput, 
    background_tasks: BackgroundTasks
):
    """Predict Iris species based on input features"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Prepare input data
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        
        # Make prediction
        prediction_idx = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get species name
        predicted_species = species_names[prediction_idx]
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            species: float(prob) 
            for species, prob in zip(species_names, probabilities)
        }
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction, 
            input_data.dict(), 
            predicted_species, 
            confidence
        )
        
        return IrisOutput(
            prediction=predicted_species,
            confidence=confidence,
            probabilities=prob_dict,
            input_features=input_data.dict()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

def log_prediction(input_features: dict, prediction: str, confidence: float):
    """Background task to log predictions"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_features,
        "prediction": prediction,
        "confidence": confidence
    }
    logger.info(f"Prediction logged: {log_entry}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
