# main.py
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Iris Species Predictor API")

# Define the request body structure using Pydantic
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the trained model from the file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    """A welcome message for the API root."""
    return {"message": "Welcome to the Iris Prediction API! Visit /docs for more info."}

@app.post("/predict")
def predict_species(iris_features: IrisRequest):
    """Predicts the Iris species based on input features."""
    # Create a pandas DataFrame from the request data
    data = pd.DataFrame([[
        iris_features.sepal_length,
        iris_features.sepal_width,
        iris_features.petal_length,
        iris_features.petal_width
    ]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    # Make a prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data).max()
    
    return {
        "predicted_species": prediction[0],
        "prediction_probability": round(probability, 4)
    }