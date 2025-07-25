# main.py
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter


# --- OpenTelemetry Setup ---

trace.set_tracer_provider(TracerProvider())

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(CloudTraceSpanExporter())
)

tracer = trace.get_tracer(__name__)
# --- End of OpenTelemetry Setup ---


# Initialize FastAPI app
app = FastAPI(title="Iris Species Predictor API")

FastAPIInstrumentor.instrument_app(app)


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
    
    with tracer.start_as_current_span("model_prediction") as span:
        data = pd.DataFrame([[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width
        ]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        prediction = model.predict(data)
        probability = model.predict_proba(data).max()
        
        span.set_attribute("prediction.species", prediction[0])
        span.set_attribute("prediction.probability", float(probability))
       
    return {
        "predicted_species": prediction[0],
        "prediction_probability": round(probability, 4)
    }