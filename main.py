# main.py
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Import OpenTelemetry libraries
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter


# --- OpenTelemetry Setup ---
# 1. Set up a TracerProvider
trace.set_tracer_provider(TracerProvider())

# 2. Configure the application to export traces to Google Cloud
# The CloudTraceSpanExporter will automatically use the permissions of the
# service account that the GKE pod is running as.
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(CloudTraceSpanExporter())
)

# 3. Get a "tracer" instance
tracer = trace.get_tracer(__name__)
# --- End of OpenTelemetry Setup ---


# Initialize FastAPI app
app = FastAPI(title="Iris Species Predictor API")

# Instrument the FastAPI application
# This will automatically create traces for all incoming requests
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
    
    # --- Custom Span for Model Prediction ---
    # We create a custom "span" to measure just the model inference time.
    # This will show up separately in our traces.
    with tracer.start_as_current_span("model_prediction") as span:
        data = pd.DataFrame([[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width
        ]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        prediction = model.predict(data)
        probability = model.predict_proba(data).max()
        
        # You can add attributes to your span for more context
        span.set_attribute("prediction.species", prediction[0])
        span.set_attribute("prediction.probability", float(probability))
    # --- End of Custom Span ---
    
    return {
        "predicted_species": prediction[0],
        "prediction_probability": round(probability, 4)
    }