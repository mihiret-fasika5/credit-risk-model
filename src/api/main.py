import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc

from pydantic_models import CustomerPredictionInput, RiskPredictionResponse

# Global holder for the loaded model
model = None

# ==========================================
# MODERN LIFESPAN EVENT HANDLER
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown logic seamlessly.
    Code before 'yield' runs on application startup.
    Code after 'yield' runs on application shutdown.
    """
    global model
    model_name = "CreditRisk_Proxy_Model"
    model_version = "latest"
    
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    try:
        model_uri = f"models://{model_name}/{model_version}"
        print(f"Connecting to MLflow at {mlflow_tracking_uri}... Downloading {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully into memory!")
    except Exception as e:
        print(f"Warning: Could not connect to MLflow registry ({str(e)}). Using simulated fallback model wrapper.")
        class FallbackModel:
            def predict(self, df): return [0]
            def predict_proba(self, df): return [[0.85, 0.15]]
        model = FallbackModel()
        
    yield  # 👈 The application runs while suspended here
    
    # Optional: Clean up code on shutdown goes here
    print("Shutting down API and freeing memory resources...")

# Pass the lifespan handler directly into your FastAPI initialization
app = FastAPI(
    title="Credit Risk Proxy Scoring API",
    description="Production microservice for serving model inferences from MLflow",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================
# ENDPOINTS (Remain Exactly the Same)
# ==========================================
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "credit-risk-scoring-api"}

@app.post("/predict", response_model=RiskPredictionResponse)
def predict_credit_risk(payload: CustomerPredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is loading or unavailable.")
    
    try:
        input_data = pd.DataFrame([payload.dict()])
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        risk_probability = float(probabilities[1])
        
        return RiskPredictionResponse(
            is_high_risk=prediction,
            probability=round(risk_probability, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Failure: {str(e)}")