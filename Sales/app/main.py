from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.models import EmbeddingModel
from src.models.classifier import SalesClassifier

app = FastAPI(
    title="Sales Conversion Predictor",
    description="API for predicting sales conversion likelihood using fine-tuned embeddings",
    version="1.0.0"
)

# Load models
generic_embedding_model = EmbeddingModel()
fine_tuned_embedding_model = EmbeddingModel()
fine_tuned_embedding_model.load_model("models/fine_tuned_embeddings")

generic_classifier = SalesClassifier()
fine_tuned_classifier = SalesClassifier()
generic_classifier.load_model("models/generic_classifier.pkl")
fine_tuned_classifier.load_model("models/fine_tuned_classifier.pkl")

class TranscriptRequest(BaseModel):
    transcript: str
    metadata: Optional[Dict] = None

class PredictionResponse(BaseModel):
    generic_prediction: float
    fine_tuned_prediction: float
    generic_confidence: float
    fine_tuned_confidence: float
    feature_importance: Optional[Dict[str, float]] = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Sales Conversion Predictor API",
        "version": "1.0.0",
        "description": "Predict sales conversion likelihood using fine-tuned embeddings"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_conversion(request: TranscriptRequest):
    """
    Predict conversion likelihood for a sales call transcript.
    
    Args:
        request: TranscriptRequest object containing transcript and optional metadata
        
    Returns:
        PredictionResponse object with predictions and confidence scores
    """
    try:
        # Get embeddings
        generic_embedding = generic_embedding_model.get_generic_embeddings([request.transcript])
        fine_tuned_embedding = fine_tuned_embedding_model.get_embeddings([request.transcript])
        
        # Get predictions
        generic_pred, generic_prob = generic_classifier.predict(generic_embedding)
        fine_tuned_pred, fine_tuned_prob = fine_tuned_classifier.predict(fine_tuned_embedding)
        
        # Get feature importance for fine-tuned model
        explanation = fine_tuned_classifier.explain_prediction(
            fine_tuned_embedding,
            feature_names=[f"dim_{i}" for i in range(fine_tuned_embedding.shape[1])]
        )
        
        return PredictionResponse(
            generic_prediction=float(generic_pred[0]),
            fine_tuned_prediction=float(fine_tuned_pred[0]),
            generic_confidence=float(generic_prob[0]),
            fine_tuned_confidence=float(fine_tuned_prob[0]),
            feature_importance=explanation['feature_importance']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 