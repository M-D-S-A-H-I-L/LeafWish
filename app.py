import sys
import logging
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
from pydantic import BaseModel
from typing import List
import uvicorn
import io
from PIL import Image
import traceback

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI(title="Plant Disease Prediction API")

# Enable CORS so frontend (test.html) can access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can replace "*" with ["http://127.0.0.1:5500"] if using VS Code Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and preprocessing objects
try:
    knn_model = joblib.load("knn_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    logging.info("Model, label encoder, and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model or preprocessing objects: {e}")
    sys.exit(1)

# Feature extraction function
def extract_features(image):
    try:
        img = cv2.resize(image, (64, 64))  # Resize to match training
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    except Exception as e:
        logging.warning(f"Error extracting features: {e}")
        return None

# Response model
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: List[float]

# Health check
@app.get("/health")
async def health_check():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Check content type
        content_type = file.content_type
        logging.info(f"Received file: {file.filename}, Content-Type: {content_type}")
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid or missing content type; file must be an image")

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Extract features
        features = extract_features(image_bgr)
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract features from image")

        # Scale features
        features = scaler.transform([features])

        # Predict
        pred = knn_model.predict(features)
        pred_class = label_encoder.inverse_transform(pred)[0]
        pred_proba = knn_model.predict_proba(features)[0]
        confidence = float(np.max(pred_proba))
        all_probabilities = pred_proba.tolist()

        logging.info(f"Prediction: {pred_class}, Confidence: {confidence:.2f}")
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
        }

    except Exception as e:
        logging.error(f"Error during prediction: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    