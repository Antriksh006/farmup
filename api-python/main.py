import math
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, validator
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
from utils.apicalls import build_single_row

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "API is working!"})

# -------- Load model and encoders with error handling --------
try:
    model = load_model("utils/crop_model.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    season_enc = pickle.load(open("season_enc.pkl", "rb"))
    label_enc = pickle.load(open("label_enc.pkl", "rb"))
    logger.info("All models and encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading models/encoders: {e}")
    model = scaler = season_enc = label_enc = None

# -------- Request model --------
class Location(BaseModel):
    latitude: float
    longitude: float
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

# -------- Prediction route --------
@app.post("/predict")
def predict_crop(loc: Location):
    try:
        # Check if models are loaded
        if any(x is None for x in [model, scaler, season_enc, label_enc]):
            raise HTTPException(status_code=500, detail="Models not properly loaded")
        
        logger.info(f"Predicting for location: {loc.latitude}, {loc.longitude}")
        
        # Fetch real-time data
        df = build_single_row(loc.latitude, loc.longitude)
        logger.info(f"Data fetched: {df.to_dict('records')[0]}")
        
        # Check if data is valid
        if df is None or df.empty:
            raise HTTPException(status_code=500, detail="Failed to fetch weather/soil data")
        
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Missing values found in data")
            # Handle missing values - you might want to use default values or raise an error
            df = df.fillna(0)  # Simple fill - you might want a more sophisticated approach
        
        # Preprocess numeric features
        numeric_cols = ["temperature", "humidity", "ph", "water availability"]
        X_num = df[numeric_cols].values.astype(np.float32)
        
        # Check if we have the right number of features for scaling
        if X_num.shape[1] != 4:
            raise HTTPException(status_code=500, detail=f"Expected 4 numeric features, got {X_num.shape[1]}")
        
        # Scale the features
        X_num_scaled = scaler.transform(X_num)
        
        # Encode season
        season_data = df[["season"]].values
        X_season = season_enc.transform(season_data)
        
        if hasattr(X_season, "toarray"):  # if OneHotEncoder with sparse output
            X_season = X_season.toarray()
        X_season = X_season.astype(np.float32)
        
        # Combine features
        X = np.hstack([X_num_scaled, X_season])
        logger.info(f"Final feature shape: {X.shape}")
        
        # Predict
        pred_probs = model.predict(X, verbose=0)[0]
        
        # Validate predictions
        if not isinstance(pred_probs, np.ndarray) or len(pred_probs) == 0:
            raise HTTPException(status_code=500, detail="Invalid model prediction")
        
        # Get top 3 predictions
        top_idx = pred_probs.argsort()[-3:][::-1]
        top_crops = label_enc.inverse_transform(top_idx)
        top_scores = pred_probs[top_idx].tolist()
        
        # Create result with proper error handling for scores
        result = []
        for crop, score in zip(top_crops, top_scores):
            # Ensure score is a valid number
            if not math.isfinite(score):
                score = 0.0
            result.append({
                "crop": str(crop),  # Ensure crop name is string
                "probability": float(score)
            })
        
        logger.info(f"Prediction successful: {result}")
        return JSONResponse(content={"predictions": result})
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)