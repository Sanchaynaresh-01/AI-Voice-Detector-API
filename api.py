from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import numpy as np
import joblib
import io

from extract_features import extract_features

app = FastAPI(title="AI Voice Detection API")

API_KEY = "23254-24478"
MODEL_PATH = "models/ai_voice_detector.pkl"

# Load model ONCE
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {
        "message": "AI Voice Detection API is live",
        "docs": "/docs",
        "endpoint": "/predict"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        features = extract_features(audio_bytes, 16000)


        # 🚨 sanity check
        if len(features) != 55:
            raise HTTPException(
                status_code=500,
                detail=f"Feature mismatch: got {len(features)}, expected 55"
            )

        features = features.reshape(1, -1)

        proba = model.predict_proba(features)[0]
        prediction = int(np.argmax(proba))

        return {
            "prediction": "AI" if prediction == 1 else "Human",
            "confidence": round(float(np.max(proba)), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

