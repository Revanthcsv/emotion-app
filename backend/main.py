from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import tempfile

# Import from your local modules
from backend.image_models import predict_emotion_image, predict_emotion_webcam
from backend.text_models import TEXT_CLASSIFIER, EMOTION_MAPPING
from backend.recommender import recommend_songs_by_emotion

app = FastAPI(title="Emotion-Based Music Recommender API", version="1.0")

# ---------------------------------------------------
# 🧾 Pydantic Model for Text and Music Recommendation
# ---------------------------------------------------
class TextInput(BaseModel):
    text: str

class MusicInput(BaseModel):
    emotion: str
    uplift: Optional[bool] = False

# ---------------------------------------------------
# 🧠 Text Emotion Detection Endpoint
# ---------------------------------------------------
@app.post("/predict/text")
async def predict_text_emotion(input: TextInput):
    """
    Predict emotion from text using the BERT-Emotions-Classifier.
    Maps 11 fine-grained emotions to 5 core facial emotions.
    """
    try:
        # Run NLP pipeline
        results = TEXT_CLASSIFIER(input.text)
        raw_emotion = results[0]['label'].lower()
        mapped_emotion = EMOTION_MAPPING.get(raw_emotion, 'Neutral')
        return {"detected_text_emotion": raw_emotion, "mapped_emotion": mapped_emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text emotion prediction failed: {str(e)}")

# ---------------------------------------------------
# 🖼️ Face Emotion Detection (Uploaded Image)
# ---------------------------------------------------
@app.post("/predict/face-image")
async def predict_face_from_image(file: UploadFile = File(...)):
    """
    Detect emotion from a face image file.
    The image is read temporarily and processed by the CNN model.
    """
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        emotion = predict_emotion_image(tmp_path)
        return {"emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face image prediction failed: {str(e)}")

# ---------------------------------------------------
# 🎥 Face Emotion Detection (Webcam)
# ---------------------------------------------------
@app.get("/predict/face-webcam")
def predict_face_from_webcam():
    """
    Detect emotion in real-time from the user's webcam.
    (Requires running locally with access to webcam)
    """
    try:
        emotion = predict_emotion_webcam()
        return {"emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webcam emotion prediction failed: {str(e)}")

# ---------------------------------------------------
# 🎵 Music Recommendation Endpoint
# ---------------------------------------------------
@app.post("/recommend")
async def recommend_music(input_data: MusicInput):
    """
    Recommend songs for a given emotion.
    Supports both mood-matching and uplifting recommendation modes.
    """
    try:
        recs = recommend_songs_by_emotion(
            emotion=input_data.emotion,
            uplift=input_data.uplift,
            n=5
        )
        return recs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Music recommendation failed: {str(e)}")

# ---------------------------------------------------
# 🏠 Root Endpoint
# ---------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to the Emotion-Based Music Recommender API 🎵",
        "available_endpoints": {
            "/predict/text": "POST - Detect emotion from text",
            "/predict/face-image": "POST - Detect emotion from uploaded image",
            "/predict/face-webcam": "GET - Detect emotion from webcam (local)",
            "/recommend": "POST - Get music recommendations by emotion"
        }
    }
