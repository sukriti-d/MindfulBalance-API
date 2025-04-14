# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import predict_mood, predict_fatigue

app = FastAPI()

class MoodInput(BaseModel):
    journal: str

class FatigueInput(BaseModel):
    screen_time_hours: float
    nighttime_use: int
    app_switches: int
    social_media_ratio: float
    unlocks: int

@app.get("/")
def read_root():
    return {"message": "MindfulBalance API is running!"}

@app.post("/predict/mood")
def mood_endpoint(data: MoodInput):
    result = predict_mood(data.journal)
    return {"mood": result}

@app.post("/predict/fatigue")
def fatigue_endpoint(data: FatigueInput):
    features = [
        data.screen_time_hours,
        data.nighttime_use,
        data.app_switches,
        data.social_media_ratio,
        data.unlocks
    ]
    result = predict_fatigue(features)
    return {"fatigue_level": result}
