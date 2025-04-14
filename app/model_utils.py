# app/model_utils.py

import joblib

# Load models and vectorizer
mood_model = joblib.load("app/mood_model.pkl")
mood_vectorizer = joblib.load("app/mood_vectorizer.pkl")
fatigue_model = joblib.load("app/fatigue_model.pkl")

def predict_mood(text: str) -> str:
    vectorized_text = mood_vectorizer.transform([text])
    prediction = mood_model.predict(vectorized_text)[0]
    return prediction

def predict_fatigue(features: list) -> str:
    prediction = fatigue_model.predict([features])[0]
    return prediction
