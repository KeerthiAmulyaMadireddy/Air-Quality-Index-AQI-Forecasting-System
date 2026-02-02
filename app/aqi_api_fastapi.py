# aqi_api_fastapi.py
# AQI Prediction API with AI Chatbot (FastAPI version)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import mlflow.sklearn
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")
genai.configure(api_key=GEMINI_API_KEY)

MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "YOUR_MLFLOW_RUN_ID")
model = None

SAMPLE_PREDICTIONS: Dict[str, Dict[str, int]] = {
    "Delhi": {"current_aqi": 245, "predicted_1d": 280, "predicted_3d": 265, "predicted_7d": 240},
    "Mumbai": {"current_aqi": 95, "predicted_1d": 105, "predicted_3d": 110, "predicted_7d": 100},
    "Bangalore": {"current_aqi": 78, "predicted_1d": 82, "predicted_3d": 85, "predicted_7d": 80},
    "Kolkata": {"current_aqi": 165, "predicted_1d": 180, "predicted_3d": 175, "predicted_7d": 160},
    "Chennai": {"current_aqi": 88, "predicted_1d": 92, "predicted_3d": 95, "predicted_7d": 90},
}

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AQI Forecasting API",
    version="1.0",
    description="AQI prediction and AI chatbot using sample data + Gemini.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class PredictRequest(BaseModel):
    city: str = "Delhi"
    days: int = 1  # 1, 3, or 7

class PredictResponse(BaseModel):
    city: str
    current_aqi: int
    predicted_aqi: int
    forecast_days: int
    category: str
    advice: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    question: str
    response: str
    timestamp: str

class AlertsEntry(BaseModel):
    city: str
    predicted_aqi: int
    severity: str

class AlertsResponse(BaseModel):
    high_risk_cities: List[AlertsEntry]
    count: int
    timestamp: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model() -> None:
    global model
    try:
        model = mlflow.sklearn.load_model(f"runs:/{MODEL_RUN_ID}/model")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Model not loaded - using sample data: {e}")

def get_city_context(city: str) -> Optional[Dict[str, Any]]:
    return SAMPLE_PREDICTIONS.get(city)

def generate_ai_response(user_question: str, context: Dict[str, Any]) -> str:
    prompt = f"""You are an AQI (Air Quality Index) forecasting assistant with access to real-time predictions.

User Question: {user_question}

Available Data: {context}

Instructions:
- Be helpful, clear, and concise (max 100 words)
- Use specific AQI numbers from the data
- Give actionable health advice
- Explain in simple terms
- If data is unavailable, say so politely

Answer:"""

    try:
        gmodel = genai.GenerativeModel("gemini-pro")
        response = gmodel.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I'm having trouble generating a response. Error: {str(e)}"

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
def home():
    return {
        "service": "AQI Forecasting API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Get AQI predictions for a city",
            "/chat": "POST - Ask questions about air quality",
            "/cities": "GET  - Get list of available cities",
            "/alerts": "GET  - High pollution alerts",
        },
    }

@app.get("/cities")
def get_cities():
    return {
        "cities": list(SAMPLE_PREDICTIONS.keys()),
        "count": len(SAMPLE_PREDICTIONS),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    city = body.city
    days = body.days

    city_data = get_city_context(city)
    if not city_data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"City '{city}' not found",
                "available_cities": list(SAMPLE_PREDICTIONS.keys()),
            },
        )

    if days == 1:
        predicted_aqi = city_data["predicted_1d"]
    elif days == 3:
        predicted_aqi = city_data["predicted_3d"]
    elif days == 7:
        predicted_aqi = city_data["predicted_7d"]
    else:
        raise HTTPException(status_code=400, detail="Days must be 1, 3, or 7")

    if predicted_aqi <= 50:
        category = "Good"
        advice = "Air quality is excellent. Great day for outdoor activities!"
    elif predicted_aqi <= 100:
        category = "Satisfactory"
        advice = "Air quality is acceptable. Enjoy outdoor activities."
    elif predicted_aqi <= 200:
        category = "Moderate"
        advice = "Sensitive individuals should limit prolonged outdoor exposure."
    elif predicted_aqi <= 300:
        category = "Poor"
        advice = "Everyone should reduce outdoor activities. Use masks outdoors."
    elif predicted_aqi <= 400:
        category = "Very Poor"
        advice = "Avoid outdoor activities. Keep windows closed. Use air purifiers."
    else:
        category = "Severe"
        advice = "Stay indoors. Health emergency for all. Medical attention may be needed."

    return PredictResponse(
        city=city,
        current_aqi=city_data["current_aqi"],
        predicted_aqi=predicted_aqi,
        forecast_days=days,
        category=category,
        advice=advice,
        timestamp=datetime.now().isoformat(),
    )

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    if not body.message:
        raise HTTPException(status_code=400, detail="Message is required")

    context = SAMPLE_PREDICTIONS
    ai_response = generate_ai_response(body.message, context)

    return ChatResponse(
        question=body.message,
        response=ai_response,
        timestamp=datetime.now().isoformat(),
    )

@app.get("/alerts", response_model=AlertsResponse)
def get_alerts():
    high_risk: List[AlertsEntry] = []

    for city, data in SAMPLE_PREDICTIONS.items():
        if data["predicted_1d"] > 200:
            severity = "High" if data["predicted_1d"] <= 300 else "Severe"
            high_risk.append(
                AlertsEntry(
                    city=city,
                    predicted_aqi=data["predicted_1d"],
                    severity=severity,
                )
            )

    return AlertsResponse(
        high_risk_cities=high_risk,
        count=len(high_risk),
        timestamp=datetime.now().isoformat(),
    )

