

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import mlflow.sklearn
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "YOUR_MLFLOW_RUN_ID")
model = None

# Backend switches
USE_HUGGINGFACE = False   
USE_OLLAMA = False        
USE_GROQ = True          

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if USE_GROQ:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✓ Groq client initialized")

# Sample data
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
    title="AQI Forecasting API (Open Source)",
    version="1.0",
    description="AQI prediction and AI chatbot using Groq Llama models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class PredictRequest(BaseModel):
    city: str = "Delhi"
    days: int = 1

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
    model_used: str

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

def generate_ai_response_groq(user_question: str, context: Dict[str, Any]) -> str:
    try:
        prompt = f"""You are an AQI (Air Quality Index) assistant.

Context data (do not repeat everything unless needed): {context}

User question: {user_question}

Instructions:
- Answer in 2–3 short sentences (max ~60 words).
- Use only the most relevant city examples.
- Do NOT repeat the full context.
- Give one clear health recommendation if appropriate.
"""

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,          # slightly lower = less rambling
            max_tokens=120,           # hard cap on length
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with Groq: {str(e)}"

def generate_ai_response_rulebased(user_question: str, context: Dict[str, Any]) -> str:
    q = user_question.lower()
    mentioned_city = None
    for city in context.keys():
        if city.lower() in q:
            mentioned_city = city
            break

    if mentioned_city:
        data = context[mentioned_city]
        return (
            f"{mentioned_city} currently has an AQI of {data['current_aqi']}. "
            f"Tomorrow's forecast is {data['predicted_1d']}. "
            f"{'Health advisory: Limit outdoor activities.' if data['predicted_1d'] > 150 else 'Air quality is acceptable.'}"
        )

    if "highest" in q or "worst" in q:
        worst_city = max(context.items(), key=lambda x: x[1]["predicted_1d"])
        return f"{worst_city[0]} has the highest predicted AQI at {worst_city[1]['predicted_1d']}."

    if "lowest" in q or "best" in q:
        best_city = min(context.items(), key=lambda x: x[1]["predicted_1d"])
        return f"{best_city[0]} has the best air quality with AQI {best_city[1]['predicted_1d']}."

    return (
        "I can provide AQI data for Delhi, Mumbai, Bangalore, Kolkata, and Chennai. "
        "Try asking about a specific city or use the /predict endpoint."
    )

def generate_ai_response(user_question: str, context: Dict[str, Any]) -> tuple[str, str]:
    if USE_GROQ:
        return generate_ai_response_groq(user_question, context), "Groq (llama-3.1-8b-instant)"
    else:
        return generate_ai_response_rulebased(user_question, context), "Rule-based"

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
def home():
    model_info = "Groq" if USE_GROQ else "Rule-based"
    return {
        "service": "AQI Forecasting API (Open Source)",
        "version": "1.0",
        "ai_model": model_info,
        "endpoints": {
            "/predict": "POST - Get AQI predictions for a city",
            "/chat": "POST - Ask questions about air quality",
            "/cities": "GET  - Get list of available cities",
            "/alerts": "GET  - High pollution alerts",
            "/health": "GET  - API health check",
        },
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ai_backend": "Groq" if USE_GROQ else "Rule-based",
        "timestamp": datetime.now().isoformat(),
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
    ai_response, model_used = generate_ai_response(body.message, context)

    return ChatResponse(
        question=body.message,
        response=ai_response,
        timestamp=datetime.now().isoformat(),
        model_used=model_used,
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

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    load_model()
    print("=" * 60)
    print("🚀 AQI Forecasting API Started")
    print("=" * 60)
    if USE_GROQ:
        print("✓ Using Groq API (llama-3.1-8b-instant)")
    else:
        print("⚠ Using rule-based responses")
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)


from pydantic import BaseModel

class AlertSubscription(BaseModel):
    city: str
    contact: str  # email or phone

ALERT_SUBSCRIPTIONS: list[AlertSubscription] = []  


@app.post("/subscribe_alerts")
def subscribe_alerts(body: AlertSubscription):
    # naive in-memory store; in real app use DB
    ALERT_SUBSCRIPTIONS.append(body)
    return {
        "message": f"Alert preference saved for {body.city}. "
                   "This is a demo; no real messages are sent yet."
    }
