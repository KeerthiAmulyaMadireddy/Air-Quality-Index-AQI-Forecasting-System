import os
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st
import plotly.graph_objects as go
from groq import Groq


st.set_page_config(
    page_title="AQI Forecasting Assistant",
    page_icon="🌍",
    layout="wide",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    print("⚠ GROQ_API_KEY not set – AI chatbot will be disabled.")


# Sample predictions (same as FastAPI)
SAMPLE_PREDICTIONS: Dict[str, Dict[str, int]] = {
    "Delhi":     {"current_aqi": 245, "predicted_1d": 280, "predicted_3d": 265, "predicted_7d": 240},
    "Mumbai":    {"current_aqi": 95,  "predicted_1d": 105, "predicted_3d": 110, "predicted_7d": 100},
    "Bangalore": {"current_aqi": 78,  "predicted_1d": 82,  "predicted_3d": 85,  "predicted_7d": 80},
    "Kolkata":   {"current_aqi": 165, "predicted_1d": 180, "predicted_3d": 175, "predicted_7d": 160},
    "Chennai":   {"current_aqi": 88,  "predicted_1d": 92,  "predicted_3d": 95,  "predicted_7d": 90},
}

# In-memory demo store for alert subscriptions
ALERT_SUBSCRIPTIONS: List[Dict[str, str]] = []

# Helper Functions
def get_aqi_color(aqi: int) -> str:
    if aqi <= 50:
        return "#00E400"  # Green
    elif aqi <= 100:
        return "#FFFF00"  # Yellow
    elif aqi <= 200:
        return "#FF7E00"  # Orange
    elif aqi <= 300:
        return "#FF0000"  # Red
    elif aqi <= 400:
        return "#8F3F97"  # Purple
    else:
        return "#7E0023"  # Maroon


def fetch_cities_local() -> List[str]:
    return list(SAMPLE_PREDICTIONS.keys())


def get_prediction_local(city: str, days: int) -> Dict[str, Any] | None:
    city_data = SAMPLE_PREDICTIONS.get(city)
    if not city_data:
        return None

    if days == 1:
        predicted_aqi = city_data["predicted_1d"]
    elif days == 3:
        predicted_aqi = city_data["predicted_3d"]
    elif days == 7:
        predicted_aqi = city_data["predicted_7d"]
    else:
        return None

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

    return {
        "city": city,
        "current_aqi": city_data["current_aqi"],
        "predicted_aqi": predicted_aqi,
        "forecast_days": days,
        "category": category,
        "advice": advice,
        "timestamp": datetime.now().isoformat(),
    }


def get_alerts_local() -> List[Dict[str, Any]]:
    high: List[Dict[str, Any]] = []
    for city, data in SAMPLE_PREDICTIONS.items():
        if data["predicted_1d"] > 200:
            severity = "High" if data["predicted_1d"] <= 300 else "Severe"
            high.append(
                {
                    "city": city,
                    "predicted_aqi": data["predicted_1d"],
                    "severity": severity,
                }
            )
    return high


def subscribe_alert_local(city: str, contact: str) -> tuple[bool, str]:
    ALERT_SUBSCRIPTIONS.append(
        {"city": city, "contact": contact, "ts": datetime.now().isoformat()}
    )
    return True, f"Alert preference saved for {city}. (Demo – no real messages yet.)"


def ask_chatbot_local(message: str) -> str:
    """Call Groq directly with a short, focused prompt."""
    if groq_client is None:
        return "⚠️ AI backend is not configured. Please set GROQ_API_KEY in environment."

    prompt = f"""You are an AQI (Air Quality Index) assistant.

Context data (cities and forecasts): {SAMPLE_PREDICTIONS}

User question: {message}

Instructions:
- Answer in 2–3 short sentences (max ~60 words).
- Use only the most relevant city examples.
- Do NOT repeat all context data.
- Give one clear health recommendation if appropriate.
"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=120,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error talking to AI: {e}"


# ============================================================================
# MAIN APP

def main():
    st.title("🌍 AQI Forecasting Assistant")
    st.markdown("**AI-Powered Air Quality Predictions for Indian Cities**")
    st.markdown("---")

    with st.sidebar:
        st.caption(f"GROQ key detected: {bool(GROQ_API_KEY)}")
        st.header("📋 Navigation")
        st.caption(f"GROQ key detected: {bool(GROQ_API_KEY)}")
        page = st.radio(
            "Choose a section:",
            ["🏠 Home", "📊 Predictions", "💬 AI Chatbot", "🚨 Alerts"],
        )

        st.markdown("---")
        st.markdown("### About")
        st.info(
            """
            This system predicts air quality 1–7 days in advance using sample ML outputs.

            **Features:**
            - City-level AQI forecasts
            - AI-powered advice via Groq
            - High pollution alerts (demo subscriptions)
            """
        )

    # -------------------------- HOME --------------------------
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cities Monitored", "5")
        with col2:
            st.metric("Prediction Accuracy", "Demo")
        with col3:
            st.metric("Forecast Horizon", "7 Days")

        st.markdown("---")

        st.subheader("🎯 What You Can Do")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                ### 📊 Get Predictions
                - Check AQI forecasts for your city
                - See 1-day, 3-day, or 7-day predictions
                - Get personalized health advice
                """
            )

        with col2:
            st.markdown(
                """
                ### 💬 Ask Questions
                - Chat with AI about air quality
                - Get safety recommendations
                - Understand pollution patterns
                """
            )

    # ----------------------- PREDICTIONS ----------------------
    elif page == "📊 Predictions":
        st.header("📊 AQI Predictions")

        col1, col2 = st.columns([2, 1])
        with col1:
            city = st.selectbox("🏙️ Select City", fetch_cities_local())
        with col2:
            days = st.selectbox(
                "📅 Forecast Period",
                [1, 3, 7],
                format_func=lambda x: f"{x} Day{'s' if x > 1 else ''}",
            )

        if st.button("🔮 Get Prediction", key="predict_btn"):
            with st.spinner("Computing prediction..."):
                result = get_prediction_local(city, days)

                if result:
                    st.success("✅ Prediction Retrieved")

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Current AQI", f"{result['current_aqi']}")
                    with c2:
                        delta = result["predicted_aqi"] - result["current_aqi"]
                        st.metric(
                            f"Predicted ({days}d)",
                            f"{result['predicted_aqi']}",
                            delta=f"{delta:+.0f}",
                        )
                    with c3:
                        st.metric("Category", result["category"])

                    aqi_color = get_aqi_color(result["predicted_aqi"])
                    st.markdown(
                        f"""
                        <div style="background-color: {aqi_color};
                                    padding: 20px; border-radius: 10px; margin: 20px 0;">
                            <h3 style="color: white; margin: 0;">
                                AQI: {result['predicted_aqi']}
                            </h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.info(f"**Health Advice:** {result['advice']}")

                    fig = go.Figure()
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number",
                            value=result["predicted_aqi"],
                            title={"text": f"{city} AQI Forecast"},
                            gauge={
                                "axis": {"range": [None, 500]},
                                "bar": {"color": aqi_color},
                                "steps": [
                                    {"range": [0, 50], "color": "#00E400"},
                                    {"range": [50, 100], "color": "#FFFF00"},
                                    {"range": [100, 200], "color": "#FF7E00"},
                                    {"range": [200, 300], "color": "#FF0000"},
                                    {"range": [300, 400], "color": "#8F3F97"},
                                    {"range": [400, 500], "color": "#7E0023"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 200,
                                },
                            },
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Could not compute prediction.")

    # ------------------------- CHATBOT ------------------------
    elif page == "💬 AI Chatbot":
        st.header("💬 AI Assistant")
        st.caption(
            "Ask about today’s air quality, whether it’s safe for outdoor plans, "
            "or how to reduce exposure."
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display history at the top
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        def send_and_store(text: str):
            st.session_state.messages.append({"role": "user", "content": text})
            reply = ask_chatbot_local(text)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        # Free text input
        if prompt := st.chat_input("Ask about air quality..."):
            send_and_store(prompt)
            st.rerun()

        # Quick questions
        st.markdown("---")
        st.markdown("**💡 Quick questions**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Is it safe to run in Delhi tomorrow?", use_container_width=True):
                send_and_store("Is it safe to run in Delhi tomorrow?")
                st.rerun()
            if st.button("Why is air quality worse in winter?", use_container_width=True):
                send_and_store("Why is air quality worse in winter?")
                st.rerun()

        with col2:
            if st.button("Which city has the best air quality today?", use_container_width=True):
                send_and_store("Which city has the best air quality today?")
                st.rerun()
            if st.button("What should I do on high pollution days?", use_container_width=True):
                send_and_store("What should I do on high pollution days?")
                st.rerun()

    # -------------------------- ALERTS ------------------------
    elif page == "🚨 Alerts":
        st.header("🚨 High Pollution Alerts")
        st.markdown("Cities predicted to have **Poor** or worse air quality tomorrow.")

        alerts = get_alerts_local()

        if alerts:
            st.warning(f"⚠️ {len(alerts)} cities are expected to have high pollution levels tomorrow.")

            for alert in alerts:
                severity_icon = "🔴" if alert["severity"] == "Severe" else "🟠"
                with st.expander(f"{severity_icon} {alert['city']} – AQI: {alert['predicted_aqi']}"):
                    st.markdown(
                        f"""
**Predicted AQI:** {alert['predicted_aqi']}  
**Severity:** {alert['severity']}

**Recommendations:**
- Avoid long outdoor activities
- Use N95 masks if going outside
- Keep windows closed
- Run air purifiers indoors
"""
                    )

            st.markdown("---")
            st.subheader("📬 Get alerts when air gets bad")

            with st.form("alert_signup"):
                city_choice = st.selectbox("City to monitor", fetch_cities_local())
                contact = st.text_input("Email or phone number")
                consent = st.checkbox(
                    "Alert me when AQI for this city is Poor or worse (max 1 alert per day)."
                )
                submitted = st.form_submit_button("Save alert preference")

            if submitted:
                if not contact or not consent:
                    st.error("Please enter your email/phone and tick the consent box.")
                else:
                    ok, msg = subscribe_alert_local(city_choice, contact)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
        else:
            st.success("✅ No high pollution alerts for tomorrow!")
            st.balloons()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>AQI Forecasting System | Powered by sample ML outputs & Groq</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
