"""
AQI Forecasting Assistant - Streamlit Frontend
Simple, clean interface for AQI predictions and AI chatbot
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

API_URL = "http://localhost:5000"  # FastAPI base URL

st.set_page_config(
    page_title="AQI Forecasting Assistant",
    page_icon="🌍",
    layout="wide",
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_aqi_color(aqi: int) -> str:
    """Return color based on AQI value."""
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


def fetch_cities():
    """Get list of available cities."""
    try:
        resp = requests.get(f"{API_URL}/cities", timeout=10)
        if resp.status_code == 200:
            return resp.json()["cities"]
        return []
    except Exception:
        # Fallback demo list
        return ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"]


def get_prediction(city: str, days: int):
    """Get AQI prediction from API."""
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"city": city, "days": days},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def ask_chatbot(message: str) -> str:
    """Send message to AI chatbot (Groq via FastAPI)."""
    if not API_URL:
        return "⚠️ API server not running. Please start: uvicorn app:app --reload"

    try:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"message": message},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "No response from AI.")
        else:
            return f"❌ Error: API returned status {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure it's running on port 5000."
    except requests.exceptions.Timeout:
        return "⏱️ Request timeout. The AI backend might be slow."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_alerts():
    """Get high-risk cities."""
    try:
        resp = requests.get(f"{API_URL}/alerts", timeout=10)
        if resp.status_code == 200:
            return resp.json()["high_risk_cities"]
        return []
    except Exception:
        return []


def subscribe_alert(city: str, contact: str):
    """
    Subscribe a user for alerts.
    Expects FastAPI endpoint POST /subscribe_alerts with body:
    { "city": "...", "contact": "..." }
    """
    try:
        resp = requests.post(
            f"{API_URL}/subscribe_alerts",
            json={"city": city, "contact": contact},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, resp.json().get("message", "Subscription saved.")
        else:
            try:
                msg = resp.json().get("detail", f"Status {resp.status_code}")
            except Exception:
                msg = f"Status {resp.status_code}"
            return False, f"API error: {msg}"
    except Exception as e:
        return False, f"Request failed: {str(e)}"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🌍 AQI Forecasting Assistant")
    st.markdown("**AI-Powered Air Quality Predictions for Indian Cities**")
    st.markdown("---")

    # Sidebar navigation
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Choose a section:",
            ["🏠 Home", "📊 Predictions", "💬 AI Chatbot", "🚨 Alerts"],
        )

        st.markdown("---")
        st.markdown("### About")
        st.info(
            """
            This system predicts air quality 1–7 days in advance using machine learning.

            **Features:**
            - Real-time predictions
            - AI-powered advice
            - High pollution alerts
            """
        )

    # ========================================================================
    # HOME PAGE
    # ========================================================================
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cities Monitored", "278")
        with col2:
            st.metric("Prediction Accuracy", "85%")
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
                - Get recommendations
                - Understand pollution patterns
                """
            )

        st.markdown("---")

    # ========================================================================
    # PREDICTIONS PAGE
    # ========================================================================
    elif page == "📊 Predictions":
        st.header("📊 AQI Predictions")

        col1, col2 = st.columns([2, 1])
        with col1:
            city = st.selectbox("🏙️ Select City", fetch_cities())
        with col2:
            days = st.selectbox(
                "📅 Forecast Period",
                [1, 3, 7],
                format_func=lambda x: f"{x} Day{'s' if x > 1 else ''}",
            )

        if st.button("🔮 Get Prediction", key="predict_btn"):
            with st.spinner("Fetching prediction..."):
                result = get_prediction(city, days)

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
                    st.error("❌ Could not fetch prediction. Make sure API server is running.")

    # ========================================================================
    # CHATBOT PAGE
    # ========================================================================
    elif page == "💬 AI Chatbot":
        st.header("💬 AI Assistant")
        st.caption(
            "Ask about today’s air quality, whether it’s safe for outdoor plans, "
            "or how to reduce exposure."
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

    # ----- 1) DISPLAY CHAT HISTORY (single place) -----
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ----- 2) FUNCTION TO CALL BACKEND + UPDATE STATE -----
        def send_and_store(text: str):
        # user
            st.session_state.messages.append({"role": "user", "content": text})
        # assistant (call API)
            reply = ask_chatbot(text)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    # ----- 3) FREE-TEXT CHAT INPUT -----
        if prompt := st.chat_input("Ask about air quality..."):
            send_and_store(prompt)
            st.rerun()

    # ----- 4) QUICK QUESTIONS (NO chat_message HERE) -----
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

    # ========================================================================
    # ALERTS PAGE
    # ========================================================================
    elif page == "🚨 Alerts":
        st.header("🚨 High Pollution Alerts")
        st.markdown("Cities predicted to have **Poor** or worse air quality tomorrow.")

        alerts = get_alerts()

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
                city_choice = st.selectbox("City to monitor", fetch_cities())
                contact = st.text_input("Email or phone number")
                consent = st.checkbox(
                    "Alert me when AQI for this city is Poor or worse (max 1 alert per day)."
                )
                submitted = st.form_submit_button("Save alert preference")

            if submitted:
                if not contact or not consent:
                    st.error("Please enter your email/phone and tick the consent box.")
                else:
                    # Call backend (you'll implement /subscribe_alerts)
                    ok, msg = subscribe_alert(city_choice, contact)
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
            <p>AQI Forecasting System | Powered by ML & AI | Data updated daily</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
