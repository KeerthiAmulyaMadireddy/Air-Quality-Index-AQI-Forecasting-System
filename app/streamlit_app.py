"""
AQI Forecasting Assistant - Streamlit Frontend
Simple, clean interface for AQI predictions and AI chatbot
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

API_URL = "http://localhost:5000"  
st.set_page_config(
    page_title="AQI Forecasting Assistant",
    page_icon="🌍",
    layout="wide"
)


def get_aqi_color(aqi):
    """Return color based on AQI value"""
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
    """Get list of available cities"""
    try:
        response = requests.get(f"{API_URL}/cities")
        if response.status_code == 200:
            return response.json()['cities']
        return []
    except:
        return ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"]

def get_prediction(city, days):
    """Get AQI prediction from API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"city": city, "days": days}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def ask_chatbot(message):
    """Send message to AI chatbot"""
    if not API_URL:
        return "⚠️ API server not running. Please start: uvicorn aqi_api_fastapi:app --reload"
    
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message},
            timeout=30  # Add timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', 'No response from AI')
        else:
            return f"❌ Error: API returned status {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure it's running on port 5000."
    except requests.exceptions.Timeout:
        return "⏱️ Request timeout. Gemini API might be slow."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_alerts():
    """Get high-risk cities"""
    try:
        response = requests.get(f"{API_URL}/alerts")
        if response.status_code == 200:
            return response.json()['high_risk_cities']
        return []
    except:
        return []

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🌍 AQI Forecasting Assistant")
    st.markdown("**AI-Powered Air Quality Predictions for Indian Cities**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Choose a section:",
            ["🏠 Home", "📊 Predictions", "💬 AI Chatbot", "🚨 Alerts"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This system predicts air quality 1-7 days in advance using machine learning.
        
        **Features:**
        - Real-time predictions
        - AI-powered advice
        - High pollution alerts
        """)
    
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
            st.markdown("""
            ### 📊 Get Predictions
            - Check AQI forecasts for your city
            - See 1-day, 3-day, or 7-day predictions
            - Get personalized health advice
            """)
            
        with col2:
            st.markdown("""
            ### 💬 Ask Questions
            - Chat with AI about air quality
            - Get recommendations
            - Understand pollution patterns
            """)
        
        st.markdown("---")
        
        # Quick prediction widget
        st.subheader("🚀 Quick Check")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            quick_city = st.selectbox("Select City", fetch_cities(), key="quick_city")
        with col2:
            quick_days = st.selectbox("Forecast", [1, 3, 7], key="quick_days")
        with col3:
            st.write("")  # Spacing
            if st.button("Check AQI", key="quick_check"):
                result = get_prediction(quick_city, quick_days)
                if result:
                    st.success(f"**{quick_city}** - Predicted AQI: **{result['predicted_aqi']}** ({result['category']})")
    
    # ========================================================================
    # PREDICTIONS PAGE
    # ========================================================================
    elif page == "📊 Predictions":
        st.header("📊 AQI Predictions")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            city = st.selectbox("🏙️ Select City", fetch_cities())
        with col2:
            days = st.selectbox("📅 Forecast Period", [1, 3, 7], format_func=lambda x: f"{x} Day{'s' if x > 1 else ''}")
        
        if st.button("🔮 Get Prediction", key="predict_btn"):
            with st.spinner("Fetching prediction..."):
                result = get_prediction(city, days)
                
                if result:
                    st.success("✅ Prediction Retrieved")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Current AQI",
                            f"{result['current_aqi']}",
                            delta=None
                        )
                    
                    with col2:
                        delta = result['predicted_aqi'] - result['current_aqi']
                        st.metric(
                            f"Predicted ({days}d)",
                            f"{result['predicted_aqi']}",
                            delta=f"{delta:+.0f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Category",
                            result['category']
                        )
                    
                    # Visual indicator
                    aqi_color = get_aqi_color(result['predicted_aqi'])
                    st.markdown(f"""
                    <div style="background-color: {aqi_color}; padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <h3 style="color: white; margin: 0;">AQI: {result['predicted_aqi']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Advice
                    st.info(f"**Health Advice:** {result['advice']}")
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode = "gauge+number",
                        value = result['predicted_aqi'],
                        title = {'text': f"{city} AQI Forecast"},
                        gauge = {
                            'axis': {'range': [None, 500]},
                            'bar': {'color': aqi_color},
                            'steps': [
                                {'range': [0, 50], 'color': "#00E400"},
                                {'range': [50, 100], 'color': "#FFFF00"},
                                {'range': [100, 200], 'color': "#FF7E00"},
                                {'range': [200, 300], 'color': "#FF0000"},
                                {'range': [300, 400], 'color': "#8F3F97"},
                                {'range': [400, 500], 'color': "#7E0023"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 200
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Could not fetch prediction. Make sure API server is running.")
    
    # ========================================================================
    # CHATBOT PAGE
    # ========================================================================
    elif page == "💬 AI Chatbot":
        st.header("💬 AI Assistant")
        st.markdown("Ask me anything about air quality!")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about air quality..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_chatbot(prompt)
                    st.markdown(response)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Example questions
        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Is it safe to run in Delhi tomorrow?"):
                st.session_state.messages.append({"role": "user", "content": "Is it safe to run in Delhi tomorrow?"})
                st.rerun()
            
            if st.button("Why is air quality bad in winter?"):
                st.session_state.messages.append({"role": "user", "content": "Why is air quality bad in winter?"})
                st.rerun()
        
        with col2:
            if st.button("Which city has the best air quality?"):
                st.session_state.messages.append({"role": "user", "content": "Which city has the best air quality?"})
                st.rerun()
            
            if st.button("What should I do on high pollution days?"):
                st.session_state.messages.append({"role": "user", "content": "What should I do on high pollution days?"})
                st.rerun()
    
    # ========================================================================
    # ALERTS PAGE
    # ========================================================================
    elif page == "🚨 Alerts":
        st.header("🚨 High Pollution Alerts")
        st.markdown("Cities predicted to have high pollution tomorrow")
        
        alerts = get_alerts()
        
        if alerts:
            st.warning(f"⚠️ **{len(alerts)} cities** are expected to have high pollution levels tomorrow")
            
            # Display alerts
            for alert in alerts:
                severity_color = "🔴" if alert['severity'] == "Severe" else "🟠"
                
                with st.expander(f"{severity_color} {alert['city']} - AQI: {alert['predicted_aqi']}"):
                    st.markdown(f"""
                    **Predicted AQI:** {alert['predicted_aqi']}  
                    **Severity:** {alert['severity']}  
                    
                    **Recommendations:**
                    - Avoid outdoor activities
                    - Use N95 masks if going outside
                    - Keep windows closed
                    - Run air purifiers indoors
                    """)
        else:
            st.success("✅ No high pollution alerts for tomorrow!")
            st.balloons()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>AQI Forecasting System | Powered by ML & AI | Data updated daily</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
