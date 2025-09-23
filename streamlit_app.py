"""
🧠 Complete Integrated Streamlit Frontend with Face Analysis
Connected to FastAPI backend with Face integration
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import io
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🧠 Integrated Emotion Detection System with Face Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API configuration
API_BASE_URL = "http://localhost:8000"

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .integration-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .api-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .api-connected {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .api-disconnected {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    
    .model-info {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .face-analysis-result {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #667eea;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class IntegratedAPI:
    """Enhanced API client with face analysis support"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json'
        })
    
    def check_health(self) -> dict:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "connected", "data": response.json()}
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "disconnected", "message": str(e)}
    
    def login(self, email: str, password: str) -> dict:
        """Login user"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/login",
                json={"email": email, "password": password},
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "message": response.json().get("detail", "Login failed")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}
    
    def analyze_emotion(self, text: str, include_demographics: bool = False) -> dict:
        """Analyze emotion using backend ML models"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/analyze",
                json={
                    "text": text,
                    "include_demographics": include_demographics
                },
                timeout=30
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "message": response.json().get("detail", "Analysis failed")}
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}
    
    def analyze_face(self, image_bytes: bytes, filename: str = "image.jpg") -> dict:
        """Analyze face using Face backend"""
        try:
            files = {
                'file': (filename, io.BytesIO(image_bytes), 'image/jpeg')
            }
            
            response = self.session.post(
                f"{self.base_url}/api/analyze-face",
                files=files,
                timeout=60
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = "Face analysis failed"
                try:
                    error_detail = response.json().get("detail", error_detail)
                except:
                    pass
                return {"success": False, "message": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}
    
    def analyze_combined(self, text: str = None, image_bytes: bytes = None, filename: str = "image.jpg") -> dict:
        """Combined text and face analysis"""
        try:
            data = {}
            files = {}
            
            if text:
                data["text"] = text
                data["include_text_demographics"] = False
            
            if image_bytes:
                files['file'] = (filename, io.BytesIO(image_bytes), 'image/jpeg')
            
            response = self.session.post(
                f"{self.base_url}/api/analyze-combined",
                data=data,
                files=files if files else None,
                timeout=60
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_detail = "Combined analysis failed"
                try:
                    error_detail = response.json().get("detail", error_detail)
                except:
                    pass
                return {"success": False, "message": error_detail}
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}
    
    def get_history(self, limit: int = 10) -> dict:
        """Get analysis history"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/history?limit={limit}",
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "message": "Failed to get history"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}
    
    def get_stats(self) -> dict:
        """Get user statistics"""
        try:
            response = self.session.get(f"{self.base_url}/api/stats", timeout=10)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "message": "Failed to get stats"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}

# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = IntegratedAPI(API_BASE_URL)

if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False

if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

def check_backend_status():
    """Check and display backend status"""
    health = st.session_state.api_client.check_health()
    
    if health["status"] == "connected":
        health_data = health.get("data", {})
        
        st.markdown("""
        <div class="api-status api-connected">
            ✅ <strong>Backend Connected</strong> - FastAPI server is running with all models loaded
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ml_status = health_data.get("ml_models", "unknown")
            st.markdown(f"""
            <div class="integration-badge">
                🤖 ML Models: {ml_status.title()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            deepface_status = health_data.get("face_models", "unknown")
            st.markdown(f"""
            <div class="integration-badge">
                🎭 Face: {deepface_status.title()}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cv_available = health_data.get("cv_available", False)
            cv_status = "Available" if cv_available else "Mock Mode"
            st.markdown(f"""
            <div class="integration-badge">
                📷 CV: {cv_status}
            </div>
            """, unsafe_allow_html=True)
        
        return True
    else:
        st.markdown(f"""
        <div class="api-status api-disconnected">
            ❌ <strong>Backend Disconnected</strong> - {health.get('message', 'Cannot connect to FastAPI server')}
        </div>
        """, unsafe_allow_html=True)
        
        st.error("""
        **To start the backend with Face:**
        1. Open Command Prompt/Terminal
        2. Navigate to: `integrated-system/backend`
        3. Run: `pip install -r requirements.txt`
        4. Run: `python main.py`
        5. Wait for Face models to download (first time only)
        6. Refresh this page
        """)
        
        return False

def authenticate_user():
    """Authentication interface"""
    st.markdown('<div class="main-header">🧠 Integrated Emotion Detection System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <div class="integration-badge">🚀 FastAPI Backend</div>
        <div class="integration-badge">🤖 Hugging Face Models</div>
        <div class="integration-badge">🎭 Face Integration</div>
        <div class="integration-badge">📷 Face Analysis</div>
        <div class="integration-badge">💾 SQLite Database</div>
        <div class="integration-badge">📊 Real-time Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    backend_connected = check_backend_status()
    
    if not backend_connected:
        return
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Sign In to Continue")
        
        with st.form("login_form"):
            email = st.text_input("📧 Email", placeholder="demo@example.com")
            password = st.text_input("🔒 Password", type="password", placeholder="demo123")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_button = st.form_submit_button("🚀 Sign In", use_container_width=True)
            with col_b:
                demo_button = st.form_submit_button("🎯 Demo Mode", use_container_width=True)
            
            if login_button or demo_button:
                if demo_button:
                    email = "demo@example.com"
                    password = "demo123"
                
                if email and password:
                    with st.spinner("🔐 Authenticating with backend..."):
                        result = st.session_state.api_client.login(email, password)
                    
                    if result["success"]:
                        st.session_state.user_authenticated = True
                        st.session_state.user_data = result["data"]
                        st.success("✅ Authentication successful! Welcome to the integrated dashboard with face analysis.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Authentication failed: {result['message']}")
                else:
                    st.error("❌ Please enter email and password")

def show_text_analysis_results(analysis_data, detailed=False):
    """Display text analysis results"""
    st.success("✅ Text analysis completed using real ML models!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment = analysis_data["primary_sentiment"]
        color = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#6b7280"}.get(sentiment, "#6b7280")
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: {color}20; border-radius: 15px; border: 3px solid {color};">
            <h3>Primary Sentiment</h3>
            <h2 style="color: {color}; font-size: 2rem;">{sentiment.upper()}</h2>
            <p style="font-size: 1.2rem;">Confidence: {analysis_data['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        score = analysis_data["overall_score"]
        score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: {score_color}20; border-radius: 15px; border: 3px solid {score_color};">
            <h3>Overall Score</h3>
            <h2 style="color: {score_color}; font-size: 2rem;">{score}/100</h2>
            <p style="font-size: 1.2rem;">Emotional Wellness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        dominant = analysis_data["dominant_emotion"]
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: #8b5cf620; border-radius: 15px; border: 3px solid #8b5cf6;">
            <h3>Dominant Emotion</h3>
            <h2 style="color: #8b5cf6; font-size: 1.5rem;">{dominant.upper()}</h2>
            <p style="font-size: 1.2rem;">Most Prominent</p>
        </div>
        """, unsafe_allow_html=True)
    
    if detailed:
        st.markdown("### 📊 Detailed Emotion Analysis")
        emotions = analysis_data["emotions"]
        top_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:10])
        
        fig = px.bar(
            x=list(top_emotions.keys()),
            y=list(top_emotions.values()),
            title="Top 10 Emotions Detected by ML Model",
            color=list(top_emotions.values()),
            color_continuous_scale="viridis"
        )
        fig.update_layout(xaxis_title="Emotions", yaxis_title="Confidence Score", height=400)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    if analysis_data.get("age") or analysis_data.get("gender"):
        st.markdown("### 👤 Demographics Analysis")
        col1, col2 = st.columns(2)
        
        if analysis_data.get("age"):
            with col1:
                st.metric("Estimated Age", f"{analysis_data['age']} years")
        
        if analysis_data.get("gender"):
            with col2:
                st.metric("Predicted Gender", analysis_data["gender"].title())

def show_face_analysis_results(analysis_data, detailed=False):
    """Display face analysis results"""
    st.markdown("""
    <div class="face-analysis-result">
        <h3 style="text-align: center; margin-bottom: 1rem;">🎭 Face Analysis Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: #3b82f620; border-radius: 10px; border: 2px solid #3b82f6;">
            <h4>Detected Faces</h4>
            <h2 style="color: #3b82f6;">{analysis_data['detected_faces']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        age = analysis_data["age"]
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: #10b98120; border-radius: 10px; border: 2px solid #10b981;">
            <h4>Age</h4>
            <h2 style="color: #10b981;">{age} years</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        gender = analysis_data["gender"].title()
        confidence = analysis_data["gender_confidence"]
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: #f59e0b20; border-radius: 10px; border: 2px solid #f59e0b;">
            <h4>Gender</h4>
            <h2 style="color: #f59e0b;">{gender}</h2>
            <p style="color: #f59e0b;">({confidence:.1%})</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        dominant_emotion = analysis_data["dominant_emotion"].title()
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: #8b5cf620; border-radius: 10px; border: 2px solid #8b5cf6;">
            <h4>Dominant Emotion</h4>
            <h2 style="color: #8b5cf6;">{dominant_emotion}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    if detailed:
        st.markdown("### 📊 Facial Emotion Breakdown")
        emotions = analysis_data["emotions"]
        
        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:6]:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"**{emotion.title()}**")
            with col2:
                st.progress(score/100)
                st.write(f"{score:.1f}%")
        
        fig_pie = px.pie(
            values=list(emotions.values()),
            names=[name.title() for name in emotions.keys()],
            title="Facial Emotion Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def show_combined_analysis_results(analysis_data):
    """Display combined analysis results"""
    st.success("✅ Combined analysis completed!")
    
    text_analysis = analysis_data.get("text_analysis")
    face_analysis = analysis_data.get("face_analysis")
    combined_insights = analysis_data.get("combined_insights", {})
    
    if text_analysis and face_analysis:
        st.markdown("### 🔄 Individual Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💬 Text Analysis")
            show_text_analysis_results(text_analysis)
        
        with col2:
            st.markdown("#### 📷 Face Analysis")
            show_face_analysis_results(face_analysis)
        
        st.markdown("### 🧠 Combined Insights")
        
        if combined_insights.get("emotion_consistency") is not None:
            consistency = combined_insights["emotion_consistency"]
            text_emotion = combined_insights.get("text_emotion", "unknown")
            face_emotion = combined_insights.get("face_emotion", "unknown")
            
            if consistency:
                st.success(f"✅ **Consistent Emotions:** Both text and face show '{text_emotion}' emotion")
            else:
                st.warning(f"⚠️ **Mixed Signals:** Text shows '{text_emotion}' but face shows '{face_emotion}'")
            
            assessment = combined_insights.get("overall_assessment", "unknown")
            confidence = combined_insights.get("confidence_score", 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Assessment", assessment.replace("_", " ").title())
            
            with col2:
                st.metric("Combined Confidence", f"{confidence:.1%}")
            
            with col3:
                demographics = combined_insights.get("demographics", {})
                if demographics.get("age"):
                    st.metric("Age", f"{demographics['age']} years")
        
    elif text_analysis:
        st.markdown("### 💬 Text Analysis Only")
        show_text_analysis_results(text_analysis, detailed=True)
        
    elif face_analysis:
        st.markdown("### 📷 Face Analysis Only")
        show_face_analysis_results(face_analysis, detailed=True)
        
    else:
        st.error("No valid analysis results received")

def main_dashboard():
    """Main integrated dashboard"""
    
    user_name = st.session_state.user_data.get('name', 'User')
    st.markdown(f'<div class="main-header">🧠 Welcome back, {user_name}!</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div class="integration-badge">✅ Backend Connected</div>
        <div class="integration-badge">🤖 ML Models Active</div>
        <div class="integration-badge">🎭 Face Ready</div>
        <div class="integration-badge">💾 Database Ready</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎛️ Integrated Dashboard")
        
        page = st.selectbox(
            "📍 Navigate to:",
            [
                "🏠 Overview", 
                "🧠 Text Analysis", 
                "📷 Face Analysis", 
                "🔄 Combined Analysis", 
                "📊 Analytics & History", 
                "⚙️ System Status"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📈 Live Stats")
        
        with st.spinner("Loading stats..."):
            stats_result = st.session_state.api_client.get_stats()
        
        if stats_result["success"]:
            stats = stats_result["data"]
            st.metric("Total Analyses", stats.get("total_analyses", 0))
            st.metric("Average Score", f"{stats.get('average_score', 0)}/100")
            st.metric("Dominant Emotion", stats.get("dominant_emotion", "neutral").title())
        
        st.markdown("---")
        if st.button("🚪 Sign Out"):
            st.session_state.user_authenticated = False
            st.session_state.user_data = {}
            st.rerun()
    
    if page == "🏠 Overview":
        show_overview()
    elif page == "🧠 Text Analysis":
        show_text_analysis()
    elif page == "📷 Face Analysis":
        show_face_analysis()
    elif page == "🔄 Combined Analysis":
        show_combined_analysis()
    elif page == "📊 Analytics & History":
        show_analytics()
    elif page == "⚙️ System Status":
        show_system_status()

def show_overview():
    """Overview page"""
    st.markdown("## 🏠 Dashboard Overview")
    
    stats_result = st.session_state.api_client.get_stats()
    
    if stats_result["success"]:
        stats = stats_result["data"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Total Analyses</h4>
                <h2 style="color: #3b82f6;">{stats.get('total_analyses', 0)}</h2>
                <p>Completed analyses</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_score = stats.get('average_score', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Average Score</h4>
                <h2 style="color: #10b981;">{avg_score}/100</h2>
                <p>Overall wellness</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            dominant = stats.get('dominant_emotion', 'neutral').title()
            st.markdown(f"""
            <div class="metric-card">
                <h4>😊 Dominant Emotion</h4>
                <h2 style="color: #f59e0b;">{dominant}</h2>
                <p>Most frequent</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            trend = stats.get('recent_trend', 'stable').title()
            color = "#10b981" if trend == "Improving" else "#6b7280"
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Trend</h4>
                <h2 style="color: {color};">{trend}</h2>
                <p>Recent pattern</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### ⚡ Quick Analysis Options")
    
    tab1, tab2 = st.tabs(["💬 Text Analysis", "📷 Face Analysis"])
    
    with tab1:
        quick_text = st.text_input(
            "Enter text for quick analysis:",
            placeholder="I'm feeling great today! This integrated system is amazing!"
        )
        
        if st.button("🔍 Analyze Text", type="primary"):
            if quick_text:
                with st.spinner("🧠 Analyzing with ML models..."):
                    result = st.session_state.api_client.analyze_emotion(quick_text, include_demographics=True)
                
                if result["success"]:
                    show_text_analysis_results(result["data"])
                else:
                    st.error(f"Analysis failed: {result['message']}")
    
    with tab2:
        st.markdown("📷 **Upload an image for face emotion analysis:**")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="overview_face_upload"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🎭 Analyze Face", type="primary"):
                with st.spinner("🎭 Analyzing face with Face..."):
                    image_bytes = uploaded_file.getvalue()
                    result = st.session_state.api_client.analyze_face(image_bytes, uploaded_file.name)
                
                if result["success"]:
                    show_face_analysis_results(result["data"])
                else:
                    st.error(f"Face analysis failed: {result['message']}")

def show_text_analysis():
    """Text analysis page"""
    st.markdown("## 🧠 Advanced Text Analysis")
    st.markdown("*Powered by Hugging Face Transformers and GoEmotions dataset*")
    
    text_input = st.text_area(
        "Enter your text for comprehensive analysis:",
        placeholder="I'm feeling really excited about this new integrated system! The ML models are working perfectly and I can see real-time results. This is amazing!",
        height=150
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        include_demographics = st.checkbox("Include age/gender estimation", value=True)
    
    with col2:
        analyze_button = st.button("🔍 Analyze with ML Models", type="primary", use_container_width=True)
    
    if analyze_button and text_input:
        with st.spinner("🧠 Processing with real Hugging Face models..."):
            result = st.session_state.api_client.analyze_emotion(text_input, include_demographics)
        
        if result["success"]:
            show_text_analysis_results(result["data"], detailed=True)
        else:
            st.error(f"❌ Analysis failed: {result['message']}")

def show_face_analysis():
    """Face analysis page"""
    st.markdown("## 📷 Advanced Face Analysis")
    st.markdown("*Powered by Face - Age, Gender, and Emotion Detection*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="face_analysis_upload"
        )
    
    with col2:
        st.markdown("### 📸 Camera Capture")
        camera_image = st.camera_input("Take a picture")
    
    image_to_analyze = uploaded_file or camera_image
    
    if image_to_analyze is not None:
        image = Image.open(image_to_analyze)
        st.image(image, caption="Image to Analyze", use_column_width=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🎭 Analyze Face with Face", type="primary", use_container_width=True):
                with st.spinner("🎭 Analyzing face with Face models... This may take a moment..."):
                    image_bytes = image_to_analyze.getvalue()
                    filename = getattr(image_to_analyze, 'name', 'camera_image.jpg')
                    result = st.session_state.api_client.analyze_face(image_bytes, filename)
                
                if result["success"]:
                    show_face_analysis_results(result["data"], detailed=True)
                else:
                    st.error(f"❌ Face analysis failed: {result['message']}")

def show_combined_analysis():
    """Combined text and face analysis"""
    st.markdown("## 🔄 Combined Text & Face Analysis")
    st.markdown("*Get comprehensive insights by analyzing both text and facial expressions*")
    
    st.markdown("### 💬 Text Input")
    text_input = st.text_area(
        "Enter text (optional):",
        placeholder="I'm feeling great today! This integrated system is working perfectly.",
        height=100
    )
    
    st.markdown("### 📷 Image Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload image (optional)",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="combined_analysis_upload"
        )
    
    with col2:
        camera_image = st.camera_input("Take picture (optional)", key="combined_camera")
    
    image_to_analyze = uploaded_file or camera_image
    
    if text_input or image_to_analyze:
        st.markdown("### 📋 Analysis Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if text_input:
                st.text_area("Text to analyze:", value=text_input, disabled=True, height=100)
            else:
                st.info("No text provided")
        
        with col2:
            if image_to_analyze:
                image = Image.open(image_to_analyze)
                st.image(image, caption="Image to analyze", use_column_width=True)
            else:
                st.info("No image provided")
    
    if text_input or image_to_analyze:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔄 Run Combined Analysis", type="primary", use_container_width=True):
                with st.spinner("🔄 Running combined analysis... This may take a moment..."):
                    image_bytes = None
                    filename = "combined_image.jpg"
                    
                    if image_to_analyze:
                        image_bytes = image_to_analyze.getvalue()
                        filename = getattr(image_to_analyze, 'name', filename)
                    
                    result = st.session_state.api_client.analyze_combined(
                        text=text_input if text_input else None,
                        image_bytes=image_bytes,
                        filename=filename
                    )
                
                if result["success"]:
                    show_combined_analysis_results(result["data"])
                else:
                    st.error(f"❌ Combined analysis failed: {result['message']}")
    else:
        st.info("👆 Please provide either text or an image (or both) for combined analysis")

def show_analytics():
    """Analytics and history page"""
    st.markdown("## 📊 Analytics & History")
    
    with st.spinner("Loading analysis history..."):
        history_result = st.session_state.api_client.get_history(50)
    
    if history_result["success"]:
        analyses = history_result["data"]["analyses"]
        
        if analyses:
            st.markdown(f"### 📋 Recent Analyses ({len(analyses)} entries)")
            
            df = pd.DataFrame(analyses)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            analysis_types = df['analysis_type'].unique() if 'analysis_type' in df.columns else ['text']
            selected_types = st.multiselect(
                "Filter by analysis type:",
                analysis_types,
                default=analysis_types
            )
            
            if selected_types:
                filtered_df = df[df['analysis_type'].isin(selected_types)] if 'analysis_type' in df.columns else df
                
                fig = px.line(
                    filtered_df,
                    x='created_at',
                    y='overall_score',
                    color='analysis_type' if 'analysis_type' in filtered_df.columns else None,
                    title='Emotional Wellness Score Over Time',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Score (0-100)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                emotion_counts = filtered_df['dominant_emotion'].value_counts()
                
                fig_pie = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Emotion Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.markdown("### 📝 Recent Entries")
                
                display_df = filtered_df[['created_at', 'text', 'primary_sentiment', 'overall_score', 'dominant_emotion']].copy()
                display_df['text'] = display_df['text'].str[:100] + '...'
                display_df.columns = ['Date', 'Text', 'Sentiment', 'Score', 'Dominant Emotion']
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("Please select at least one analysis type to view.")
        
        else:
            st.info("📊 No analysis history yet. Perform some analyses to see trends!")
    
    else:
        st.error(f"❌ Failed to load history: {history_result['message']}")

def show_system_status():
    """System status page"""
    st.markdown("## ⚙️ System Status")
    
    health = st.session_state.api_client.check_health()
    
    if health["status"] == "connected":
        st.success("✅ Backend API is connected and operational")
        
        health_data = health.get("data", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 System Components")
            st.markdown(f"""
            - **API Status:** ✅ Connected
            - **ML Models:** {health_data.get('ml_models', 'Unknown').title()}
            - **Face Models:** {health_data.get('face_models', 'Unknown').title()}
            - **Computer Vision:** {'✅ Available' if health_data.get('cv_available') else '⚠️ Mock Mode'}
            - **Database:** {health_data.get('database', 'Unknown').title()}
            - **Last Check:** {health_data.get('timestamp', 'Unknown')}
            """)
        
        with col2:
            st.markdown("### 📊 API Endpoints")
            st.markdown("""
            - **Health Check:** `/health`
            - **Authentication:** `/api/login`
            - **Text Analysis:** `/api/analyze`
            - **Face Analysis:** `/api/analyze-face`
            - **Combined Analysis:** `/api/analyze-combined`
            - **History:** `/api/history`
            - **Statistics:** `/api/stats`
            """)
        
        st.markdown("### 🧪 API Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Test Text Analysis API"):
                with st.spinner("Testing text analysis API..."):
                    test_result = st.session_state.api_client.analyze_emotion("This is a test message for API integration.")
                
                if test_result["success"]:
                    st.success("✅ Text Analysis API is working correctly")
                    with st.expander("View test results"):
                        st.json(test_result["data"])
                else:
                    st.error(f"❌ Text API Test Failed: {test_result['message']}")
        
        with col2:
            if st.button("🎭 Test Face Analysis API"):
                st.info("Face analysis test requires uploading an image. Use the Face Analysis page to test this functionality.")
    
    else:
        st.error(f"❌ Backend API is not connected: {health.get('message', 'Unknown error')}")
        
        st.markdown("""
        ### 🚀 To Start the Backend with Face:
        
        1. Open Command Prompt/Terminal
        2. Navigate to: `integrated-system/backend`
        3. Install dependencies: `pip install -r requirements.txt`
        4. Start server: `python main.py`
        5. Wait for Face models to download (first time only - may take several minutes)
        6. Refresh this page
        
        The backend should start on: http://localhost:8000
        
        **Note:** Face will automatically download required models on first use:
        - Emotion detection models (~100MB)
        - Age prediction models (~50MB)
        - Gender classification models (~30MB)
        """)

def main():
    if not st.session_state.user_authenticated:
        authenticate_user()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()