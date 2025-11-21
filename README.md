<<<<<<< HEAD
# 🧠 AI-Driven Emotion & Sentiment Analyzer

A comprehensive emotion and sentiment analysis system that combines **text analysis** using Hugging Face transformers and **facial emotion detection** using DeepFace. Built with FastAPI backend and Streamlit frontend.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Components](#-components)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🧠 Text Emotion Analysis
- **Real ML Models**: Uses Hugging Face transformers (`j-hartmann/emotion-english-distilroberta-base`)
- **Sentiment Analysis**: Powered by `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **27 Emotion Categories**: Detects emotions like joy, sadness, anger, fear, surprise, and more
- **Confidence Scores**: Provides detailed confidence metrics for each emotion
- **Overall Wellness Score**: Calculates emotional wellness score (0-100)

### 📷 Facial Emotion Detection
- **DeepFace Integration**: Advanced facial emotion recognition
- **Multi-Modal Analysis**: Detects age, gender, race, and emotions from facial images
- **Real-time Processing**: Fast image processing with support for multiple image formats
- **Emotion Mapping**: Maps facial expressions to 7 core emotions (happy, sad, angry, fear, surprise, disgust, neutral)

### 🔄 Combined Analysis
- **Multi-Modal Insights**: Combines text and facial analysis for comprehensive emotion detection
- **Consistency Checking**: Compares emotions from text and face to identify mixed signals
- **Demographics Integration**: Provides age and gender estimation

### 📊 Analytics & Tracking
- **Analysis History**: Stores all analyses in SQLite database
- **Mood Calendar**: Interactive calendar for tracking daily moods
- **Statistics Dashboard**: Visual analytics with charts and trends
- **Task Suggestions**: AI-powered mood-boosting task recommendations

### 🎯 Additional Features
- **User Authentication**: Simple login system with session management
- **Real-time Dashboard**: Beautiful Streamlit interface with live updates
- **API Documentation**: Auto-generated FastAPI docs at `/docs`
- **Health Monitoring**: System status and model availability checks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  (User Interface, Visualizations, Interactive Dashboard)     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST API
┌──────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ ML Manager   │  │ DeepFace     │  │ Database     │      │
│  │ (HuggingFace)│  │ Manager      │  │ Manager      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐
│ Hugging Face │ │  DeepFace  │ │ SQLite    │
│  Models      │ │  Models    │ │ Database  │
└──────────────┘ └────────────┘ └───────────┘
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** (0.95.2) - Modern, fast web framework for building APIs
- **Uvicorn** - ASGI server for FastAPI
- **Pydantic** - Data validation using Python type annotations

### Machine Learning
- **PyTorch** (2.1.1) - Deep learning framework
- **Transformers** (4.36.0) - Hugging Face transformers library
- **DeepFace** (0.0.87) - Facial recognition and emotion detection
- **OpenCV** (4.8.1.78) - Computer vision library
- **TensorFlow** (2.15.0) - Deep learning framework (for DeepFace)

### Frontend
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive data visualization
- **Pandas** - Data manipulation and analysis

### Database
- **SQLite** - Lightweight, serverless database

### Additional Libraries
- **Pillow** - Image processing
- **NumPy** - Numerical computing
- **Requests** - HTTP library

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/abhiruchi006/AI-DRIVEN-EMOTION-SENTIMENT-ANALYSIZER.git
cd AI-DRIVEN-EMOTION-SENTIMENT-ANALYSIZER
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

#### Option A: Using requirements.txt
```bash
pip install -r requirements.txt
```

#### Option B: Using the batch script (Windows)
```bash
run_backend.bat
```

**Note**: The first installation may take 10-15 minutes as it downloads:
- PyTorch models (~2GB)
- Hugging Face transformers models (~500MB)
- DeepFace models (~200MB)

### Step 4: Verify Installation
```bash
python -c "import torch; import transformers; import deepface; print('✅ All dependencies installed successfully!')"
```

## 🚀 Usage

### Starting the Backend Server

#### Option 1: Direct Python
```bash
python main.py
```

#### Option 2: Using Uvicorn
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start on `http://localhost:8000`

**First Run**: DeepFace will automatically download required models (this may take 5-10 minutes):
- Emotion detection models
- Age prediction models
- Gender classification models
- Race classification models

### Starting the Frontend

Open a new terminal and run:
```bash
streamlit run streamlit_app.py
```

The frontend will start on `http://localhost:8501`

### Access Points
- **Frontend Dashboard**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
AI-DRIVEN-EMOTION-SENTIMENT-ANALYSIZER/
│
├── main.py                 # FastAPI backend server
├── streamlit_app.py        # Streamlit frontend application
├── requirements.txt         # Python dependencies
├── run_backend.bat         # Windows batch script for setup
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── emotion_data.db         # SQLite database (created on first run)
│
└── __pycache__/            # Python cache files (auto-generated)
```

## 🔌 API Endpoints

### Authentication
- `POST /api/login` - User login/authentication

### Text Analysis
- `POST /api/analyze` - Analyze text for emotions and sentiment
  ```json
  {
    "text": "I'm feeling great today!",
    "include_demographics": false
  }
  ```

### Face Analysis
- `POST /api/analyze-face` - Analyze facial emotions from image
  - Accepts: Image file (jpg, jpeg, png, webp)
  - Returns: Emotions, age, gender, race, confidence scores

### Combined Analysis
- `POST /api/analyze-combined` - Combined text and face analysis
  - Accepts: Text (optional) and Image file (optional)
  - Returns: Combined insights with consistency checking

### Analytics
- `GET /api/history` - Get analysis history
- `GET /api/stats` - Get user statistics
- `GET /api/mood-calendar` - Get mood calendar data
- `GET /api/mood-tasks` - Get mood-boosting task suggestions

### Mood Tracking
- `POST /api/mood-entry` - Save daily mood entry
- `POST /api/mood-task/complete` - Mark task as completed

### System
- `GET /` - API information
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)

## 🧩 Components

### 1. IntegratedMLManager
- **Purpose**: Manages Hugging Face ML models for text emotion analysis
- **Models Used**:
  - `j-hartmann/emotion-english-distilroberta-base` (Emotion detection)
  - `cardiffnlp/twitter-roberta-base-sentiment-latest` (Sentiment analysis)
- **Features**:
  - 27 emotion categories detection
  - Sentiment classification (positive/negative/neutral)
  - Confidence scoring
  - Fallback mock models for testing

### 2. DeepFaceManager
- **Purpose**: Handles facial emotion detection and demographic analysis
- **Capabilities**:
  - Emotion detection (7 emotions)
  - Age estimation
  - Gender classification
  - Race classification (optional)
- **Models**: Uses DeepFace's pre-trained models
- **Features**:
  - Multi-face detection
  - Confidence scoring
  - Real-time processing

### 3. DatabaseManager
- **Purpose**: Manages SQLite database operations
- **Tables**:
  - `users` - User accounts
  - `emotion_analyses` - Text analysis history
  - `face_analyses` - Face analysis history
  - `mood_entries` - Daily mood tracking
  - `task_completions` - Completed mood-boosting tasks
- **Features**:
  - Automatic schema creation
  - Data persistence
  - Query optimization

### 4. Streamlit Frontend
- **Pages**:
  - Overview Dashboard
  - Text Analysis
  - Face Analysis
  - Combined Analysis
  - Analytics & History
  - Mood Calendar
  - System Status
- **Features**:
  - Interactive visualizations
  - Real-time updates
  - Beautiful UI with custom CSS
  - Responsive design

## 📊 Emotion Categories

### Text Emotions (27 categories)
- **Positive**: joy, love, admiration, amusement, approval, caring, excitement, gratitude, optimism, pride, relief
- **Negative**: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
- **Neutral**: confusion, curiosity, desire, neutral, realization, surprise

### Facial Emotions (7 categories)
- Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral

## 🎯 Use Cases

1. **Mental Health Monitoring**: Track daily emotions and mood patterns
2. **Customer Sentiment Analysis**: Analyze customer feedback and reviews
3. **Social Media Monitoring**: Detect emotions in social media posts
4. **Human-Computer Interaction**: Improve user experience through emotion recognition
5. **Research & Analytics**: Study emotional patterns and trends
6. **Wellness Applications**: Provide mood-boosting suggestions and tasks

## 🔧 Configuration

### Backend Configuration
Edit `main.py` to modify:
- API port (default: 8000)
- CORS origins
- Model selection
- Database path

### Frontend Configuration
Edit `streamlit_app.py` to modify:
- API base URL (default: http://localhost:8000)
- UI themes
- Page layout

## 🐛 Troubleshooting

### Issue: Models not loading
**Solution**: Ensure you have sufficient disk space (at least 5GB free) and stable internet connection for first-time model downloads.

### Issue: DeepFace errors
**Solution**: 
- Install TensorFlow: `pip install tensorflow`
- Ensure OpenCV is properly installed: `pip install opencv-python opencv-contrib-python`

### Issue: Port already in use
**Solution**: Change the port in `main.py` or kill the process using the port:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill
```

### Issue: Database errors
**Solution**: Delete `emotion_data.db` and restart the application (it will be recreated).

## 📈 Performance

- **Text Analysis**: ~1-2 seconds per request
- **Face Analysis**: ~3-5 seconds per image (first time may be slower due to model loading)
- **Combined Analysis**: ~4-7 seconds
- **Database Queries**: <100ms

## 🔒 Security Notes

- This is a demo application with basic authentication
- For production use, implement:
  - Proper password hashing
  - JWT tokens
  - Rate limiting
  - Input validation
  - HTTPS encryption

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Abhiruchi**
- GitHub: [@abhiruchi006](https://github.com/abhiruchi006)

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for pre-trained transformer models
- [DeepFace](https://github.com/serengil/deepface) for facial emotion detection
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Streamlit](https://streamlit.io/) for the interactive frontend framework

## 📞 Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**⭐ If you find this project helpful, please give it a star!**
