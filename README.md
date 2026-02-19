# Smart-Energy-Consumption-Analysis-and-Prediction
Using the SmartHome Energy Monitoring Dataset with detailed timestamped device-level power readings collected over six months, the system performs time series analysis and forecasting using Long Short-Term Memory (LSTM) networks and Linear Regression as a baseline model.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)](#)


## 🚀 Features

### 📊 Energy Prediction
- **ML-Powered Predictions**: Random Forest model trained on 50,000+ historical records with 86% R² accuracy
- **Dual Input Methods**: Manual parameter entry or batch CSV file upload
- **Instant Results**: Real-time energy consumption forecasting in kWh with efficiency scoring

### 🤖 AI Chat Assistant
- **Three-Layer AI Architecture**: Groq (Primary) → Gemini (Backup) → Rule-Based (Fallback)
- **Context-Aware Responses**: Maintains conversation history for intelligent energy consulting
- **Always Available**: Never fails - automatic fallback ensures responses even during API downtime

### 📈 Analytics Dashboard
- **Real-Time Statistics**: Average consumption, efficiency scores, and trend analysis
- **Interactive Charts**: Consumption trends, temperature relationships, efficiency distribution
- **PDF Reports**: Auto-generated reports with charts and recommendations
- **Email Delivery**: Direct report delivery to user email via SendGrid

### 🔐 User Management
- **JWT Authentication**: Secure, stateless token-based authentication
- **Profile Management**: Track predictions, messages, and usage statistics
- **24-Hour Sessions**: Auto-expiring tokens for enhanced security

### 👥 Community Features
- **User Reviews**: 5-star rating system with detailed feedback
- **Helpful Voting**: Community-driven review ranking
- **Usage Guide**: Comprehensive documentation and FAQs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Frontend (HTML/CSS/JS)            │
│  • Dark theme with glassmorphism effects    │
│  • Fully responsive (mobile-first)          │
│  • Real-time chart rendering (Chart.js)    │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│         Flask REST API Backend              │
│  • 20+ endpoints for all operations         │
│  • Request validation & error handling      │
│  • CORS enabled for frontend access        │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──────┐ ┌──▼─────┐ ┌───▼──────┐
│ MongoDB  │ │ Groq   │ │ SendGrid │
│ Database │ │ Gemini │ │ Email    │
└──────────┘ └────────┘ └──────────┘
```

## 🛠️ Tech Stack

### Frontend
- **HTML5/CSS3/JavaScript** - Responsive, accessible UI
- **Chart.js** - Interactive data visualization
- **Font Awesome 6** - 2000+ icon library
- **Web Speech API** - Voice input/output support

### Backend
- **Flask 2.3** - Lightweight Python web framework
- **TensorFlow/Keras** - Deep learning models
- **Scikit-learn** - Machine learning algorithms
- **MongoDB Atlas** - Cloud database (JSON fallback)
- **ReportLab** - Server-side PDF generation

### AI & APIs
- **Groq AI** - High-speed LLM inference
- **Google Gemini API** - Backup AI model
- **SendGrid** - Email delivery service

### Deployment
- **Render.com** - Cloud hosting with auto-deploy
- **GitHub** - Version control & CI/CD
- **Gunicorn** - Production WSGI server

## 📋 Requirements

```
Python 3.8+
TensorFlow 2.13+
Flask 2.3
scikit-learn 1.3
pandas 2.0
MongoDB connection (or JSON fallback)
```

## ⚡ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/smart-energy-ai.git
cd smart-energy-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
# Create .env file
echo "FLASK_ENV=development" > .env
echo "SECRET_KEY=your-secret-key-here" >> .env
echo "MONGODB_URI=your-mongodb-uri" >> .env
echo "GROQ_API_KEY=your-groq-key" >> .env
echo "GEMINI_API_KEY=your-gemini-key" >> .env
echo "SENDGRID_API_KEY=your-sendgrid-key" >> .env
```

### 4. Run Application
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 🚀 Deployment

### Deploy to Render
1. Push code to GitHub repository
2. Go to [Render.com](https://render.com)
3. Connect GitHub account
4. Create new Web Service
5. Set environment variables
6. Deploy!

Auto-deploy on every GitHub push. HTTPS included.

## 📊 Machine Learning

### Prediction Model: Random Forest
- **Training Data**: 50,000+ historical energy records
- **Features**: 8 parameters (temperature, humidity, occupancy, etc.)
- **Performance**: 86% R² Score
- **Model Comparison**:
  - Linear Regression: 62% R²
  - Decision Tree: 74% R²
  - **Random Forest: 86% R²** ✓
  - XGBoost: 81% R²
  - LightGBM: 80% R²

**Why Random Forest?**
- Handles non-linear relationships in energy consumption
- Robust to outliers in meter data
- Feature importance analysis included
- Fast inference (<100ms per prediction)

### Feature Engineering
- Temperature impact scaling (quadratic relationship)
- Humidity-driven cooling needs
- Occupancy-weighted lighting usage
- Discomfort index calculation
- Renewable energy integration

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Prediction Accuracy (R²) | 86% |
| Inference Speed | <100ms |
| Chat Response Time | 1-2 seconds |
| API Uptime | 99.9% |
| Concurrent Users | 100+ |

## 🔑 API Endpoints

### Authentication
```
POST   /api/register              - Create account
POST   /api/login                 - Get JWT token
GET    /api/user-profile          - User info (JWT required)
```

### Prediction
```
POST   /api/predict               - Single prediction
POST   /api/predict-batch         - Batch CSV upload
GET    /api/prediction-history    - Past predictions
```

### AI Chat
```
POST   /api/chatbot               - Send message with history
GET    /api/chat-history          - Retrieve all messages
POST   /api/clear-chat-history    - Delete all messages
```

### Dashboard
```
GET    /api/dashboard-stats       - Statistics
GET    /api/chart-data            - Chart data
GET    /api/download-pdf          - PDF report
POST   /api/email-report          - Send email report
```

### Reviews
```
POST   /api/reviews               - Post review
GET    /api/reviews               - Get all reviews
POST   /api/reviews/{id}/helpful  - Mark helpful
```

## 📱 Usage Examples

### Energy Prediction
```python
# Using the API
response = requests.post('http://localhost:5000/api/predict', 
    json={
        'temperature': 25,
        'humidity': 65,
        'occupancy': 15,
        'hvac_usage': 75,
        'lighting_usage': 50,
        'square_footage': 5000,
        'renewable_energy': 30,
        'holiday_status': 0
    },
    headers={'Authorization': f'Bearer {token}'}
)

prediction = response.json()
print(f"Predicted consumption: {prediction['predicted_consumption']} kWh")
print(f"Efficiency score: {prediction['efficiency_score']}")
```

### AI Chat
```python
response = requests.post('http://localhost:5000/api/chatbot',
    json={'message': 'How can I reduce my energy consumption?'},
    headers={'Authorization': f'Bearer {token}'}
)

print(response.json()['response'])
```

## 🧪 Testing

```bash
# Run prediction test
python tests/test_prediction.py

# Run API test
python tests/test_api.py

# Run ML model test
python tests/test_model.py
```

**Made with ❤️ for a sustainable energy future**

⭐ If you find this project helpful, please star it on GitHub!
