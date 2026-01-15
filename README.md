# ⚡ Smart Energy AI Platform

AI-powered energy consumption prediction system with voice interaction and real-time analytics.

---

## 🌟 Features

- **🤖 AI-Powered Predictions** - Random Forest ML model for accurate energy forecasting
- **🔄 Triple-Layer AI System** - Groq → Gemini → Rule-based fallback for 100% reliability
- **🎤 Voice Interaction** - Speak your questions, hear AI responses
- **📊 Interactive Dashboard** - Real-time charts and analytics
- **📁 Smart File Upload** - Auto-extract parameters from documents
- **💬 Intelligent Chatbot** - Get energy-saving tips and platform guidance
- **⭐ User Reviews** - Community feedback system

---

## 🏗️ Tech Stack

### **Backend**
- **Python 3.11** - Core language
- **Flask 2.3** - Web framework
- **scikit-learn** - Machine Learning
- **Random Forest** - Prediction model

### **AI Services**
- **Groq** - Primary AI (fast, reliable, higher limits) ⚡
- **Gemini** - Backup AI (excellent quality) ✨
- **Rule-based** - Fallback system (always works) 🛡️

### **Frontend**
- **HTML5/CSS3** - Modern UI
- **JavaScript** - Interactive features
- **Chart.js** - Data visualization
- **Web Speech API** - Voice features

### **Deployment**
- **Render.com** - Cloud hosting
- **GitHub** - Version control
- **Gunicorn** - WSGI server

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.11+
- Git
- Groq API key (free from console.groq.com)
- Gemini API key (optional, free from aistudio.google.com)

### **Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/smart-energy-ai.git
cd smart-energy-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_groq_key_here" > .env
echo "GEMINI_API_KEY=your_gemini_key_here" >> .env
echo "SECRET_KEY=your-secret-key-here" >> .env

# Run application
python app.py
```

Visit `http://localhost:5000`

---

## 🔑 API Keys Setup

### **Groq API Key** (Primary - Required)
1. Visit https://console.groq.com
2. Sign up (free, no credit card)
3. Create API key
4. Copy key: `gsk_xxxxx...`

### **Gemini API Key** (Backup - Optional)
1. Visit https://aistudio.google.com/app/apikey
2. Create API key
3. Copy key: `AIzaSy...`

### **Environment Variables**
Create `.env` file in project root:
```env
GROQ_API_KEY=gsk_your_groq_key_here
GEMINI_API_KEY=AIzaSy_your_gemini_key_here
SECRET_KEY=your-secret-key-for-sessions
```

---

## 🎯 Triple-Layer AI System

### **How It Works**

```
User Query/Request
    ↓
┌─────────────────────┐
│ Layer 1: Groq API   │ ⚡ Primary (Fast, reliable)
│ (llama-3.3-70b)     │
└─────────────────────┘
    ↓ (if fails)
┌─────────────────────┐
│ Layer 2: Gemini AI  │ ✨ Backup (High quality)
│ (gemini-2.0-flash)  │
└─────────────────────┘
    ↓ (if fails)
┌─────────────────────┐
│ Layer 3: Rule-Based │ 🛡️ Fallback (Always works)
│ (Pattern matching)  │
└─────────────────────┘
    ↓
Response with Badge
```

### **System Features**
- ✅ **100% Uptime** - Always responds with fallback
- ✅ **Transparent** - Shows which AI responded
- ✅ **Cost-Free** - All layers are free
- ✅ **Smart Failover** - Automatic switching
- ✅ **Production-Ready** - Enterprise reliability

### **Response Badges**
- `⚡ Groq AI` - Primary AI (most common)
- `✨ Gemini AI` - Backup AI (if Groq fails)
- `🛡️ Rule-based` - Fallback (if both APIs fail)

---

## 📊 Features Overview

### **1. Energy Predictions**
- Manual parameter entry
- File upload (PDF, TXT, DOC, CSV)
- AI-powered parameter extraction
- Real-time predictions
- Personalized recommendations

**Input Parameters:**
- Temperature (°C)
- Humidity (%)
- Square Footage (sq ft)
- Occupancy (people)
- Renewable Energy (kWh)
- HVAC Usage (On/Off)
- Lighting Usage (On/Off)
- Holiday Status (Yes/No)

**Output:**
- Energy consumption (kWh)
- Usage level (Low/Moderate/High)
- Efficiency score (0-100%)
- 5+ optimization tips
- Peak hour detection
- Comfort index

### **2. AI Chatbot**
- Natural language processing
- Energy-saving guidance
- Platform help
- Feature explanations
- 24/7 availability

**Try asking:**
- "How do I use this platform?"
- "Give me energy-saving tips"
- "Predict my energy consumption"
- "What features do you have?"

### **3. Interactive Dashboard**
Four dynamic charts:
- Energy Consumption Trend
- Device Usage Breakdown
- Temperature vs Energy
- Occupancy Impact Analysis

### **4. Voice Features**
- 🎤 **Voice Input** - Speak your questions
- 🔊 **Voice Output** - Hear AI responses
- Hands-free operation
- Multi-language support
- Browser-based (no installation)

### **5. File Upload**
Supported formats:
- PDF documents
- Text files (.txt, .csv)
- Word documents (.doc, .docx)

AI automatically extracts:
- All 8 energy parameters
- Fills prediction form
- Ready to predict instantly

---

**Built with ❤️ for sustainable energy management**

*Version 1.0.0 - January 2026*
