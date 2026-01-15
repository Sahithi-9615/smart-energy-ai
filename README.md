# 🔋 Smart Energy Analysis and Prediction Platform

An advanced Machine Learning-powered web application for predicting energy consumption with device-level insights, real-time analytics, and AI-powered recommendations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🎯 **Core Features**
- **AI-Powered Predictions**: Random Forest ML model for accurate energy consumption forecasting
- **Device-Level Insights**: Detailed breakdown of energy usage by devices (HVAC, Lighting, Appliances)
- **Real-Time Analytics**: Interactive dashboards with Chart.js visualizations
- **Smart Recommendations**: Personalized suggestions to optimize energy usage
- **AI Chatbot**: Conversational interface for guided predictions (Gemini API integration ready)
- **API Integration**: RESTful API for external system integration
- **User Reviews**: Feedback system with rating capabilities

### 🎨 **UI/UX Features**
- Modern, responsive design with dynamic animations
- Animated robot assistant that waves from corner
- Gradient backgrounds and smooth transitions
- Dark theme with cyan/purple accent colors
- Mobile-friendly responsive layout

## 🏗️ Project Structure

```
smart_energy_app/
│
├── app.py                  # Flask backend application
├── randomforest_model.pkl               # Trained ML model
├── requirements.txt        # Python dependencies
├── reviews.json           # User reviews storage
│
├── templates/
│   └── index.html         # Main frontend HTML
│
└── README.md              # This file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone/Download the Project
```bash
cd smart_energy_app
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## 📱 Usage Guide

### 1️⃣ **Home Tab**
- Welcome screen with project overview
- Eye-catching hero section with animated energy visualization
- Feature highlights and call-to-action buttons

### 2️⃣ **Prediction Tab**
Enter the following parameters:
- **Date & Time**: When the prediction is for
- **Temperature**: Current temperature in Celsius
- **Humidity**: Relative humidity percentage
- **Square Footage**: Building size in square feet
- **Occupancy**: Number of people present
- **Renewable Energy**: Solar/wind generation in kWh
- **HVAC Usage**: On/Off status
- **Lighting Usage**: On/Off status
- **Holiday**: Yes/No

Click "Predict" to receive:
- Energy consumption prediction in kWh
- Usage level classification
- Efficiency score
- Comfort index
- Personalized recommendations

### 3️⃣ **AI Chat Tab**
- Interactive chatbot for guided predictions
- Natural language interface
- Step-by-step data collection
- *Note: Currently uses rule-based responses. Integrate Gemini API for advanced NLP*

### 4️⃣ **Dashboard Tab**
View comprehensive analytics:
- Energy consumption trends over time
- Device breakdown pie chart
- Temperature vs Energy correlation
- Occupancy impact analysis

### 5️⃣ **About Tab**
- Project description and objectives
- Key features overview
- Technology stack information
- Model performance metrics

### 6️⃣ **Guide Tab**
- Step-by-step usage instructions
- Best practices for accurate predictions
- Tips for energy optimization

### 7️⃣ **API Tab**
- REST API documentation
- Request/response examples
- Integration code samples

### 8️⃣ **Reviews Tab**
- Submit feedback and ratings
- View community reviews
- Star rating system (1-5 stars)

## 🔌 API Documentation

### Predict Energy Consumption

**Endpoint**: `POST /api/predict`

**Request Body**:
```json
{
  "timestamp": "2024-01-15T14:30:00",
  "Temperature": 25.5,
  "Humidity": 50,
  "SquareFootage": 1500,
  "Occupancy": 4,
  "RenewableEnergy": 10,
  "HVACUsage": "On",
  "LightingUsage": "On",
  "Holiday": "No"
}
```

**Response**:
```json
{
  "success": true,
  "prediction": 78.5,
  "usage_level": "Normal",
  "efficiency_score": 71.5,
  "peak_hour": true,
  "comfort_index": 1275.0,
  "recommendations": [
    "Consider raising thermostat by 2°C to reduce energy consumption",
    "Use natural lighting when possible with high occupancy"
  ]
}
```

### Other Endpoints
- `POST /api/chatbot` - Chatbot interactions
- `POST /api/submit-review` - Submit user reviews
- `GET /api/get-reviews` - Retrieve all reviews
- `GET /api/charts-data` - Dashboard data

## 🤖 AI Chatbot Integration (Gemini API)

To integrate Google's Gemini API for advanced chatbot functionality:

1. **Get API Key**: Obtain from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Install Google GenAI**:
```bash
pip install google-generativeai
```

3. **Update app.py**:
```python
import google.generativeai as genai

genai.configure(api_key='YOUR_API_KEY')
model = genai.GenerativeModel('gemini-pro')

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data.get('message', '')
    
    # Create context-aware prompt
    prompt = f"""You are a Smart Energy Assistant. Help users predict energy consumption.
    User message: {message}
    
    Respond helpfully and guide them through providing: temperature, humidity, occupancy, etc."""
    
    response = model.generate_content(prompt)
    return jsonify({'response': response.text})
```

## 🧪 Model Information

### Algorithm
- **Primary Model**: Gradient Boosting Regressor
- **Training Features**: 31 engineered features including:
  - Environmental: Temperature, Humidity
  - Building: Square Footage, Occupancy
  - Temporal: Hour, Day, Month, DayOfWeek
  - Behavioral: IsWeekend, IsPeakHour
  - Device Interactions: Temp_HVAC_Interaction, Occupancy_HVAC
  - Time Series: Lag features, Rolling means

### Feature Engineering
The model uses advanced feature engineering:
```python
# Interaction Features
Temp_HVAC_Interaction = Temperature × HVAC_Usage
Occupancy_HVAC = Occupancy × HVAC_Usage
HVAC_Peak_Usage = HVAC_Usage × IsPeakHour
Comfort_Index = Temperature × Humidity

# Temporal Features
Hour, Day, Month, DayOfWeek
IsWeekend, IsPeakHour

# Time Series Features
Energy_lag_1, Energy_lag_3, Energy_lag_24
Energy_roll_mean_6, Energy_roll_mean_24
```

### Performance Metrics
*(Based on your training - update with actual values)*
- R² Score: ~0.85+
- MAE: Low prediction error
- RMSE: Minimal deviation

## 🎨 Customization

### Change Color Scheme
Edit CSS variables in `templates/index.html`:
```css
:root {
    --primary: #00f0ff;      /* Main accent color */
    --secondary: #ff006e;    /* Secondary color */
    --accent: #8338ec;       /* Tertiary color */
    --dark: #0a0e27;         /* Background */
}
```

### Modify Predictions
Update feature engineering in `app.py` function `create_features()` to match your training pipeline.

### Add More Charts
Extend dashboard in the JavaScript section with additional Chart.js configurations.

## 🔒 Security Considerations

For production deployment:
1. Use environment variables for API keys
2. Implement authentication (JWT, OAuth)
3. Add rate limiting
4. Enable HTTPS
5. Sanitize user inputs
6. Use secure session management

## 📊 Sample Data Format

Example CSV format for training data:
```csv
Timestamp,Temperature,Humidity,SquareFootage,Occupancy,RenewableEnergy,HVACUsage,LightingUsage,DayOfWeek,Holiday,EnergyConsumption
2024-01-01 08:00:00,22.5,45,1500,3,8.5,On,On,Monday,No,75.2
```

## 🐛 Troubleshooting

### Model Loading Error
- Ensure `model.pkl` is in the same directory as `app.py`
- Verify the model was saved with the same scikit-learn version

### Port Already in Use
```bash
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 🚀 Deployment

### Deploy on Heroku
```bash
# Create Procfile
web: gunicorn app:app

# Add to requirements.txt
gunicorn==21.2.0

# Deploy
heroku create your-app-name
git push heroku main
```

### Deploy on PythonAnywhere
1. Upload files to PythonAnywhere
2. Create virtual environment
3. Install requirements
4. Configure WSGI file
5. Reload web app

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Enhanced ML models (LSTM, Prophet for time-series)
- Real-time data integration
- Mobile app version
- Multi-language support
- Advanced analytics features

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created with ❤️ for smart energy management

## 🙏 Acknowledgments

- Scikit-learn for ML framework
- Flask for web framework
- Chart.js for visualizations
- Google Fonts for typography
- Font Awesome for icons

## 📧 Support

For questions or issues, please create an issue in the repository or contact the development team.

---

**Note**: This is a demonstration project. For production use, implement proper security measures, data validation, and error handling.

**Happy Energy Saving! ⚡🌱**
