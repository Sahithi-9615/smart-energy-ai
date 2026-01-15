from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import secrets

# Try to import CORS
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

import pickle
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

# File processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyPDF2 not installed - PDF extraction disabled")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx not installed - DOCX extraction disabled")

# Import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai not installed")

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.permanent_session_lifetime = timedelta(hours=24)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

USERS_FILE = 'users.json'

if CORS_AVAILABLE:
    CORS(app)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_READY = False
gemini_model = None

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try different model names
        model_names = [
            'models/gemini-2.0-flash-exp',       # Experimental - HIGHER RATE LIMITS
            'models/gemini-exp-1206',             # Experimental - HIGHER RATE LIMITS  
            'models/gemini-flash-latest',         # Latest stable
            'models/gemini-2.0-flash',            # 2.0 stable
            'models/gemini-3-flash-preview',      # 3.0 preview
            'models/gemini-1.5-flash',            # 1.5 fallback
            'gemini-pro'                           # Last resort
        ]
        
        print("🔍 Testing Gemini models...")
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                test_response = gemini_model.generate_content("Hi")
                GEMINI_READY = True
                print(f"✅ Gemini AI configured with: {model_name}")
                break
            except Exception as e:
                print(f"⚠️  {model_name}: {str(e)[:100]}")
                continue
        
        if not GEMINI_READY:
            print("❌ All Gemini models failed - File upload and AI chat disabled")
            print(f"❌ Check your GEMINI_API_KEY in .env file")
    except Exception as e:
        print(f"❌ Gemini configuration error: {e}")
else:
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env file")
    if not GEMINI_AVAILABLE:
        print("❌ google-generativeai package not installed")

# Load ML model
model = None
try:
    for model_file in ['model.pkl', 'randomforest_energy_model.pkl']:
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                loaded_data = pickle.load(f)
            if hasattr(loaded_data, 'predict'):
                model = loaded_data
                print(f"✅ ML Model loaded from {model_file}")
                break
except Exception as e:
    print(f"⚠️ Model loading error: {e}")

# ==================== AUTHENTICATION ====================

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== FILE EXTRACTION ====================

def extract_text_from_file(file_path, file_ext):
    """Extract text from various file types"""
    try:
        if file_ext == 'pdf' and PDF_AVAILABLE:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        
        elif file_ext == 'docx' and DOCX_AVAILABLE:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        
        else:  # txt, csv, or fallback
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            return text
    
    except Exception as e:
        print(f"❌ Text extraction error: {e}")
        return None

def extract_data_with_gemini(text_content):
    """Use Gemini AI to extract parameters"""
    if not GEMINI_READY:
        print("❌ Gemini not ready - cannot extract")
        return None
    
    try:
        prompt = f"""Extract energy prediction parameters from this text and return ONLY valid JSON.

Required fields (extract from text, use defaults if not found):
- DateTime: date and time in "YYYY-MM-DDTHH:MM" format (if date like "11-01-2026 10:30" found, convert to "2026-01-11T10:30")
- Temperature: number (remove any units like °C)
- Humidity: number 0-100 (remove any % symbols)
- SquareFootage: number (square footage or area)
- Occupancy: number 1-10 (number of people)
- RenewableEnergy: number (renewable energy in kWh)
- HVACUsage: must be exactly "On" or "Off" (convert "on" to "On", "off" to "Off")
- LightingUsage: must be exactly "On" or "Off" (also check for "Lightning Usage" typo - treat as Lighting)
- Holiday: must be exactly "Yes" or "No" (convert "yes" to "Yes", "no" to "No")

IMPORTANT RULES:
1. Convert all text values to proper case: "on" → "On", "yes" → "Yes"
2. Handle typos: "Lightning" → "Lighting"
3. Convert date formats: "11-01-2026 10:30" → "2026-01-11T10:30"
4. Remove units: "29.2°C" → 29.2, "48.3%" → 48.3
5. If any field is missing, use these defaults:
   - Temperature: 22, Humidity: 50, SquareFootage: 1500, Occupancy: 3
   - RenewableEnergy: 5, HVACUsage: "Off", LightingUsage: "Off", Holiday: "No"

Return ONLY the JSON object. No markdown, no explanations, no extra text.

TEXT:
{text_content[:2000]}

JSON:"""

        print(f"📤 Sending to Gemini... (text length: {len(text_content)} chars)")
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        print(f"📥 Gemini response: {response_text[:300]}")
        
        # Clean response
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        try:
            data = json.loads(response_text)
            print(f"✅ Parsed data: {list(data.keys())}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"   Response was: {response_text}")
            return None
        
        # Normalize data (fix common issues)
        if 'HVACUsage' in data:
            data['HVACUsage'] = data['HVACUsage'].capitalize()
        if 'LightingUsage' in data:
            data['LightingUsage'] = data['LightingUsage'].capitalize()
        elif 'LightningUsage' in data:
            data['LightingUsage'] = data['LightningUsage'].capitalize()
            print("⚠️ Fixed typo: LightningUsage → LightingUsage")
        if 'Holiday' in data:
            data['Holiday'] = data['Holiday'].capitalize()
        
        # Validate fields
        required = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 
                   'RenewableEnergy', 'HVACUsage', 'LightingUsage', 'Holiday']
        
        missing = [f for f in required if f not in data]
        if missing:
            print(f"❌ Missing fields: {missing}")
            print(f"   Available fields: {list(data.keys())}")
            return None
        
        print(f"✅ All fields present")
        return data
        
    except Exception as e:
        print(f"❌ Gemini extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== ROUTES ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password required'}), 400
        
        users = load_users()
        
        if email in users and check_password_hash(users[email]['password'], password):
            session.permanent = True
            session['user_id'] = email
            session['username'] = users[email]['name']
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not all([name, email, password]):
        return jsonify({'success': False, 'message': 'All fields required'}), 400
    
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be 6+ characters'}), 400
    
    users = load_users()
    
    if email in users:
        return jsonify({'success': False, 'message': 'Email already registered'}), 400
    
    users[email] = {
        'name': name,
        'password': generate_password_hash(password),
        'created_at': datetime.now().isoformat()
    }
    
    save_users(users)
    return jsonify({'success': True, 'message': 'Account created'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html', username=session.get('username', 'User'))

# ==================== FILE UPLOAD ====================

@app.route('/api/extract-from-file', methods=['POST'])
@login_required
def extract_from_file():
    print("\n" + "="*60)
    print("📁 FILE UPLOAD REQUEST RECEIVED")
    print("="*60)
    
    try:
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print(f"❌ Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: PDF, TXT, DOC, DOCX, CSV'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"✅ File saved: {filename}")
        
        # Extract text
        file_ext = filename.rsplit('.', 1)[1].lower()
        text_content = extract_text_from_file(file_path, file_ext)
        
        # Clean up
        try:
            os.remove(file_path)
            print(f"✅ Temporary file deleted")
        except:
            pass
        
        if not text_content:
            print("❌ Could not extract text from file")
            return jsonify({'success': False, 'error': 'Could not extract text from file'}), 400
        
        print(f"✅ Text extracted ({len(text_content)} characters)")
        
        # Extract data with Gemini
        if not GEMINI_READY:
            print("❌ Gemini AI not available")
            return jsonify({'success': False, 'error': 'Gemini AI is not configured. Check server logs for details.'}), 400
        
        extracted_data = extract_data_with_gemini(text_content)
        
        if not extracted_data:
            print("❌ Gemini could not extract parameters")
            return jsonify({'success': False, 'error': 'Could not extract energy parameters from file. Please check the file contains: Temperature, Humidity, Square Footage, Occupancy, Renewable Energy, HVAC Usage, Lighting Usage, Holiday.'}), 400
        
        # Use timestamp from Gemini extraction if available, otherwise use current time
        if 'DateTime' in extracted_data and extracted_data['DateTime']:
            extracted_data['timestamp'] = extracted_data['DateTime']
            print(f"✅ Using timestamp from file: {extracted_data['timestamp']}")
        else:
            extracted_data['timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M')
            print(f"⚠️ No timestamp in file, using current: {extracted_data['timestamp']}")
        
        # Auto-generate prediction
        try:
            features_df = create_features(extracted_data)
            
            prediction = None
            if model is not None:
                try:
                    prediction = float(model.predict(features_df)[0])
                except Exception as e:
                    print(f"⚠️ Model prediction failed: {e}")
                    prediction = None
            
            if prediction is None:
                prediction = fallback_prediction(extracted_data)
                print(f"⚠️ Using fallback prediction")
            
            is_high_usage = prediction > 80
            efficiency_score = max(0, min(100, 100 - (prediction - 50)))
            
            # Generate recommendations
            recommendations = []
            if extracted_data['Temperature'] > 25 and extracted_data['HVACUsage'] == 'On':
                recommendations.append("Consider raising thermostat by 2°C to save energy")
            if extracted_data['Occupancy'] > 6 and extracted_data['LightingUsage'] == 'On':
                recommendations.append("Use natural lighting when possible with high occupancy")
            if prediction > 85:
                recommendations.append("Peak usage detected - consider load balancing across the day")
            if extracted_data['RenewableEnergy'] < 5:
                recommendations.append("Increase renewable energy usage to reduce costs")
            
            print(f"✅ Generated {len(recommendations)} recommendations")
            
            prediction_result = {
                'prediction': round(prediction, 2),
                'usage_level': 'High' if is_high_usage else 'Normal',
                'efficiency_score': round(efficiency_score, 1),
                'recommendations': recommendations,
                'peak_hour': int(features_df['Is_Peak_Hour'].iloc[0]) == 1,
                'comfort_index': round(float(features_df['Environmental_Stress_Level'].iloc[0]), 2)
            }
            
            print(f"✅ Prediction: {prediction_result['prediction']} kWh")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Auto-prediction error: {e}")
            import traceback
            traceback.print_exc()
            prediction_result = None
        
        return jsonify({
            'success': True,
            'data': extracted_data,
            'prediction': prediction_result,
            'message': 'Data extracted and prediction generated successfully!'
        })
        
    except Exception as e:
        print(f"❌ Extraction endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

# ==================== FEATURE ENGINEERING ====================

def create_features(input_data):
    timestamp = datetime.strptime(input_data['timestamp'], '%Y-%m-%dT%H:%M')
    hour = timestamp.hour
    
    temperature = input_data['Temperature']
    humidity = input_data['Humidity']
    occupancy = input_data['Occupancy']
    hvac_usage = 1 if input_data['HVACUsage'] == 'On' else 0
    lighting_usage = 1 if input_data['LightingUsage'] == 'On' else 0
    
    thermal_energy_load = temperature * hvac_usage * 1.5
    occupancy_energy_load = occupancy * 3.0
    environmental_stress_level = (temperature * humidity) / 100.0
    high_temp_regime = 1 if temperature > 25 else 0
    high_occupancy_regime = 1 if occupancy >= 5 else 0
    is_peak_hour = 1 if 18 <= hour <= 22 else 0
    hvac_on_peak = hvac_usage * is_peak_hour
    
    temp_bucket = 0 if temperature < 20 else (1 if temperature < 25 else 2)
    
    recent_consumption_level = 50 + (temperature - 20) * 1.5 + occupancy * 3
    load_consistency = 0.8
    daily_usage_sin = np.sin(2 * np.pi * hour / 24)
    daily_usage_cos = np.cos(2 * np.pi * hour / 24)
    load_change_1h = 0.0
    hvac_stress = hvac_usage * (abs(temperature - 22) / 10.0)
    lighting_demand_intensity = lighting_usage * occupancy * 0.5
    short_term_trend = 0.0
    
    features = pd.DataFrame([{
        'Temperature': temperature,
        'Humidity': humidity,
        'Occupancy': occupancy,
        'HVACUsage': hvac_usage,
        'LightingUsage': lighting_usage,
        'Thermal_Energy_Load': thermal_energy_load,
        'Occupancy_Energy_Load': occupancy_energy_load,
        'Environmental_Stress_Level': environmental_stress_level,
        'High_Temp_Regime': high_temp_regime,
        'High_Occupancy_Regime': high_occupancy_regime,
        'HVAC_On_Peak': hvac_on_peak,
        'Temp_Bucket': temp_bucket,
        'Recent_Consumption_Level': recent_consumption_level,
        'Load_Consistency': load_consistency,
        'Daily_Usage_Sin': daily_usage_sin,
        'Daily_Usage_Cos': daily_usage_cos,
        'Load_Change_1H': load_change_1h,
        'HVAC_Stress': hvac_stress,
        'Lighting_Demand_Intensity': lighting_demand_intensity,
        'Is_Peak_Hour': is_peak_hour,
        'Short_Term_Trend': short_term_trend
    }])
    
    return features

def fallback_prediction(data):
    base = 50.0
    temp_factor = (data['Temperature'] - 20) * 1.5
    occupancy_factor = data['Occupancy'] * 3
    sqft_factor = data['SquareFootage'] / 50
    hvac_factor = 20 if data['HVACUsage'] == 'On' else 0
    lighting_factor = 10 if data['LightingUsage'] == 'On' else 0
    renewable_offset = data['RenewableEnergy'] * -0.5
    holiday_factor = -5 if data['Holiday'] == 'Yes' else 0
    
    prediction = base + temp_factor + occupancy_factor + sqft_factor + hvac_factor + lighting_factor + renewable_offset + holiday_factor
    return float(max(30, min(150, prediction)))

# ==================== API ENDPOINTS ====================

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    print("\n" + "="*60)
    print("⚡ MANUAL PREDICTION REQUEST")
    print("="*60)
    
    try:
        data = request.json
        print(f"📊 Input data: Temp={data['Temperature']}, Humidity={data['Humidity']}, Occupancy={data['Occupancy']}")
        
        features_df = create_features(data)
        
        prediction = None
        if model is not None:
            try:
                prediction = float(model.predict(features_df)[0])
                print(f"✅ Model prediction: {prediction} kWh")
            except Exception as e:
                print(f"⚠️ Model prediction failed: {e}")
                prediction = None
        
        if prediction is None:
            prediction = fallback_prediction(data)
            print(f"✅ Fallback prediction: {prediction} kWh")
        
        is_high_usage = prediction > 80
        efficiency_score = max(0, min(100, 100 - (prediction - 50)))
        
        # Generate recommendations
        recommendations = []
        if data['Temperature'] > 25 and data['HVACUsage'] == 'On':
            recommendations.append("Consider raising thermostat by 2°C to save energy")
        if data['Occupancy'] > 6 and data['LightingUsage'] == 'On':
            recommendations.append("Use natural lighting when possible with high occupancy")
        if prediction > 85:
            recommendations.append("Peak usage detected - consider load balancing across the day")
        if data['RenewableEnergy'] < 5:
            recommendations.append("Increase renewable energy usage to reduce costs")
        
        print(f"✅ Generated {len(recommendations)} recommendations: {recommendations}")
        
        response = {
            'success': True,
            'prediction': round(prediction, 2),
            'usage_level': 'High' if is_high_usage else 'Normal',
            'efficiency_score': round(efficiency_score, 1),
            'recommendations': recommendations,
            'peak_hour': int(features_df['Is_Peak_Hour'].iloc[0]) == 1,
            'comfort_index': round(float(features_df['Environmental_Stress_Level'].iloc[0]), 2)
        }
        
        print(f"📤 Response: {response}")
        print("="*60 + "\n")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/chatbot', methods=['POST'])
@login_required
def chatbot():
    try:
        message = request.json.get('message', '').strip()
        print(f"\n💬 Chatbot message: {message}")
        
        if GEMINI_READY:
            try:
                # Comprehensive system prompt
                system_context = """You are the Smart Energy AI Assistant for a web platform that predicts energy consumption.

YOUR ROLE AND CAPABILITIES:

1. WEBSITE GUIDANCE - Explain how to use the platform:
   - Prediction tab: Manual entry or file upload for predictions
   - AI Chat: Conversational predictions (where we are now)
   - Dashboard: Visual analytics and charts
   - Reviews: User feedback section
   - About: Information about the technology

2. WEBSITE PURPOSE - Explain what this does:
   - Uses Machine Learning to predict energy consumption
   - Analyzes: Temperature, Humidity, Occupancy, HVAC/Lighting usage, Square footage, Renewable energy, Holiday status
   - Provides: Predictions in kWh, Usage level, Efficiency score, Personalized recommendations
   - Visualizes: Energy patterns, device breakdown, trends

3. ENERGY PREDICTIONS - Guide users through predictions:
   - When users want predictions, ask step-by-step for the 8 parameters
   - Be encouraging and helpful throughout the process
   - After collecting data, the system will calculate the prediction

4. HANDLE GRATITUDE & FOLLOW-UP - After predictions:
   - When users say "thank you", "thanks", "great", "awesome" after getting predictions:
     * Respond warmly and offer additional help
     * Example: "You're welcome! Happy to help you optimize your energy usage. Would you like to try another prediction or learn more about the platform's features?"
   - Keep the conversation going by offering next steps
   - Be encouraging about their energy-saving journey


5. HANDLE IRRELEVANT QUERIES - Stay on topic:
   - If asked about unrelated topics (jokes, math, Python, general knowledge), politely redirect
   - Example: "I'm specialized in energy analysis and predictions. How can I help you with energy consumption forecasting or learning about this platform?"
   - Be professional but friendly when redirecting

RESPONSE STYLE:
- Keep responses to 2-3 sentences unless explaining features
- Be conversational and helpful
- Never mention you're an AI or large language model
- Focus on energy, sustainability, and this platform

EXAMPLES OF GOOD RESPONSES:

User: "How do I use this website?"
You: "You can use the Prediction tab to enter energy parameters manually or upload a file. I can also guide you through predictions right here in the chat! The Dashboard shows visual analytics. What would you like to try?"

User: "What does this website do?"
You: "This platform uses Machine Learning to predict your energy consumption based on factors like temperature, humidity, and device usage. It provides personalized recommendations to help you save energy and reduce costs. Would you like to try a prediction?"

User: "Predict my energy"
You: "Great! Let's predict your energy consumption. First, what's the current temperature in Celsius? (For example, 25)"

User: "Tell me a joke"
You: "I'm specialized in energy analysis! How can I help you with energy predictions or understanding this platform's features?"

User: "What is Python?"
You: "I focus on energy topics. Would you like to predict energy consumption or learn how this platform works?"

Remember: Stay focused on energy, be helpful, and guide users toward using the platform effectively!"""

                full_prompt = f"""{system_context}

User message: {message}

Your response:"""
                
                response = gemini_model.generate_content(full_prompt)
                print(f"✅ Gemini response generated")
                return jsonify({'response': response.text, 'powered_by': 'gemini'})
                
            except Exception as e:
                print(f"❌ Gemini chatbot error: {e}")
                # Fallthrough to fallback
        
        # Fallback response
        print("⚠️ Using fallback response (Gemini not available)")
        return jsonify({
            'response': "I'm your Smart Energy AI Assistant! I can help you predict energy consumption or guide you through using this platform. What would you like to know?",
            'powered_by': 'fallback'
        })
    
    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return jsonify({'response': f"Error: {str(e)}"}), 400

@app.route('/api/submit-review', methods=['POST'])
@login_required
def submit_review():
    try:
        data = request.json
        reviews_file = 'reviews.json'
        
        reviews = []
        if os.path.exists(reviews_file):
            with open(reviews_file, 'r') as f:
                reviews = json.load(f)
        
        reviews.append({
            'name': data.get('name', session.get('username', 'Anonymous')),
            'rating': data.get('rating', 5),
            'comment': data.get('comment', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        with open(reviews_file, 'w') as f:
            json.dump(reviews, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Thank you!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/get-reviews', methods=['GET'])
@login_required
def get_reviews():
    try:
        reviews_file = 'reviews.json'
        if os.path.exists(reviews_file):
            with open(reviews_file, 'r') as f:
                reviews = json.load(f)
            return jsonify({'success': True, 'reviews': reviews})
        return jsonify({'success': True, 'reviews': []})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/charts-data', methods=['GET'])
@login_required
def get_charts_data():
    return jsonify({
        'energy_trend': {
            'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'data': [75, 82, 78, 85, 88, 72, 70]
        },
        'device_breakdown': {
            'labels': ['HVAC', 'Lighting', 'Appliances', 'Others'],
            'data': [45, 20, 25, 10]
        },
        'temperature_correlation': {
            'temperature': [20, 22, 24, 26, 28, 30],
            'energy': [65, 70, 75, 82, 88, 95]
        },
        'occupancy_impact': {
            'occupancy': [1, 2, 3, 4, 5, 6, 7, 8],
            'energy': [60, 65, 68, 72, 76, 80, 84, 88]
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SMART ENERGY ANALYSIS SERVER")
    print("="*60)
    print(f"✅ ML Model: {'Loaded' if model else 'Using Fallback'}")
    print(f"✅ Gemini AI: {'READY ✓' if GEMINI_READY else 'NOT AVAILABLE ✗'}")
    print(f"✅ File Upload: {'Enabled' if GEMINI_READY else 'Disabled (Gemini required)'}")
    print(f"✅ AI Chat: {'Gemini-powered' if GEMINI_READY else 'Fallback mode'}")
    if not GEMINI_READY:
        print(f"\n⚠️  To enable Gemini AI:")
        print(f"   1. Install: pip install google-generativeai")
        print(f"   2. Add GEMINI_API_KEY to .env file")
        print(f"   3. Restart server")
    print(f"\n🌐 Server running at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)