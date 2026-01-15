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

# Import Groq AI (Now Primary)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed")

# Import Gemini AI (Now Backup)
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

# ==================== AI CONFIGURATION ====================

# Configure Groq AI (NOW PRIMARY)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_READY = False
groq_client = None

if GROQ_AVAILABLE and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Test Groq connection
        test_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama-3.3-70b-versatile",
            max_tokens=10
        )
        GROQ_READY = True
        print(f"✅ Groq AI configured with: llama-3.3-70b-versatile (PRIMARY)")
    except Exception as e:
        print(f"❌ Groq configuration error: {e}")
        GROQ_READY = False
else:
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env file")
    if not GROQ_AVAILABLE:
        print("❌ groq package not installed")

# Configure Gemini AI (NOW BACKUP)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_READY = False
gemini_model = None

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try different model names
        model_names = [
            'models/gemini-2.0-flash-exp',
            'models/gemini-exp-1206',
            'models/gemini-flash-latest',
            'models/gemini-2.0-flash',
            'models/gemini-3-flash-preview',
            'models/gemini-1.5-flash',
            'gemini-pro'
        ]
        
        print("🔍 Testing Gemini models...")
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                test_response = gemini_model.generate_content("Hi")
                GEMINI_READY = True
                print(f"✅ Gemini AI configured with: {model_name} (BACKUP)")
                break
            except Exception as e:
                print(f"⚠️  {model_name}: {str(e)[:100]}")
                continue
        
        if not GEMINI_READY:
            print("❌ All Gemini models failed")
    except Exception as e:
        print(f"❌ Gemini configuration error: {e}")
else:
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env file")
    if not GEMINI_AVAILABLE:
        print("❌ google-generativeai package not installed")

# AI System Status (Updated priority)
AI_STATUS = {
    'groq': GROQ_READY,
    'gemini': GEMINI_READY,
    'primary': 'groq' if GROQ_READY else ('gemini' if GEMINI_READY else 'fallback')
}

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

# ==================== AI FUNCTIONS WITH NEW FALLBACK CHAIN ====================
# NEW ORDER: Groq (Primary) → Gemini (Backup) → Rule-based Fallback

def extract_with_groq(prompt):
    """Try extraction with Groq (NOW PRIMARY)"""
    if not GROQ_READY:
        return None, "Groq not available"
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured data from text. Always return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Groq extraction failed: {error_msg[:200]}")
        return None, error_msg

def extract_with_gemini(prompt):
    """Try extraction with Gemini (NOW BACKUP)"""
    if not GEMINI_READY:
        return None, "Gemini not available"
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Gemini extraction failed: {error_msg[:200]}")
        return None, error_msg

def extract_data_with_ai(text_content):
    """
    Triple-layer AI extraction system (NEW ORDER):
    1. Try Groq (primary)
    2. Try Gemini (backup)
    3. Use rule-based fallback
    """
    print("\n" + "="*60)
    print("🤖 STARTING AI EXTRACTION WITH TRIPLE-LAYER FALLBACK")
    print("   NEW ORDER: Groq → Gemini → Rule-based")
    print("="*60)
    
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

    # Layer 1: Try Groq (PRIMARY)
    print("🟦 Layer 1: Trying Groq AI (Primary)...")
    response_text, error = extract_with_groq(prompt)
    ai_used = 'groq'
    
    # Layer 2: Try Gemini if Groq failed (BACKUP)
    if response_text is None:
        print(f"⚠️  Groq failed: {error[:100]}")
        print("🔷 Layer 2: Trying Gemini AI (Backup)...")
        response_text, error = extract_with_gemini(prompt)
        ai_used = 'gemini'
    
    # Layer 3: Rule-based fallback
    if response_text is None:
        print(f"⚠️  Gemini failed: {error[:100]}")
        print("🔶 Layer 3: Using rule-based fallback...")
        return rule_based_extraction(text_content), 'fallback'
    
    # Parse and validate AI response
    try:
        print(f"✅ {ai_used.upper()} responded, parsing JSON...")
        
        # Clean response
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        data = json.loads(response_text)
        print(f"✅ JSON parsed successfully with {len(data)} fields")
        
        # Normalize data
        if 'HVACUsage' in data:
            data['HVACUsage'] = data['HVACUsage'].capitalize()
        if 'LightingUsage' in data:
            data['LightingUsage'] = data['LightingUsage'].capitalize()
        elif 'LightningUsage' in data:
            data['LightingUsage'] = data['LightningUsage'].capitalize()
            print("⚠️ Fixed typo: LightningUsage → LightingUsage")
        if 'Holiday' in data:
            data['Holiday'] = data['Holiday'].capitalize()
        
        # Validate required fields
        required = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 
                   'RenewableEnergy', 'HVACUsage', 'LightingUsage', 'Holiday']
        
        missing = [f for f in required if f not in data]
        if missing:
            print(f"❌ Missing fields: {missing}")
            print(f"🔶 Falling back to rule-based extraction...")
            return rule_based_extraction(text_content), 'fallback'
        
        print(f"✅ Extraction successful via {ai_used.upper()}")
        print("="*60 + "\n")
        return data, ai_used
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"🔶 Falling back to rule-based extraction...")
        return rule_based_extraction(text_content), 'fallback'
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"🔶 Falling back to rule-based extraction...")
        return rule_based_extraction(text_content), 'fallback'

def rule_based_extraction(text_content):
    """
    Rule-based fallback extraction using pattern matching
    """
    import re
    
    print("🔧 Performing rule-based extraction...")
    
    data = {
        'Temperature': 22,
        'Humidity': 50,
        'SquareFootage': 1500,
        'Occupancy': 3,
        'RenewableEnergy': 5,
        'HVACUsage': 'Off',
        'LightingUsage': 'Off',
        'Holiday': 'No',
        'DateTime': datetime.now().strftime('%Y-%m-%dT%H:%M')
    }
    
    text_lower = text_content.lower()
    
    # Extract temperature
    temp_match = re.search(r'temperature[:\s]+(\d+\.?\d*)', text_lower)
    if temp_match:
        data['Temperature'] = float(temp_match.group(1))
        print(f"  ✓ Temperature: {data['Temperature']}")
    
    # Extract humidity
    humidity_match = re.search(r'humidity[:\s]+(\d+\.?\d*)', text_lower)
    if humidity_match:
        data['Humidity'] = float(humidity_match.group(1))
        print(f"  ✓ Humidity: {data['Humidity']}")
    
    # Extract square footage
    sqft_match = re.search(r'square\s*footage[:\s]+(\d+)', text_lower)
    if sqft_match:
        data['SquareFootage'] = int(sqft_match.group(1))
        print(f"  ✓ SquareFootage: {data['SquareFootage']}")
    
    # Extract occupancy
    occupancy_match = re.search(r'occupancy[:\s]+(\d+)', text_lower)
    if occupancy_match:
        data['Occupancy'] = int(occupancy_match.group(1))
        print(f"  ✓ Occupancy: {data['Occupancy']}")
    
    # Extract renewable energy
    renewable_match = re.search(r'renewable\s*energy[:\s]+(\d+\.?\d*)', text_lower)
    if renewable_match:
        data['RenewableEnergy'] = float(renewable_match.group(1))
        print(f"  ✓ RenewableEnergy: {data['RenewableEnergy']}")
    
    # Extract HVAC usage
    if 'hvac' in text_lower and 'on' in text_lower:
        data['HVACUsage'] = 'On'
        print(f"  ✓ HVACUsage: On")
    
    # Extract lighting usage (handle typo)
    if ('lighting' in text_lower or 'lightning' in text_lower) and 'on' in text_lower:
        data['LightingUsage'] = 'On'
        print(f"  ✓ LightingUsage: On")
    
    # Extract holiday
    if 'holiday' in text_lower and 'yes' in text_lower:
        data['Holiday'] = 'Yes'
        print(f"  ✓ Holiday: Yes")
    
    # Extract date/time
    date_match = re.search(r'(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2})', text_content)
    if date_match:
        try:
            dt = datetime.strptime(date_match.group(1), '%d-%m-%Y %H:%M')
            data['DateTime'] = dt.strftime('%Y-%m-%dT%H:%M')
            print(f"  ✓ DateTime: {data['DateTime']}")
        except:
            pass
    
    print(f"✅ Rule-based extraction complete")
    return data

def chat_with_groq(message, system_context):
    """Chat with Groq (NOW PRIMARY)"""
    if not GROQ_READY:
        return None, "Groq not available"
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": message}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

def chat_with_gemini(message, system_context):
    """Chat with Gemini (NOW BACKUP)"""
    if not GEMINI_READY:
        return None, "Gemini not available"
    
    try:
        full_prompt = f"{system_context}\n\nUser message: {message}\n\nYour response:"
        response = gemini_model.generate_content(full_prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)

def get_fallback_chat_response(message):
    """
    Enhanced rule-based fallback chat with diverse responses
    """
    message_lower = message.lower().strip()
    
    # ========== GREETINGS ==========
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening']):
        return """Hello! 👋 I'm your Smart Energy AI Assistant. I can help you:
- Predict energy consumption
- Guide you through the platform
- Answer energy-related questions

What would you like to do today?"""
    
    # ========== WHAT IS THIS WEBSITE / PLATFORM PURPOSE ==========
    elif any(phrase in message_lower for phrase in ['what is this', 'what does this', 'what is this website', 'what is this platform', 'purpose', 'about this']):
        return """This is a **Smart Energy AI Platform** that helps you manage energy consumption! 

🎯 **What it does:**
- Predicts your energy usage based on 8 parameters
- Uses Machine Learning (Random Forest model)
- Analyzes temperature, humidity, occupancy, and more
- Provides personalized energy-saving recommendations

💡 **You get:**
- Accurate predictions in kWh
- Usage level (Low/Normal/High)
- Efficiency score (0-100%)
- Interactive charts and analytics

Want to try a prediction?"""
    
    # ========== HOW TO USE / WEBSITE GUIDANCE ==========
    elif any(phrase in message_lower for phrase in ['how to use', 'how do i', 'guide me', 'show me how', 'help me use', 'tutorial']):
        return """📖 **How to Use This Platform:**

**Option 1 - Prediction Tab:**
1. Click "Prediction" in the menu
2. Choose "Manual Entry" or "Upload File"
3. Enter 8 parameters (temp, humidity, etc.)
4. Get instant predictions!

**Option 2 - AI Chat (here!):**
1. Tell me you want a prediction
2. I'll ask for each parameter step-by-step
3. You answer, I'll calculate!

**Option 3 - Dashboard:**
- View analytics and charts
- See energy trends

Which would you like to try?"""
    
    # ========== ENERGY PREDICTION REQUESTS ==========
    elif any(word in message_lower for word in ['predict', 'prediction', 'calculate', 'energy consumption', 'how much energy']):
        return """⚡ **Let's predict your energy consumption!**

I'll need these 8 parameters:

1. 🌡️ **Temperature** (°C)
2. 💧 **Humidity** (%)
3. 🏠 **Square Footage** (sq ft)
4. 👥 **Occupancy** (number of people)
5. 🌱 **Renewable Energy** (kWh)
6. ❄️ **HVAC Usage** (On/Off)
7. 💡 **Lighting Usage** (On/Off)
8. 🎉 **Holiday** (Yes/No)

Go ahead and provide these values, or say "manual entry" to use the form!"""
    
    # ========== THANK YOU / GRATITUDE ==========
    elif any(word in message_lower for word in ['thank', 'thanks', 'appreciate', 'helpful', 'great', 'awesome', 'perfect']):
        return """You're very welcome! 😊 

I'm here to help you optimize your energy usage anytime!

**What's next?**
- Try another prediction?
- Explore the Dashboard charts?
- Learn energy-saving tips?

Just let me know!"""
    
    # ========== ENERGY SAVING TIPS ==========
    elif any(word in message_lower for word in ['tips', 'save energy', 'reduce', 'lower', 'optimize', 'efficiency']):
        return """💡 **Energy-Saving Tips:**

**🌡️ Temperature Control:**
- Keep thermostat at 22-24°C
- Use programmable thermostats
- Close windows when HVAC is on

**💡 Lighting:**
- Switch to LED bulbs
- Use natural light
- Turn off lights when leaving rooms

**❄️ HVAC:**
- Regular maintenance
- Clean filters monthly
- Use ceiling fans

**🌱 Renewable Energy:**
- Install solar panels
- Consider battery storage

Want a personalized prediction to see your specific savings potential?"""
    
    # ========== FEATURES / CAPABILITIES ==========
    elif any(word in message_lower for word in ['feature', 'what can you', 'capabilities', 'what do you do']):
        return """🤖 **What I Can Do:**

**1. Energy Predictions** ⚡
   • Calculate consumption based on your data
   • Give efficiency scores
   • Provide recommendations

**2. Platform Guidance** 📖
   • Show you how to use features
   • Explain the technology
   • Help navigate the interface

**3. Energy Insights** 💡
   • Share energy-saving tips
   • Explain patterns
   • Answer questions

**4. File Processing** 📁
   • Upload documents with your data
   • Auto-extract parameters
   • Instant predictions

What would you like to explore?"""
    
    # ========== GOODBYE / END CONVERSATION ==========
    elif any(word in message_lower for word in ['bye', 'goodbye', 'see you', 'exit', 'quit', 'leave']):
        return """Goodbye! ⚡ Thanks for using Smart Energy AI Platform.

Remember to check your Dashboard for energy trends!

Come back anytime for predictions or energy insights. 

Have a great day! 👋"""
    
    # ========== IRRELEVANT / OFF-TOPIC QUERIES ==========
    elif any(word in message_lower for word in ['java', 'python programming', 'code', 'movie', 'weather', 'news', 'sports', 'game']):
        # Check if it's REALLY off-topic (not energy-related)
        if not any(word in message_lower for word in ['energy', 'power', 'electricity', 'consumption', 'predict', 'hvac', 'temperature']):
            return """I appreciate your question, but I specialize in **energy consumption predictions** and this platform's features! 

I can help you with:
- Energy predictions
- Platform guidance
- Energy-saving tips
- Understanding your consumption patterns

Do you have any energy-related questions I can help with?"""
    
    # ========== DEFAULT / FALLBACK ==========
    else:
        return """I'm your Smart Energy AI Assistant! 🤖

I specialize in:
- ⚡ **Energy Predictions** - Calculate your consumption
- 📖 **Platform Help** - Guide you through features  
- 💡 **Energy Tips** - Optimize your usage

**Popular commands:**
- "Predict my energy"
- "How to use this platform"
- "Give me energy tips"
- "What is this website"

What would you like to know?"""

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
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
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
            return jsonify({'success': False, 'error': 'Could not extract text from file'}), 400
        
        print(f"✅ Text extracted ({len(text_content)} characters)")
        
        # Extract data with AI (NEW ORDER: Groq → Gemini → Fallback)
        print(f"✅ Text extracted ({len(text_content)} characters)")

        extracted_data, ai_used = extract_data_with_ai(text_content)
        
        if not extracted_data:
            return jsonify({'success': False, 'error': 'Could not extract energy parameters from file.'}), 400
        
        # Use timestamp from extraction if available
        if 'DateTime' in extracted_data and extracted_data['DateTime']:
            extracted_data['timestamp'] = extracted_data['DateTime']
        else:
            extracted_data['timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M')
        
        # Auto-generate prediction
        try:
            features_df = create_features(extracted_data)
            
            prediction = None
            if model is not None:
                try:
                    prediction = float(model.predict(features_df)[0])
                except Exception as e:
                    print(f"⚠️ Model prediction failed: {e}")
            
            if prediction is None:
                prediction = fallback_prediction(extracted_data)
            
            is_high_usage = prediction > 80
            efficiency_score = max(0, min(100, 100 - (prediction - 50)))
            
            # Generate recommendations
            recommendations = []
            if extracted_data['Temperature'] > 25 and extracted_data['HVACUsage'] == 'On':
                recommendations.append("Consider raising thermostat by 2°C to save energy")
            if extracted_data['Occupancy'] > 6 and extracted_data['LightingUsage'] == 'On':
                recommendations.append("Use natural lighting when possible with high occupancy")
            if prediction > 85:
                recommendations.append("Peak usage detected - consider load balancing")
            if extracted_data['RenewableEnergy'] < 5:
                recommendations.append("Increase renewable energy usage to reduce costs")
            
            prediction_result = {
                'prediction': round(prediction, 2),
                'usage_level': 'High' if is_high_usage else 'Normal',
                'efficiency_score': round(efficiency_score, 1),
                'recommendations': recommendations,
                'peak_hour': int(features_df['Is_Peak_Hour'].iloc[0]) == 1,
                'comfort_index': round(float(features_df['Environmental_Stress_Level'].iloc[0]), 2)
            }
            
            print(f"✅ Prediction: {prediction_result['prediction']} kWh (via {ai_used})")
            
        except Exception as e:
            print(f"❌ Auto-prediction error: {e}")
            prediction_result = None
        
        return jsonify({
            'success': True,
            'data': extracted_data,
            'prediction': prediction_result,
            'ai_used': ai_used,
            'message': f'Data extracted via {ai_used.upper()} and prediction generated!'
        })
        
    except Exception as e:
        print(f"❌ Extraction error: {e}")
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
            recommendations.append("Peak usage detected - consider load balancing")
        if data['RenewableEnergy'] < 5:
            recommendations.append("Increase renewable energy usage to reduce costs")
        
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
    """
    Triple-layer chatbot system (NEW ORDER):
    1. Try Groq (primary)
    2. Try Gemini (backup)
    3. Use rule-based fallback
    """
    try:
        message = request.json.get('message', '').strip()
        print(f"\n💬 Chatbot message: {message}")
        
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

4. HANDLE GRATITUDE & FOLLOW-UP - After predictions:
   - When users say "thank you", "thanks", respond warmly and offer additional help
   - Keep the conversation going by offering next steps

5. HANDLE IRRELEVANT QUERIES - Stay on topic:
   - If asked about unrelated topics, politely redirect to energy topics
   - Be professional but friendly

RESPONSE STYLE:
- Keep responses to 2-3 sentences unless explaining features
- Be conversational and helpful
- Focus on energy, sustainability, and this platform"""

        # Layer 1: Try Groq (PRIMARY)
        print("🟦 Layer 1: Trying Groq (Primary)...")
        response_text, error = chat_with_groq(message, system_context)
        ai_used = 'groq'
        
        # Layer 2: Try Gemini if Groq failed (BACKUP)
        if response_text is None:
            print(f"⚠️  Groq failed: {error[:100]}")
            print("🔷 Layer 2: Trying Gemini (Backup)...")
            response_text, error = chat_with_gemini(message, system_context)
            ai_used = 'gemini'
        
        # Layer 3: Rule-based fallback
        if response_text is None:
            print(f"⚠️  Gemini failed: {error[:100]}")
            print("🔶 Layer 3: Using rule-based fallback...")
            response_text = get_fallback_chat_response(message)
            ai_used = 'fallback'
        
        print(f"✅ Response generated via {ai_used.upper()}")
        return jsonify({
            'response': response_text,
            'powered_by': ai_used
        })
    
    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return jsonify({
            'response': "I'm here to help with energy predictions! What would you like to know?",
            'powered_by': 'error_fallback'
        }), 200

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

@app.route('/api/system-status', methods=['GET'])
@login_required
def system_status():
    """Return current AI system status (updated priorities)"""
    return jsonify({
        'success': True,
        'ai_status': {
            'groq': {'available': GROQ_READY, 'priority': 1},
            'gemini': {'available': GEMINI_READY, 'priority': 2},
            'fallback': {'available': True, 'priority': 3},
            'primary_system': AI_STATUS['primary']
        },
        'ml_model': {'loaded': model is not None},
        'file_processing': {
            'pdf': PDF_AVAILABLE,
            'docx': DOCX_AVAILABLE
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SMART ENERGY ANALYSIS SERVER")
    print("="*60)
    print(f"✅ ML Model: {'Loaded' if model else 'Using Fallback'}")
    print(f"\n🤖 AI SYSTEMS (NEW ORDER):")
    print(f"   {'✅' if GROQ_READY else '❌'} Groq AI (Primary): {'READY' if GROQ_READY else 'NOT AVAILABLE'}")
    print(f"   {'✅' if GEMINI_READY else '❌'} Gemini AI (Backup): {'READY' if GEMINI_READY else 'NOT AVAILABLE'}")
    print(f"   ✅ Rule-based Fallback: READY")
    print(f"\n📊 ACTIVE SYSTEM: {AI_STATUS['primary'].upper()}")
    
    if not (GROQ_READY or GEMINI_READY):
        print(f"\n⚠️  WARNING: No AI systems available!")
        print(f"   To enable AI features:")
        print(f"   1. For Groq (Primary): pip install groq")
        print(f"      Add GROQ_API_KEY to .env")
        print(f"   2. For Gemini (Backup): pip install google-generativeai")
        print(f"      Add GEMINI_API_KEY to .env")
    
    print(f"\n📁 FILE PROCESSING:")
    print(f"   {'✅' if PDF_AVAILABLE else '❌'} PDF Support")
    print(f"   {'✅' if DOCX_AVAILABLE else '❌'} DOCX Support")
    
    print(f"\n🌐 Server running at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
