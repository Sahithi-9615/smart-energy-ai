from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import secrets
from io import BytesIO
from datetime import datetime, timedelta

# Import our new modules
from auth import create_token, jwt_required, get_current_user
from email_service import send_report, send_simple_email
from functools import wraps
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO


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

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️ pymongo not installed - using JSON fallback")

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

# Email Configuration
EMAIL_SENDER = os.getenv('EMAIL_SENDER', 'your-email@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'your-app-password')
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# ==================== FLASK APP INITIALIZATION ====================
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

# ==================== MONGODB CONFIGURATION ====================
MONGODB_URI = os.getenv('MONGODB_URI')
USE_MONGODB = False
mongo_client = None
db = None

if MONGODB_AVAILABLE and MONGODB_URI:
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        mongo_client.admin.command('ping')
        
        db = mongo_client['smart_energy_db']
        
        # Create collections
        users_collection = db['users']
        predictions_collection = db['predictions']
        reviews_collection = db['reviews']
        
        # Create indexes
        users_collection.create_index('email', unique=True)
        predictions_collection.create_index([('user_email', 1), ('timestamp', -1)])
        reviews_collection.create_index('timestamp')
        
        USE_MONGODB = True
        print("✅ MongoDB Atlas connected successfully")
    except Exception as e:
        print(f"⚠️ MongoDB connection error: {e}")
        USE_MONGODB = False
        mongo_client = None
        db = None
else:
    print("⚠️ MongoDB not available - using JSON fallback")

# JSON fallback paths
JSON_DB_DIR = 'data'
USERS_FILE = os.path.join(JSON_DB_DIR, 'users.json')
PREDICTIONS_FILE = os.path.join(JSON_DB_DIR, 'predictions_history.json')
REVIEWS_FILE = os.path.join(JSON_DB_DIR, 'reviews.json')

# ==================== DATABASE INITIALIZATION (JSON Fallback) ====================
def init_json_databases():
    """Initialize JSON database files if they don't exist"""
    from pathlib import Path
    Path(JSON_DB_DIR).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f, indent=2)
        print(f"✅ Created {USERS_FILE}")
    
    if not os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump({}, f, indent=2)
        print(f"✅ Created {PREDICTIONS_FILE}")
    
    if not os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, 'w') as f:
            json.dump([], f, indent=2)
        print(f"✅ Created {REVIEWS_FILE}")

if CORS_AVAILABLE:
    CORS(app)

# ==================== AI CONFIGURATION ====================
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_READY = False
groq_client = None

if GROQ_AVAILABLE and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        test_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama-3.3-70b-versatile",
            max_tokens=10
        )
        GROQ_READY = True
        print(f"✅ Groq AI configured (PRIMARY)")
    except Exception as e:
        print(f"❌ Groq configuration error: {e}")
        GROQ_READY = False

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_READY = False
gemini_model = None

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_names = [
            'models/gemini-2.0-flash-exp',
            'models/gemini-exp-1206',
            'models/gemini-flash-latest',
            'models/gemini-2.0-flash',
            'models/gemini-1.5-flash',
            'gemini-pro'
        ]
        
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                test_response = gemini_model.generate_content("Hi")
                GEMINI_READY = True
                print(f"✅ Gemini AI configured with: {model_name} (BACKUP)")
                break
            except:
                continue
        
        if not GEMINI_READY:
            print("❌ All Gemini models failed")
    except Exception as e:
        print(f"❌ Gemini configuration error: {e}")

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

# ==================== USER MANAGEMENT FUNCTIONS ====================

def load_users():
    """Load users from MongoDB or JSON fallback"""
    if USE_MONGODB and db is not None:
        try:
            users_dict = {}
            for user in db.users.find():
                users_dict[user['email']] = {
                    'name': user['name'],
                    'password': user['password_hash'],
                    'created_at': user.get('created_at', '')
                }
            return users_dict
        except Exception as e:
            print(f"⚠️ Error loading users from MongoDB: {e}")
            return {}
    else:
        # JSON fallback
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"⚠️ Error loading users from JSON: {e}")
            return {}
        return {}

def save_users(users):
    """Save users to MongoDB or JSON fallback"""
    if USE_MONGODB and db is not None:
        try:
            for email, data in users.items():
                db.users.update_one(
                    {'email': email},
                    {
                        '$set': {
                            'email': email,
                            'name': data.get('name', 'Unknown'),
                            'password_hash': data.get('password', ''),
                            'created_at': data.get('created_at', datetime.now().isoformat())
                        }
                    },
                    upsert=True
                )
        except Exception as e:
            print(f"❌ Error saving users to MongoDB: {e}")
    else:
        # JSON fallback
        try:
            from pathlib import Path
            Path(JSON_DB_DIR).mkdir(parents=True, exist_ok=True)
            with open(USERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving users to JSON: {e}")

def register_user(email, name, password):
    """Register a new user"""
    email = email.lower().strip()
    
    if USE_MONGODB and db is not None:
        try:
            if db.users.find_one({'email': email}):
                return False, "User already exists"
            
            db.users.insert_one({
                'email': email,
                'name': name,
                'password_hash': generate_password_hash(password),
                'created_at': datetime.now().isoformat()
            })
            return True, "User registered successfully"
        except Exception as e:
            return False, f"Error: {e}"
    else:
        # JSON fallback
        users = load_users()
        if email in users:
            return False, "User already exists"
        
        users[email] = {
            'name': name,
            'password': generate_password_hash(password),
            'created_at': datetime.now().isoformat()
        }
        save_users(users)
        return True, "User registered successfully"

def login_user(email, password):
    """Verify user credentials"""
    email = email.lower().strip()
    
    if USE_MONGODB and db is not None:
        try:
            user = db.users.find_one({'email': email})
            if not user:
                return False, "User not found"
            
            if not check_password_hash(user['password_hash'], password):
                return False, "Invalid password"
            
            return True, user['name']
        except Exception as e:
            return False, f"Error: {e}"
    else:
        # JSON fallback
        users = load_users()
        if email not in users:
            return False, "User not found"
        
        if not check_password_hash(users[email]['password'], password):
            return False, "Invalid password"
        
        return True, users[email]['name']

# ==================== PREDICTION FUNCTIONS ====================

def load_predictions_history():
    """Load prediction history from MongoDB or JSON"""
    if USE_MONGODB and db is not None:
        try:
            history = {}
            for pred in db.predictions.find():
                user_email = pred['user_email']
                if user_email not in history:
                    history[user_email] = []
                
                history[user_email].append({
                    'timestamp': pred.get('timestamp', ''),
                    'prediction': pred['prediction'],
                    'usage_level': pred['usage_level'],
                    'efficiency_score': pred['efficiency_score'],
                    'temperature': pred.get('temperature'),
                    'occupancy': pred.get('occupancy'),
                    'hvac': pred.get('hvac'),
                    'date': pred['date']
                })
            
            return history
        except Exception as e:
            print(f"⚠️ Error loading predictions from MongoDB: {e}")
            return {}
    else:
        # JSON fallback
        try:
            if os.path.exists(PREDICTIONS_FILE):
                with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"⚠️ Error loading predictions from JSON: {e}")
            return {}
        return {}

def save_prediction(user_email, prediction_data):
    """Save a prediction to database"""
    user_email = user_email.lower().strip()
    
    if USE_MONGODB and db is not None:
        try:
            db.predictions.insert_one({
                'user_email': user_email,
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction_data['prediction'],
                'usage_level': prediction_data['usage_level'],
                'efficiency_score': prediction_data['efficiency_score'],
                'temperature': prediction_data.get('temperature'),
                'occupancy': prediction_data.get('occupancy'),
                'hvac': prediction_data.get('hvac'),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return True
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
            return False
    else:
        # JSON fallback
        try:
            history = load_predictions_history()
            
            if user_email not in history:
                history[user_email] = []
            
            prediction_entry = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction_data['prediction'],
                'usage_level': prediction_data['usage_level'],
                'efficiency_score': prediction_data['efficiency_score'],
                'temperature': prediction_data.get('temperature'),
                'occupancy': prediction_data.get('occupancy'),
                'hvac': prediction_data.get('hvac'),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            history[user_email].append(prediction_entry)
            history[user_email] = history[user_email][-50:]
            
            from pathlib import Path
            Path(JSON_DB_DIR).mkdir(parents=True, exist_ok=True)
            with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"❌ Error saving prediction to JSON: {e}")
            return False

import pytz

def convert_utc_to_ist(utc_time_str):
    """Convert UTC ISO string to IST - handles both ISO and pre-formatted dates"""
    if not utc_time_str:
        return utc_time_str
    
    try:
        utc_time_str = str(utc_time_str).strip()
        
        # Check if it's already formatted (contains spaces but no timezone info)
        # Pattern: "2026-01-21 10:30:45"
        if ' ' in utc_time_str and 'T' not in utc_time_str and '+' not in utc_time_str and 'Z' not in utc_time_str:
            # Already formatted correctly - just return it
            return utc_time_str
        
        # Parse ISO format UTC timestamp
        if 'Z' in utc_time_str:
            utc_time = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))
        elif 'T' in utc_time_str:
            # ISO format with or without timezone
            if '+' in utc_time_str or utc_time_str.endswith('Z'):
                utc_time = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))
            else:
                # No timezone info, assume UTC
                utc_time = datetime.fromisoformat(utc_time_str)
                utc_time = utc_time.replace(tzinfo=pytz.utc)
        else:
            # Not a recognized timestamp format, return as-is
            return utc_time_str
        
        # Ensure it has timezone info (UTC)
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=pytz.utc)
        
        # Convert to IST
        ist = pytz.timezone('Asia/Kolkata')
        ist_time = utc_time.astimezone(ist)
        
        # Return formatted string
        return ist_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"⚠️ Date conversion error for '{utc_time_str}': {e}")
        # If conversion fails, return original string
        return utc_time_str

# Then modify your get_user_predictions function (around line 405):
def get_user_predictions(user_email):
    """Get predictions for a specific user"""
    user_email = user_email.lower().strip()
    
    if USE_MONGODB and db is not None:
        try:
            predictions = list(db.predictions.find(
                {'user_email': user_email}
            ).sort('timestamp', -1).limit(50))
            
            return [
                {
                    'timestamp': convert_utc_to_ist(p.get('timestamp', '')),  # ← CONVERT HERE
                    'prediction': p['prediction'],
                    'usage_level': p['usage_level'],
                    'efficiency_score': p['efficiency_score'],
                    'temperature': p.get('temperature'),
                    'occupancy': p.get('occupancy'),
                    'hvac': p.get('hvac'),
                    'date': convert_utc_to_ist(p.get('date', ''))  # ← CONVERT HERE - NOW FIXED FOR ALL USERS
                }
                for p in predictions
            ]
        except Exception as e:
            print(f"❌ Error getting predictions: {e}")
            return []
    else:
        # JSON fallback - dates are already in correct format
        history = load_predictions_history()
        return history.get(user_email, [])

# ==================== REVIEW FUNCTIONS ====================

def load_reviews():
    """Load all reviews from MongoDB or JSON"""
    if USE_MONGODB and db is not None:
        try:
            reviews = list(db.reviews.find().sort('timestamp', -1))
            return [
                {
                    'user_email': r['user_email'],
                    'user_name': r['user_name'],
                    'rating': r['rating'],
                    'comment': r['comment'],
                    'timestamp': r.get('timestamp', '')
                }
                for r in reviews
            ]
        except Exception as e:
            print(f"❌ Error loading reviews: {e}")
            return []
    else:
        # JSON fallback
        try:
            if os.path.exists(REVIEWS_FILE):
                with open(REVIEWS_FILE, 'r') as f:
                    reviews = json.load(f)
                return reviews if isinstance(reviews, list) else []
        except Exception as e:
            print(f"❌ Error loading reviews from JSON: {e}")
            return []
        return []

def add_review(user_email, user_name, rating, comment):
    """Add a new review"""
    user_email = user_email.lower().strip()
    
    if USE_MONGODB and db is not None:
        try:
            db.reviews.insert_one({
                'user_email': user_email,
                'user_name': user_name,
                'rating': rating,
                'comment': comment,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return True
        except Exception as e:
            print(f"❌ Error adding review: {e}")
            return False
    else:
        # JSON fallback
        try:
            reviews = load_reviews()
            
            reviews.append({
                'user_email': user_email,
                'user_name': user_name,
                'rating': rating,
                'comment': comment,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            from pathlib import Path
            Path(JSON_DB_DIR).mkdir(parents=True, exist_ok=True)
            with open(REVIEWS_FILE, 'w') as f:
                json.dump(reviews, f, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ Error adding review to JSON: {e}")
            return False

# ==================== EMAIL FUNCTION ====================
def send_email_with_report(user_email, user_name, pdf_buffer):
    """Send email with PDF report attached"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = user_email
        msg['Subject'] = 'Your Smart Energy AI - Prediction Report'
        
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #00f0ff;">Smart Energy AI - Prediction Report</h2>
                    
                    <p>Hello {user_name},</p>
                    
                    <p>Thank you for using Smart Energy AI Platform!</p>
                    
                    <p>Please find attached your comprehensive energy prediction report. 
                    This report includes:</p>
                    
                    <ul>
                        <li>Summary statistics of your energy consumption</li>
                        <li>Recent prediction history</li>
                        <li>Usage patterns and efficiency metrics</li>
                    </ul>
                    
                    <p>Keep monitoring your energy consumption to optimize efficiency!</p>
                    
                    <p style="margin-top: 30px;">
                        <strong>Best regards,</strong><br/>
                        Smart Energy AI Team
                    </p>
                    
                    <hr style="margin-top: 30px; border: none; border-top: 1px solid #ddd;">
                    
                    <p style="font-size: 12px; color: #666;">
                        This is an automated message. Please do not reply to this email.
                    </p>
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        pdf_buffer.seek(0)
        attachment = MIMEBase('application', 'pdf')
        attachment.set_payload(pdf_buffer.read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', 
                            f'attachment; filename=energy_report_{datetime.now().strftime("%Y%m%d")}.pdf')
        msg.attach(attachment)
        
        print(f"📧 Connecting to SMTP server {SMTP_SERVER}:{SMTP_PORT}...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        
        print(f"📧 Logging in as {EMAIL_SENDER}...")
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        
        print(f"📧 Sending email to {user_email}...")
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent successfully to {user_email}")
        return True
    
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Email authentication failed: {e}")
        print("   Check that EMAIL_SENDER and EMAIL_PASSWORD are correct")
        return False
    except smtplib.SMTPException as e:
        print(f"❌ SMTP error: {e}")
        return False
    except Exception as e:
        print(f"❌ Email error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== AUTHENTICATION DECORATOR ====================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'error': 'Unauthorized. Please login again.'
            }), 401
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
        
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            return text
    
    except Exception as e:
        print(f"❌ Text extraction error: {e}")
        return None

# ==================== AI EXTRACTION FUNCTIONS ====================

def extract_with_groq(prompt):
    """Try extraction with Groq (PRIMARY)"""
    if not GROQ_READY:
        return None, "Groq not available"
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured data. Always return valid JSON only."},
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
    """Try extraction with Gemini (BACKUP)"""
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
    """Triple-layer AI extraction: Groq → Gemini → Rule-based"""
    print("\n" + "="*60)
    print("🤖 STARTING AI EXTRACTION WITH TRIPLE-LAYER FALLBACK")
    print("   NEW ORDER: Groq → Gemini → Rule-based")
    print("="*60)
    
    prompt = f"""Extract energy prediction parameters from this text and return ONLY valid JSON.

Required fields (extract from text, use defaults if not found):
- DateTime: date and time in "YYYY-MM-DDTHH:MM" format
- Temperature: number (remove any units like °C)
- Humidity: number 0-100 (remove any % symbols)
- SquareFootage: number (square footage or area)
- Occupancy: number 1-10 (number of people)
- RenewableEnergy: number (renewable energy in kWh)
- HVACUsage: must be exactly "On" or "Off"
- LightingUsage: must be exactly "On" or "Off"
- Holiday: must be exactly "Yes" or "No"

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
        
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        data = json.loads(response_text)
        print(f"✅ JSON parsed successfully with {len(data)} fields")
        
        # Normalize data
        if 'HVACUsage' in data:
            data['HVACUsage'] = data['HVACUsage'].capitalize()
        if 'LightingUsage' in data:
            data['LightingUsage'] = data['LightingUsage'].capitalize()
        elif 'LightningUsage' in data:
            data['LightingUsage'] = data['LightningUsage'].capitalize()
        if 'Holiday' in data:
            data['Holiday'] = data['Holiday'].capitalize()
        
        required = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 
                   'RenewableEnergy', 'HVACUsage', 'LightingUsage', 'Holiday']
        
        missing = [f for f in required if f not in data]
        if missing:
            print(f"❌ Missing fields: {missing}")
            return rule_based_extraction(text_content), 'fallback'
        
        print(f"✅ Extraction successful via {ai_used.upper()}")
        print("="*60 + "\n")
        return data, ai_used
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        return rule_based_extraction(text_content), 'fallback'
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return rule_based_extraction(text_content), 'fallback'

def rule_based_extraction(text_content):
    """Rule-based fallback extraction using pattern matching"""
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
    
    # Extract parameters using regex
    temp_match = re.search(r'temperature[:\s]+(\d+\.?\d*)', text_lower)
    if temp_match:
        data['Temperature'] = float(temp_match.group(1))
    
    humidity_match = re.search(r'humidity[:\s]+(\d+\.?\d*)', text_lower)
    if humidity_match:
        data['Humidity'] = float(humidity_match.group(1))
    
    sqft_match = re.search(r'square\s*footage[:\s]+(\d+)', text_lower)
    if sqft_match:
        data['SquareFootage'] = int(sqft_match.group(1))
    
    occupancy_match = re.search(r'occupancy[:\s]+(\d+)', text_lower)
    if occupancy_match:
        data['Occupancy'] = int(occupancy_match.group(1))
    
    renewable_match = re.search(r'renewable\s*energy[:\s]+(\d+\.?\d*)', text_lower)
    if renewable_match:
        data['RenewableEnergy'] = float(renewable_match.group(1))
    
    if 'hvac' in text_lower and 'on' in text_lower:
        data['HVACUsage'] = 'On'
    
    if ('lighting' in text_lower or 'lightning' in text_lower) and 'on' in text_lower:
        data['LightingUsage'] = 'On'
    
    if 'holiday' in text_lower and 'yes' in text_lower:
        data['Holiday'] = 'Yes'
    
    return data

def chat_with_groq(message, system_context):
    """Chat with Groq (PRIMARY)"""
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
    """Chat with Gemini (BACKUP)"""
    if not GEMINI_READY:
        return None, "Gemini not available"
    
    try:
        full_prompt = f"{system_context}\n\nUser message: {message}\n\nYour response:"
        response = gemini_model.generate_content(full_prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)

def get_fallback_chat_response(message):
    """Enhanced rule-based fallback chat"""
    message_lower = message.lower().strip()
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return """Hello! 👋 I'm your Smart Energy AI Assistant. I can help you:
- Predict energy consumption
- Guide you through the platform
- Answer energy-related questions

What would you like to do today?"""
    
    elif any(phrase in message_lower for phrase in ['what is this', 'what does this', 'purpose', 'about']):
        return """This is a **Smart Energy AI Platform** that helps you manage energy consumption! 

🎯 **What it does:**
- Predicts your energy usage based on 8 parameters
- Uses Machine Learning and AI
- Analyzes temperature, humidity, occupancy, and more
- Provides personalized energy-saving recommendations

💡 **You get:**
- Accurate predictions in kWh
- Usage level (Low/Normal/High)
- Efficiency score (0-100%)
- Interactive charts and analytics

Want to try a prediction?"""
    
    elif any(phrase in message_lower for phrase in ['how to use', 'how do i', 'guide me']):
        return """📖 **How to Use This Platform:**

**Option 1 - Prediction Tab:**
1. Click "Prediction" in the menu
2. Choose "Manual Entry" or "Upload File"
3. Enter 8 parameters
4. Get instant predictions!

**Option 2 - AI Chat (here!):**
1. Tell me you want a prediction
2. I'll ask for each parameter
3. You answer, I'll calculate!

Which would you like to try?"""
    
    elif any(word in message_lower for word in ['predict', 'prediction', 'calculate', 'energy']):
        return """⚡ **Let's predict your energy consumption!**

I'll need these 8 parameters:
1. 🌡️ Temperature (°C)
2. 💧 Humidity (%)
3. 🏠 Square Footage (sq ft)
4. 👥 Occupancy (number of people)
5. 🌱 Renewable Energy (kWh)
6. ❄️ HVAC Usage (On/Off)
7. 💡 Lighting Usage (On/Off)
8. 🎉 Holiday (Yes/No)

Go ahead and provide these values!"""
    
    elif any(word in message_lower for word in ['thanks', 'thank', 'appreciate']):
        return """You're very welcome! 😊 

I'm here to help you optimize your energy usage anytime!

What's next?
- Try another prediction?
- Explore the Dashboard?
- Learn energy-saving tips?"""
    
    else:
        return """I'm your Smart Energy AI Assistant! 🤖

I specialize in:
- ⚡ **Energy Predictions** - Calculate your consumption
- 📖 **Platform Help** - Guide you through features  
- 💡 **Energy Tips** - Optimize your usage

Popular commands:
- "Predict my energy"
- "How to use this platform"
- "Give me energy tips"

What would you like to know?"""

def generate_pdf_report(user_email):
    """Generate PDF report for user's energy predictions"""
    user_email = user_email.lower().strip()
    
    users = load_users()
    history = load_predictions_history()
    user_predictions = history.get(user_email, [])
    user_data = users.get(user_email, {})
    
    if not user_predictions:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#00f0ff'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#00f0ff'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    title = Paragraph("Smart Energy AI - Prediction Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    user_info = Paragraph(f"<b>User:</b> {user_data.get('name', 'N/A')}<br/>"
                          f"<b>Email:</b> {user_email}<br/>"
                          f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                          styles['Normal'])
    elements.append(user_info)
    elements.append(Spacer(1, 0.3*inch))
    
    total = len(user_predictions)
    avg_consumption = sum(p['prediction'] for p in user_predictions) / total
    avg_efficiency = sum(p['efficiency_score'] for p in user_predictions) / total
    high_count = sum(1 for p in user_predictions if p['usage_level'] == 'High')
    
    elements.append(Paragraph("Summary Statistics", heading_style))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Predictions', str(total)],
        ['Average Consumption', f'{avg_consumption:.2f} kWh'],
        ['Average Efficiency', f'{avg_efficiency:.1f}%'],
        ['High Usage Events', str(high_count)],
        ['High Usage Percentage', f'{(high_count/total*100):.1f}%']
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00f0ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 0.4*inch))
    
    elements.append(Paragraph("Recent Predictions (Last 10)", heading_style))
    
    prediction_data = [['Date', 'Consumption', 'Usage Level', 'Efficiency']]
    
    for pred in user_predictions[-10:][::-1]:
        prediction_data.append([
            pred['date'],
            f"{pred['prediction']} kWh",
            pred['usage_level'],
            f"{pred['efficiency_score']}%"
        ])
    
    prediction_table = Table(prediction_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00f0ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9)
    ]))
    
    elements.append(prediction_table)
    elements.append(Spacer(1, 0.4*inch))
    
    footer = Paragraph("<i>Generated by Smart Energy AI Platform</i>", 
                      ParagraphStyle('Footer', parent=styles['Normal'], 
                                   fontSize=8, textColor=colors.grey, alignment=TA_CENTER))
    elements.append(footer)
    
    doc.build(elements)
    
    buffer.seek(0)
    return buffer

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

def escapeHtml(text):
    from html import escape
    return escape(text)

# ==================== ROUTES ====================

@app.route('/api/download-report', methods=['GET'])
@jwt_required
def download_report():
    """Generate and download PDF report (JWT protected)"""
    try:
        user = get_current_user()
        pdf_buffer = generate_pdf_report(user['email'])
        
        if not pdf_buffer:
            return jsonify({'success': False, 'error': 'No predictions available'}), 400
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'energy_report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
        
    except Exception as e:
        print(f"Download report error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """JWT-based login"""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password required'}), 400
        
        # Verify user credentials
        success, result = login_user(email, password)
        
        if not success:
            return jsonify({'success': False, 'message': result}), 401
        
        # result is the user's name
        user_name = result
        
        # Create JWT token
        token = create_token(email, email)  # Using email as user_id
        
        return jsonify({
            'success': True,
            'token': token,
            'name': user_name,
            'email': email,
            'message': 'Login successful'
        })
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/signup', methods=['POST'])
def signup():
    """User registration"""
    try:
        data = request.json
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not all([name, email, password]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be 6+ characters'}), 400
        
        success, message = register_user(email, name, password)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message}), 400
            
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/logout', methods=['POST'])
@jwt_required
def logout():
    """Logout (client-side token deletion)"""
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/')
def home():
    """Serve the main page (no session check, handled client-side)"""
    return render_template('index.html', username='User')


@app.route('/login-page')
def login_page():
    """Serve the login page"""
    return render_template('login.html')

# ==================== FILE UPLOAD ====================

@app.route('/api/extract-from-file', methods=['POST'])
@jwt_required
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
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"✅ File saved: {filename}")
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        text_content = extract_text_from_file(file_path, file_ext)
        
        try:
            os.remove(file_path)
        except:
            pass
        
        if not text_content:
            return jsonify({'success': False, 'error': 'Could not extract text from file'}), 400
        
        print(f"✅ Text extracted ({len(text_content)} characters)")
        
        extracted_data, ai_used = extract_data_with_ai(text_content)
        
        if not extracted_data:
            return jsonify({'success': False, 'error': 'Could not extract energy parameters from file.'}), 400
        
        if 'DateTime' in extracted_data and extracted_data['DateTime']:
            extracted_data['timestamp'] = extracted_data['DateTime']
        else:
            extracted_data['timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M')
        
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
        
        if prediction_result:
            user = get_current_user()
            save_prediction(user['email'], {
                'prediction': prediction_result['prediction'],
                'usage_level': prediction_result['usage_level'],
                'efficiency_score': prediction_result['efficiency_score'],
                'temperature': extracted_data.get('Temperature'),
                'occupancy': extracted_data.get('Occupancy'),
                'hvac': extracted_data.get('HVACUsage')
            })
        
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

# ==================== API ENDPOINTS ====================

@app.route('/api/predict', methods=['POST'])
@jwt_required
def predict():
    print("\n" + "="*60)
    print("⚡ MANUAL PREDICTION REQUEST")
    print("="*60)
    
    try:
        user = get_current_user()
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
        
        save_prediction(user['email'], {
            'prediction': response['prediction'],
            'usage_level': response['usage_level'],
            'efficiency_score': response['efficiency_score'],
            'temperature': data.get('Temperature'),
            'occupancy': data.get('Occupancy'),
            'hvac': data.get('HVACUsage')
        })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/chatbot', methods=['POST'])
@jwt_required
def chatbot():
    """AI chatbot endpoint (JWT protected)"""
    try:
        message = request.json.get('message', '').strip()
        print(f"\n💬 Chatbot message: {message}")
        
        system_context = """You are the Smart Energy AI Assistant for a web platform that predicts energy consumption.

YOUR ROLE AND CAPABILITIES:

1. WEBSITE GUIDANCE - Explain how to use the platform:
   - Prediction tab: Manual entry or file upload for predictions
   - AI Chat: Conversational predictions
   - Dashboard: Visual analytics and charts
   - Reviews: User feedback section

2. WEBSITE PURPOSE - Explain what this does:
   - Uses Machine Learning to predict energy consumption
   - Analyzes: Temperature, Humidity, Occupancy, HVAC/Lighting usage, Square footage, Renewable energy, Holiday status
   - Provides: Predictions in kWh, Usage level, Efficiency score, Recommendations

3. ENERGY PREDICTIONS - Guide users through predictions:
   - When users want predictions, ask for the 8 parameters step-by-step
   - Be encouraging and helpful

RESPONSE STYLE:
- Keep responses to 2-3 sentences unless explaining features
- Be conversational and helpful
- Focus on energy and sustainability"""

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
        return jsonify({'response': response_text, 'powered_by': ai_used})
        
    except Exception as e:
        return jsonify({'response': 'Error processing request', 'powered_by': 'error'}), 500

@app.route('/api/submit-review', methods=['POST'])
@jwt_required
def submit_review():
    try:
        data = request.get_json()
        user = get_current_user()
        user_email = user['email']
        
        success = add_review(
            user_email=user_email,
            user_name=data.get('name', 'Anonymous'),
            rating=data.get('rating', 5),
            comment=data.get('comment', '')
        )
        
        if success:
            return jsonify({'success': True, 'message': 'Thank you!'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save review'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/get-reviews', methods=['GET'])
@jwt_required
def get_reviews():
    try:
        reviews = load_reviews()
        return jsonify({'success': True, 'reviews': reviews})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/charts-data', methods=['GET'])
@jwt_required
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
@jwt_required
def system_status():
    """Return current system status"""
    return jsonify({
        'success': True,
        'database': {
            'type': 'mongodb' if USE_MONGODB else 'json',
            'ready': USE_MONGODB and db is not None
        },
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

# ==================== USER PROFILE API ====================

@app.route('/api/user-profile', methods=['GET'])
@jwt_required
def get_user_profile():
    """Get user profile (JWT protected)"""
    try:
        user = get_current_user()
        users = load_users()
        
        if user['email'] in users:
            user_data = users[user['email']]
            history = load_predictions_history()
            user_predictions = history.get(user['email'], [])
            
            total_predictions = len(user_predictions)
            avg_consumption = 0
            avg_efficiency = 0
            
            if user_predictions:
                avg_consumption = sum(p['prediction'] for p in user_predictions) / total_predictions
                avg_efficiency = sum(p['efficiency_score'] for p in user_predictions) / total_predictions
            
            profile = {
                'success': True,
                'name': user_data['name'],
                'email': user['email'],
                'member_since': user_data.get('created_at', 'N/A'),
                'total_predictions': total_predictions,
                'avg_consumption': round(avg_consumption, 2),
                'avg_efficiency': round(avg_efficiency, 1)
            }
            
            return jsonify(profile)
        else:
            return jsonify({'success': False, 'error': 'User not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/prediction-history', methods=['GET'])
@jwt_required
def get_prediction_history():
    """Get user's prediction history"""
    try:
        user = get_current_user()
        user_email = user['email']
        user_predictions = get_user_predictions(user_email)
        
        return jsonify({
            'success': True,
            'predictions': user_predictions[::-1]  # Newest first
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard-stats', methods=['GET'])
@jwt_required
def get_dashboard_stats():
    """Get dashboard statistics (JWT protected)"""
    try:
        user = get_current_user()
        history = load_predictions_history()
        user_predictions = history.get(user['email'], [])
        
        if not user_predictions:
            return jsonify({
                'success': True,
                'stats': {
                    'total_predictions': 0,
                    'avg_consumption': 0,
                    'avg_efficiency': 0,
                    'total_energy_predicted': 0,
                    'high_usage_count': 0,
                    'low_usage_count': 0,
                    'trend': 'stable'
                },
                'recent': [],
                'consumption_trend': []
            })
        
        total = len(user_predictions)
        avg_consumption = sum(p['prediction'] for p in user_predictions) / total
        avg_efficiency = sum(p['efficiency_score'] for p in user_predictions) / total
        total_energy = sum(p['prediction'] for p in user_predictions)
        high_count = sum(1 for p in user_predictions if p['usage_level'] == 'High')
        low_count = total - high_count
        
        trend = 'stable'
        if len(user_predictions) >= 10:
            recent_avg = sum(p['prediction'] for p in user_predictions[-5:]) / 5
            previous_avg = sum(p['prediction'] for p in user_predictions[-10:-5]) / 5
            
            if recent_avg > previous_avg * 1.1:
                trend = 'increasing'
            elif recent_avg < previous_avg * 0.9:
                trend = 'decreasing'
        
        recent_10 = user_predictions[-10:] if len(user_predictions) >= 10 else user_predictions
        trend_data = [
            {
                'date': p['date'],
                'consumption': p['prediction']
            }
            for p in recent_10
        ]
        
        stats = {
            'success': True,
            'stats': {
                'total_predictions': total,
                'avg_consumption': round(avg_consumption, 2),
                'avg_efficiency': round(avg_efficiency, 1),
                'total_energy_predicted': round(total_energy, 2),
                'high_usage_count': high_count,
                'low_usage_count': low_count,
                'trend': trend
            },
            'recent': user_predictions[-5:][::-1],
            'consumption_trend': trend_data
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/email-report', methods=['POST'])
@jwt_required
def email_report():
    """Email PDF report to user (JWT protected)"""
    try:
        user = get_current_user()
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(user['email'])
        
        if not pdf_buffer:
            return jsonify({'success': False, 'error': 'No predictions available'}), 400
        
        # Save PDF temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_buffer.getvalue())
            tmp_path = tmp_file.name
        
        # Send email using SendGrid
        success = send_report(user['email'], tmp_path)
        
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except:
            pass
        
        if success:
            return jsonify({'success': True, 'message': 'Report sent to your email'})
        else:
            return jsonify({'success': False, 'error': 'Failed to send email'}), 500
            
    except Exception as e:
        print(f"Email report error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test-email', methods=['GET'])
def test_email():
    """Test SendGrid email"""
    from email_service import send_simple_email
    
    success = send_simple_email(
        recipient_email='your-email@gmail.com',  # Your actual email
        subject='Test Email from Smart Energy AI',
        html_content='<p>This is a test email. If you receive this, SendGrid is working!</p>'
    )
    
    if success:
        return jsonify({'success': True, 'message': 'Email sent successfully!'})
    else:
        return jsonify({'success': False, 'message': 'Email failed to send'}), 500

# ==================== APP STARTUP ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SMART ENERGY ANALYSIS SERVER")
    print("="*60)
    
    # Initialize JSON databases (always for fallback)
    init_json_databases()
    print("✅ Database initialization complete\n")
    
    # Check database type
    if USE_MONGODB and db is not None:
        print("📊 Database: MongoDB Atlas ✅")
    else:
        print("📊 Database: JSON Fallback ⚠️")
    
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
