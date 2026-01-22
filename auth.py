import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
import os

SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
JWT_EXPIRATION_HOURS = 24

def create_token(user_id, email):
    """Create a JWT token for authenticated user"""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token

def decode_token(token):
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload, None
    except jwt.ExpiredSignatureError:
        return None, 'Token has expired'
    except jwt.InvalidTokenError:
        return None, 'Invalid token'

def jwt_required(f):
    """Decorator to protect routes with JWT authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Check for token in Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'success': False, 'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'success': False, 'error': 'Token is missing'}), 401
        
        # Decode token
        payload, error = decode_token(token)
        
        if error:
            return jsonify({'success': False, 'error': error}), 401
        
        # Add user info to request context
        request.user_id = payload['user_id']
        request.user_email = payload['email']
        
        return f(*args, **kwargs)
    
    return decorated_function

def get_current_user():
    """Get current authenticated user from request context"""
    return {
        'user_id': getattr(request, 'user_id', None),
        'email': getattr(request, 'user_email', None)
    }
