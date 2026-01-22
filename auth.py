import jwt
from datetime import datetime, timedelta
from flask import request, jsonify, current_app

def create_token(user_id, email):
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, current_app.config["JWT_SECRET_KEY"], algorithm="HS256")

def jwt_required():
    header = request.headers.get("Authorization")

    if not header or not header.startswith("Bearer "):
        return None, jsonify({"success": False, "error": "Token missing"}), 401

    token = header.split(" ")[1]

    try:
        data = jwt.decode(
            token,
            current_app.config["JWT_SECRET_KEY"],
            algorithms=["HS256"]
        )
        return data, None, None
    except jwt.ExpiredSignatureError:
        return None, jsonify({"success": False, "error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return None, jsonify({"success": False, "error": "Invalid token"}), 401
