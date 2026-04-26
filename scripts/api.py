# ==========================================
# Brain Tumor Detection Web App — Flask API
# ==========================================

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt, datetime, bcrypt
from scripts.db import users_col, history_col, save_history
from scripts.model_cnn import build_model
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------
# CONFIG
# -------------------------------
SECRET_KEY = "super_secret_key_2025"  # change in production
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_finetuned_weights_20251109-214947.h5")
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
print("🧠 Loading model... please wait.")
model = build_model(input_shape=(224, 224, 3), num_classes=len(CLASS_LABELS))
model.load_weights(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------
# JWT HELPERS
# -------------------------------
def create_token(email):
    """Generate a JWT token for authenticated users."""
    payload = {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def auth_required(func):
    """Decorator for routes that require a valid JWT token."""
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Missing token"}), 401

        if token.startswith("Bearer "):
            token = token.split(" ", 1)[1]

        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# -------------------------------
# SIGNUP
# -------------------------------
@app.route("/signup", methods=["POST"])
def signup():
    """Create a new user account."""
    data = request.get_json(force=True)
    if not data or not all(k in data for k in ("name", "email", "password")):
        return jsonify({"success": False, "message": "Invalid data"}), 400

    # Check existing user
    if users_col.find_one({"email": data["email"]}):
        return jsonify({"success": False, "message": "Email already registered"}), 400

    hashed_pw = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt()).decode("utf-8")
    users_col.insert_one({
        "name": data["name"],
        "email": data["email"],
        "password": hashed_pw,
        "created_at": datetime.datetime.utcnow().isoformat()
    })

    return jsonify({"success": True, "message": "Account created successfully!"})

# -------------------------------
# LOGIN
# -------------------------------
@app.route("/login", methods=["POST"])
def login():
    """Authenticate user and issue JWT."""
    data = request.get_json(force=True)
    if not data or not all(k in data for k in ("email", "password")):
        return jsonify({"success": False, "message": "Invalid data"}), 400

    user = users_col.find_one({"email": data["email"]})
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    stored_pw = user.get("password", "")
    if not bcrypt.checkpw(data["password"].encode(), stored_pw.encode()):
        return jsonify({"success": False, "message": "Incorrect password"}), 401

    token = create_token(user["email"])
    return jsonify({"success": True, "message": "Login successful", "token": token})

# -------------------------------
# PREDICT
# -------------------------------
@app.route("/predict", methods=["POST"])
@auth_required
def predict():
    """Perform brain tumor prediction on uploaded MRI image."""
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    name = request.form.get("name", "")
    age = request.form.get("age", "")
    gender = request.form.get("gender", "")
    notes = request.form.get("notes", "")
    file = request.files["file"]

    # Convert image
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id] * 100.0)

    # Save to MongoDB
    save_history(name, age, gender, notes, file.filename, CLASS_LABELS[class_id], confidence)

    return jsonify({
        "prediction": CLASS_LABELS[class_id],
        "confidence": confidence
    })

# -------------------------------
# HISTORY
# -------------------------------
@app.route("/history", methods=["GET"])
@auth_required
def history():
    """Return all patient predictions."""
    data = list(history_col.find({}, {"_id": 0}).sort("timestamp", -1))
    return jsonify(data)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("🚀 Flask API running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
