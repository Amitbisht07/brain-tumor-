from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["brain_tumor_ai"]

# Collections
users_col = db["users"]
history_col = db["patient_history"]

def save_history(name, age, gender, notes, image_name, prediction, confidence):
    history_col.insert_one({
        "name": name,
        "age": age,
        "gender": gender,
        "notes": notes,
        "image_name": image_name,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
