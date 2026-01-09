from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import uuid
import json
from datetime import datetime

from encoder import Encoder
from recognizer import Recognizer
from database import FaceDatabase

# ================== CONFIG ==================
UPLOAD_DIR = "uploads"
DB_PATH = "faces.json"
HISTORY_PATH = "history.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# ================== LOAD MODEL & DATABASE ==================
encoder = Encoder(cache_dir="insightface_models")

db = FaceDatabase(DB_PATH)
db.load()

recognizer = Recognizer(threshold=1.2)
for name, emb_list in db.items():
    recognizer.add_face(name, emb_list)

# ================== HISTORY ==================
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

# ================== ROUTES ==================

# ---- FRONTEND ----
@app.route("/")
def index():
    return render_template("index.html")

# ---- REGISTER ----
@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or not file:
        return jsonify({"message": "Thiếu tên hoặc ảnh"}), 400

    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return jsonify({"message": "Không đọc được ảnh"}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = encoder.get_face(img_rgb)

    if not faces:
        return jsonify({"message": "Không tìm thấy khuôn mặt"}), 400

    db.add(name, [faces[0].embedding])
    db.save()
    recognizer.add_face(name, db.data[name])

    return jsonify({"message": f"Đăng ký thành công cho {name}"})


# ---- VERIFY ----
@app.route("/verify", methods=["POST"])
def verify():
    file = request.files.get("image")
    if not file:
        return jsonify({"status": "error", "message": "Thiếu ảnh"}), 400

    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return jsonify({"status": "error", "message": "Không đọc được ảnh"}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = encoder.get_face(img_rgb)

    results = []
    if faces:
        for face in faces:
            name = recognizer.predict(face.embedding)
            results.append({
                "name": name if name else "Unknown"
            })

    history = load_history()
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": filename,
        "results": results
    })
    save_history(history)

    return jsonify({
        "status": "success",
        "results": results
    })


# ---- HISTORY ----
@app.route("/history", methods=["GET"])
def history():
    return jsonify({
        "status": "success",
        "history": load_history()
    })


# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
