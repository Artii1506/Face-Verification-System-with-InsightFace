import cv2
import os
import numpy as np
from encoder import Encoder
from recognizer import Recognizer
from database import FaceDatabase

DATASET_DIR = "dataset"
DB_PATH = "faces.json"
MODEL_CACHE = "insightface_models"

encoder = Encoder(cache_dir=MODEL_CACHE)
db = FaceDatabase(DB_PATH)

# Nếu chưa có DB thì build
if not os.path.exists(DB_PATH):
    print("Database chưa có, hãy chạy build_dataset.py trước.")
    exit()

db.load()
print("Đã load database.")

# Tạo recognizer
recognizer = Recognizer()

# Thêm tất cả embeddings
for name, emb_list in db.items():
    recognizer.add_face(name,emb_list)

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = encoder.get_face(img_rgb)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        emb = face.embedding

        name = recognizer.predict(emb)
        label = name if name else "Unknown"
        color = (0,255,0) if name else (0,0,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
