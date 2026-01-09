import os
import cv2
import numpy as np
from encoder import Encoder
from database import FaceDatabase

DATASET_DIR = "dataset"
DB_PATH = "faces.json"

encoder = Encoder(cache_dir="insightface_models")
db = FaceDatabase(DB_PATH)
db.load()

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []
    for file in os.listdir(person_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_dir, file)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = encoder.get_face(img_rgb)

            if not faces:
                print("Không tìm thấy mặt trong ảnh:", img_path)
                continue

            embeddings.append(faces[0].embedding)

    if embeddings:
        db.add(person_name, embeddings)

db.save()
print("Database đã được xây dựng xong.")
