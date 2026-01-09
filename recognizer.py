# recognizer.py
# Face recognition using embeddings + L2 distance

import numpy as np

class Recognizer:
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.known_faces = {}  

    def add_face(self, name, embeddings):

        self.known_faces[name] = embeddings

    def predict(self, embedding):
        if embedding is None:
            return None

        best_dist = 1e9
        best_name = None

        for name, emb_list in self.known_faces.items():
            for emb in emb_list:
                dist = np.linalg.norm(emb - embedding)

                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        if best_dist <= self.threshold:
            return best_name
        return None
