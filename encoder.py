
import numpy as np
from insightface.app import FaceAnalysis
import os

class Encoder:
    def __init__(self, model_name='buffalo_l', providers=None, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "insightface_models")
        os.makedirs(cache_dir, exist_ok=True)
        
        self.app = FaceAnalysis(name=model_name, providers=providers, root=cache_dir)
        self.app.prepare(ctx_id=-1, det_size=(640, 640)) 

    def get_face(self, img_rgb):
        return self.app.get(img_rgb)

    def get_embedding(self, img_rgb):
        faces = self.get_face(img_rgb)
        if not faces:
            return None
        return faces[0].embedding
