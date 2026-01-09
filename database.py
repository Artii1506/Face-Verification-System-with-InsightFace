# database.py
# Save & load embeddings from JSON, hỗ trợ nhiều embedding cho mỗi người

import json
import numpy as np

class FaceDatabase:
    def __init__(self, path="faces.json"):
        self.path = path
        self.data = {}  

    def load(self):
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
              
                self.data = {k: [np.array(e) for e in v] for k, v in raw.items()}
        except FileNotFoundError:
            self.data = {}

    def save(self):
       
        raw = {k: [e.tolist() for e in v] for k, v in self.data.items()}
        with open(self.path, "w") as f:
            json.dump(raw, f, indent=4)

    def add(self, name, embeddings):
    
        if name in self.data:
            self.data[name].extend(embeddings)
        else:
            self.data[name] = embeddings

    def items(self):
        return self.data.items()
