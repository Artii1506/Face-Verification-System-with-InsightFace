# Face Recognition System using InsightFace

## Project Overview
This project implements a **face recognition and verification system** using **InsightFace**, a state-of-the-art deep learning model for facial feature extraction. The system is built as a **full-stack application** with a Flask backend, a simple HTML/CSS/JS frontend, and JSON-based storage for embeddings and user history.

## Features
- **User Registration:** Capture a user's face and store embeddings for later verification.
- **Face Verification:** Verify identity by comparing the input face against registered embeddings.
- **History Logging:** Maintain a log of all verification attempts with timestamps and results.
- **Interactive Frontend:** Web interface for registering, verifying, and viewing verification history.
- **Efficient Embedding Storage:** Embeddings are stored instead of raw images for faster matching and lower storage requirements.

## System Architecture
1. **Frontend:** HTML, CSS, JavaScript for user interaction.
2. **Backend:** Flask app integrates InsightFace model for feature extraction.
3. **Database:** JSON files store face embeddings, user IDs, and verification history.
4. **Pipeline:** Input image → Pre-processing → Embedding extraction → Similarity computation → Verification result → Log storage.

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-insightface.git
