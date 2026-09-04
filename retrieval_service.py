"""
Retrieval / RAG Service - Exercise 4/5
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np

app = Flask(__name__)
CORS(app)

# Locally these default to localhost. In Docker, docker-compose sets them
# to the other services' container names.
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:5003")
DATA_SERVICE_URL = os.environ.get("DATA_SERVICE_URL", "http://localhost:5002")
TOP_K = 4


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.route("/retrieve", methods=["POST"])
def retrieve():
    question = request.json.get("question", "")

    embed_response = requests.post(f"{LLM_SERVICE_URL}/embed", json={"text": question})
    question_embedding = embed_response.json()["embedding"]

    chunks_response = requests.get(f"{DATA_SERVICE_URL}/chunks")
    all_chunks = chunks_response.json()

    scored = []
    for entry in all_chunks:
        score = cosine_similarity(question_embedding, entry["embedding"])
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:TOP_K]

    results = [
        {"source": e["source"], "page": e["page"], "text": e["text"], "similarity": float(s)}
        for s, e in top_chunks
    ]
    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"[Retrieval Service] Ready - LLM:{LLM_SERVICE_URL} Data:{DATA_SERVICE_URL}")
    app.run(host="0.0.0.0", port=5001)
