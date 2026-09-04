"""
LLM Service - Exercise 4/5
Wraps all Ollama calls. Works locally AND inside Docker.
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# When running locally: defaults to localhost.
# When running in Docker: docker-compose sets OLLAMA_HOST to host.docker.internal
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_EMBED_URL = f"{OLLAMA_HOST}/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"


@app.route("/embed", methods=["POST"])
def embed():
    text = request.json.get("text", "")
    response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    if response.status_code != 200:
        return jsonify({"error": response.text}), 500
    return jsonify({"embedding": response.json()["embedding"]})


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")

    response = requests.post(OLLAMA_GENERATE_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 150,
            "temperature": 0.2
        }
    }, timeout=180)

    if response.status_code != 200:
        return jsonify({"error": response.text}), 500

    return jsonify({"response": response.json().get("response", "")})
    if response.status_code != 200:
        return jsonify({"error": response.text}), 500
    return jsonify({"response": response.json().get("response", "")})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"[LLM Service] Ready - Ollama at {OLLAMA_HOST}")
    app.run(host="0.0.0.0", port=5003)
