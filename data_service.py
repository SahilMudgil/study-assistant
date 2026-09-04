"""
Data Service - Exercise 4/5
Serves the knowledge base chunks over HTTP.
"""
from flask import Flask, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

with open("knowledge_base.json", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)

print(f"[Data Service] Loaded {len(KNOWLEDGE_BASE)} chunks into memory.")


@app.route("/chunks", methods=["GET"])
def get_chunks():
    return jsonify(KNOWLEDGE_BASE)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "chunks_loaded": len(KNOWLEDGE_BASE)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
