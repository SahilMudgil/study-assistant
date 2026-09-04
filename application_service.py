"""
Application Service - Exercise 4/5
The main entry point. Orchestrates Retrieval Service + LLM Service.
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:5001")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:5003")
DATA_SERVICE_URL = os.environ.get("DATA_SERVICE_URL", "http://localhost:5002")


def build_context(chunks):
    parts = [f"[Source: {c['source']}, Page: {c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"[Application Service] Received question: {question}")
    print("[Application Service] -> calling Retrieval Service...")
    retrieval_response = requests.post(f"{RETRIEVAL_SERVICE_URL}/retrieve", json={"question": question})
    top_chunks = retrieval_response.json()["results"]
    context = build_context(top_chunks)

    print("[Application Service] -> calling LLM Service...")
    prompt = (
        "You are a helpful study assistant for a Generative AI university course. "
        "Use the CONTEXT below - taken from the professor's lecture slides - "
        "to answer the question. If the context doesn't contain the answer, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    generate_response = requests.post(f"{LLM_SERVICE_URL}/generate", json={"prompt": prompt})
    answer = generate_response.json()["response"]

    print("[Application Service] -> returning answer to user.\n")
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": [{"source": c["source"], "page": c["page"], "similarity": c["similarity"]} for c in top_chunks]
    })


@app.route("/stats", methods=["GET"])
def stats():
    chunks_response = requests.get(f"{DATA_SERVICE_URL}/chunks")
    chunks = chunks_response.json()
    sources = set(c["source"] for c in chunks)

    return jsonify({
        "total_chunks": len(chunks),
        "total_sources": len(sources)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("[Application Service] Ready")
    app.run(host="0.0.0.0", port=5000)
