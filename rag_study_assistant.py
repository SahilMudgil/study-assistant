"""
Study Assistant - Exercise 3 (Retrieval + RAG)
----------------------------------------------------
Flow:
  Question -> Query Embedding -> Vector Similarity -> Relevant Chunks -> Context
  Context + Question -> Ollama -> Code Llama -> Response

Also demonstrates the difference between:
  (A) Asking Code Llama directly (no retrieval)
  (B) Asking Code Llama WITH retrieved context (RAG)

Requirements:
  - numpy: pip install numpy
  - Ollama running with: codellama, nomic-embed-text
  - knowledge_base.json already built (Exercise 2)
"""

import json
import requests
import numpy as np

KNOWLEDGE_BASE_FILE = "knowledge_base.json"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "codellama"
TOP_K = 4  # how many relevant chunks to retrieve


def load_knowledge_base():
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text: str):
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(OLLAMA_EMBED_URL, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Embedding request failed: {response.text}")
    return response.json()["embedding"]


def cosine_similarity(vec_a, vec_b):
    """
    Measures how similar two vectors are (1.0 = identical meaning, 0 = unrelated).
    This is the core of 'vector similarity' search.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunks(question: str, knowledge_base: list, top_k: int = TOP_K):
    """
    Step 1: embed the question
    Step 2: compare it against every stored chunk's embedding
    Step 3: return the top_k most similar chunks
    """
    question_embedding = get_embedding(question)

    scored_chunks = []
    for entry in knowledge_base:
        score = cosine_similarity(question_embedding, entry["embedding"])
        scored_chunks.append((score, entry))

    # Highest similarity first
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return scored_chunks[:top_k]


def build_context(top_chunks):
    """
    Turns retrieved chunks into a single text block to hand to Code Llama,
    including source/page so answers can be traced back to a slide.
    """
    context_parts = []
    for score, entry in top_chunks:
        context_parts.append(
            f"[Source: {entry['source']}, Page: {entry['page']}]\n{entry['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def ask_codellama(prompt: str) -> str:
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_GENERATE_URL, json=payload)
    if response.status_code != 200:
        return f"Error: {response.text}"
    return response.json().get("response", "No response received.")


def answer_without_rag(question: str) -> str:
    """Baseline: Code Llama answers using ONLY its own general training knowledge."""
    prompt = (
        "You are a helpful study assistant. Answer the following question.\n\n"
        f"Question: {question}"
    )
    return ask_codellama(prompt)


def answer_with_rag(question: str, knowledge_base: list) -> tuple:
    """RAG: retrieve relevant course content first, then answer using it."""
    top_chunks = retrieve_relevant_chunks(question, knowledge_base)
    context = build_context(top_chunks)

    prompt = (
        "You are a helpful study assistant for a Generative AI university course. "
        "Use the CONTEXT below - taken directly from the professor's lecture slides - "
        "to answer the question as accurately as possible. "
        "If the context does not contain the answer, say so honestly instead of guessing.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )
    answer = ask_codellama(prompt)
    return answer, top_chunks


def main():
    print("Loading knowledge base...")
    knowledge_base = load_knowledge_base()
    print(f"Loaded {len(knowledge_base)} chunks.\n")

    print("=" * 60)
    print(" Study Assistant with RAG (Exercise 3)")
    print(" Type 'quit' to exit")
    print("=" * 60)

    while True:
        question = input("\nAsk your study question: ").strip()
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not question:
            continue

        print("\n--- WITHOUT RAG (Code Llama's own general knowledge) ---")
        baseline_answer = answer_without_rag(question)
        print(baseline_answer)

        print("\n--- WITH RAG (using your professor's lecture slides) ---")
        rag_answer, sources = answer_with_rag(question, knowledge_base)
        print(rag_answer)

        print("\n--- Retrieved from ---")
        for score, entry in sources:
            print(f"  {entry['source']} (page {entry['page']}) - similarity: {score:.3f}")


if __name__ == "__main__":
    main()