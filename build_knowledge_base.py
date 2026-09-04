"""
Study Assistant - Exercise 2
------------------------------
This script builds the knowledge base for our Study Assistant.

Flow:
  PDFs -> Extract text -> Chunk text -> Embed each chunk -> Save as knowledge_base.json

Requirements:
  - pypdf:  pip install pypdf
  - numpy:  pip install numpy
  - Ollama running, with the embedding model pulled:
        ollama pull nomic-embed-text
"""

import os
import json
import requests
from pypdf import PdfReader

# ---- SETTINGS: change these if needed ----
PDF_FOLDER = r"D:\Gen_AI_Material"      # folder containing your 11 course PDFs
OUTPUT_FILE = "knowledge_base.json"      # where we save the final result
CHUNK_SIZE = 500                         # approx words per chunk
CHUNK_OVERLAP = 50                       # words shared between consecutive chunks

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
# --------------------------------------------


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads a PDF file and returns all its text as one big string.
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Splits a long piece of text into smaller overlapping chunks.

    Why overlap? So that an idea split across the boundary of two chunks
    still appears fully in at least one chunk.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += (chunk_size - overlap)

    return chunks


def get_embedding(text: str):
    """
    Sends a chunk of text to Ollama's embedding model and returns
    the resulting vector (a list of numbers representing its meaning).
    """
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    response = requests.post(OLLAMA_EMBED_URL, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Embedding request failed: {response.text}")

    return response.json()["embedding"]


def build_knowledge_base():
    if not os.path.isdir(PDF_FOLDER):
        print(f"ERROR: Folder not found: {PDF_FOLDER}")
        print("Double-check the PDF_FOLDER path at the top of this script.")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting to process...\n")

    knowledge_base = []  # will hold {text, embedding, source} entries

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"Reading: {pdf_file}")

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"  Warning: no extractable text found in {pdf_file} (might be scanned images).")
            continue

        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunk(s). Generating embeddings...")

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            knowledge_base.append({
                "source": pdf_file,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            })

        print(f"  Done with {pdf_file}.\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f)

    print("=" * 50)
    print(f"Knowledge base built successfully!")
    print(f"Total chunks stored: {len(knowledge_base)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    build_knowledge_base()