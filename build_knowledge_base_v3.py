"""
Study Assistant - Exercise 2 (Hybrid Version - FAST)
--------------------------------------------------------
Strategy:
  1. Try fast plain-text extraction on each page first (near-instant).
  2. Only if that page has little/no extractable text (likely a slide that's
     mostly image/diagram), fall back to a lightweight vision model.

Requirements:
  - pymupdf: pip install pymupdf
  - numpy:   pip install numpy
  - Ollama running, with these models pulled:
        ollama pull moondream
        ollama pull nomic-embed-text
"""

import os
import json
import base64
import requests
import fitz

# ---- SETTINGS ----
PDF_FOLDER = r"D:\Gen_AI_Material"
OUTPUT_FILE = "knowledge_base.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

MIN_TEXT_LENGTH = 30  # if extracted text is shorter than this, use vision model instead

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
VISION_MODEL = "moondream"
EMBED_MODEL = "nomic-embed-text"
# --------------------


def pdf_page_to_base64_image(page) -> str:
    pix = page.get_pixmap(dpi=120)
    image_bytes = pix.tobytes("png")
    return base64.b64encode(image_bytes).decode("utf-8")


def read_slide_with_vision_model(base64_image: str) -> str:
    payload = {
        "model": VISION_MODEL,
        "prompt": (
            "This is a slide from a university lecture. "
            "Transcribe all readable text on it exactly. "
            "If there are diagrams or charts, briefly describe what they show. "
            "Do not add commentary, just the content of the slide."
        ),
        "images": [base64_image],
        "stream": False
    }
    response = requests.post(OLLAMA_GENERATE_URL, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Vision model request failed: {response.text}")
    return response.json().get("response", "")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += (chunk_size - overlap)
    return chunks


def get_embedding(text: str):
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(OLLAMA_EMBED_URL, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Embedding request failed: {response.text}")
    return response.json()["embedding"]


def build_knowledge_base():
    if not os.path.isdir(PDF_FOLDER):
        print(f"ERROR: Folder not found: {PDF_FOLDER}")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting to process...\n")

    knowledge_base = []
    vision_calls = 0
    text_calls = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"Reading: {pdf_file}")

        doc = fitz.open(pdf_path)
        print(f"  {len(doc)} page(s) found.")

        for page_num, page in enumerate(doc):
            extracted = page.get_text().strip()

            if len(extracted) >= MIN_TEXT_LENGTH:
                slide_text = extracted
                text_calls += 1
            else:
                b64_image = pdf_page_to_base64_image(page)
                slide_text = read_slide_with_vision_model(b64_image)
                vision_calls += 1

            if not slide_text.strip():
                print(f"  Warning: page {page_num + 1} produced no text.")
                continue

            chunks = chunk_text(slide_text)
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                knowledge_base.append({
                    "source": pdf_file,
                    "page": page_num + 1,
                    "chunk_id": i,
                    "text": chunk,
                    "embedding": embedding
                })

            print(f"  Page {page_num + 1}/{len(doc)} done.")

        doc.close()
        print(f"Finished {pdf_file}.\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f)

    print("=" * 50)
    print("Knowledge base built successfully!")
    print(f"Total chunks stored: {len(knowledge_base)}")
    print(f"Pages handled by fast text extraction: {text_calls}")
    print(f"Pages handled by vision model (fallback): {vision_calls}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    build_knowledge_base()