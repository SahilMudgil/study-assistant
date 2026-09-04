import os, json, base64, requests, fitz
from collections import Counter

PDF_FOLDER = r"D:\Gen_AI_Material"
OUTPUT_FILE = "knowledge_base.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_TEXT_LENGTH = 30

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
VISION_MODEL = "moondream"
EMBED_MODEL = "nomic-embed-text"


def pdf_page_to_base64_image(page):
    pix = page.get_pixmap(dpi=200)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def is_repetitive_garbage(text):
    words = text.lower().split()
    if len(words) < 5:
        return False
    counts = Counter(words)
    _, count = counts.most_common(1)[0]
    return (count / len(words)) > 0.15


def read_slide_with_vision_model(base64_image):
    payload = {
        "model": VISION_MODEL,
        "prompt": (
            "This is a slide from a university lecture. Transcribe all readable "
            "text on it exactly. If there are diagrams or charts, briefly describe "
            "what they show. Do not add commentary, just the content of the slide."
        ),
        "images": [base64_image],
        "stream": False,
        "options": {"repeat_penalty": 1.3}
    }
    response = requests.post(OLLAMA_GENERATE_URL, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Vision model request failed: {response.text}")
    result = response.json().get("response", "").strip()
    if not result or is_repetitive_garbage(result):
        return "[UNCLEAR: vision model could not reliably read this slide]"
    return result


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += (chunk_size - overlap)
    return chunks


def get_embedding(text):
    response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    if response.status_code != 200:
        raise RuntimeError(f"Embedding request failed: {response.text}")
    return response.json()["embedding"]


def build_knowledge_base():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF(s).\n")

    knowledge_base = []
    text_calls = vision_calls = 0
    unclear_pages = []

    for pdf_file in pdf_files:
        doc = fitz.open(os.path.join(PDF_FOLDER, pdf_file))
        print(f"Reading: {pdf_file} ({len(doc)} pages)")

        for page_num, page in enumerate(doc):
            extracted = page.get_text().strip()
            if len(extracted) >= MIN_TEXT_LENGTH:
                slide_text = extracted
                text_calls += 1
            else:
                slide_text = read_slide_with_vision_model(pdf_page_to_base64_image(page))
                vision_calls += 1
                if slide_text.startswith("[UNCLEAR"):
                    unclear_pages.append((pdf_file, page_num + 1))

            if not slide_text.strip():
                continue

            for i, chunk in enumerate(chunk_text(slide_text)):
                knowledge_base.append({
                    "source": pdf_file, "page": page_num + 1, "chunk_id": i,
                    "text": chunk, "embedding": get_embedding(chunk)
                })
            print(f"  Page {page_num + 1}/{len(doc)} done.")
        doc.close()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f)

    print("\n" + "=" * 50)
    print(f"Total chunks: {len(knowledge_base)}")
    print(f"Fast text pages: {text_calls} | Vision pages: {vision_calls}")
    print(f"Unclear pages: {len(unclear_pages)}")
    for s, p in unclear_pages:
        print(f"  - {s}, page {p}")
    print("=" * 50)


if __name__ == "__main__":
    build_knowledge_base()