"""
Displays a clean, human-readable summary of knowledge_base.json
Useful for showing your professor what the knowledge base contains,
without dumping the raw 4MB file full of embedding numbers.
"""

import json
from collections import defaultdict

with open("knowledge_base.json", "r", encoding="utf-8") as f:
    kb = json.load(f)

print("=" * 70)
print("KNOWLEDGE BASE SUMMARY")
print("=" * 70)
print(f"Total chunks stored: {len(kb)}")
print(f"Embedding size per chunk: {len(kb[0]['embedding'])} numbers\n")

by_source = defaultdict(list)
for entry in kb:
    by_source[entry["source"]].append(entry)

print(f"Number of source PDFs: {len(by_source)}\n")
print(f"{'Source PDF':<70} {'Chunks':>8}")
print("-" * 80)
for source, entries in by_source.items():
    print(f"{source:<70} {len(entries):>8}")

print("\n" + "=" * 70)
print("SAMPLE CONTENT (first chunk of each PDF, text only)")
print("=" * 70)
for source, entries in by_source.items():
    print(f"\n--- {source} (page {entries[0]['page']}) ---")
    print(entries[0]["text"][:300])