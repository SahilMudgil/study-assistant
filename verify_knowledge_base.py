"""
Verification script - Exercise 2 sanity check
-------------------------------------------------
Prints a readable sample of what got stored in knowledge_base.json,
so you can manually confirm the extraction quality before moving on.
"""

import json

with open("knowledge_base.json", "r", encoding="utf-8") as f:
    kb = json.load(f)

print(f"Total chunks in knowledge base: {len(kb)}\n")

# Show one text-path example and one vision-path example per PDF, if possible.
seen_sources = set()

print("=" * 70)
print("SAMPLE ENTRIES (first 200 characters of text shown)")
print("=" * 70)

for entry in kb:
    key = entry["source"]
    if key in seen_sources:
        continue
    seen_sources.add(key)

    print(f"\nSource: {entry['source']} | Page: {entry['page']}")
    print(f"Text preview: {entry['text'][:200]}")
    print(f"Embedding length: {len(entry['embedding'])} numbers")
    print("-" * 70)

print(f"\nShowed {len(seen_sources)} sample entries (one per PDF).")
print("Scroll up and check: does the text preview actually match what's")
print("likely on that page? Especially look for diagram-heavy pages -")
print("did we get a real description, or just a short caption?")