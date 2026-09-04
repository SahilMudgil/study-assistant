"""
Verification script v2 - inspect ONE specific PDF's extracted content in full.
Useful for spotting missing/garbled pages before moving to Exercise 3.
"""

import json

TARGET_SOURCE = "Gen-AI_n_LLMs_Lecture_10_GenAI_LifeCycle_Challenges_Compute_Optimal.pdf"

with open("knowledge_base.json", "r", encoding="utf-8") as f:
    kb = json.load(f)

entries = [e for e in kb if e["source"] == TARGET_SOURCE]
entries.sort(key=lambda e: e["page"])

print(f"Found {len(entries)} chunk(s) for: {TARGET_SOURCE}\n")

pages_present = sorted(set(e["page"] for e in entries))
print(f"Pages with stored content: {pages_present}\n")

print("=" * 70)
for e in entries:
    print(f"\n--- Page {e['page']} (chunk {e['chunk_id']}) ---")
    print(e["text"][:500])
    print("-" * 70)