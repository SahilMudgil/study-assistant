"""
Test client - Exercise 4
---------------------------
Sends a question to the Application Service (the only one the user talks to)
and prints the final orchestrated answer.

Run this AFTER all 4 services are running in separate terminals.
"""

import requests

APPLICATION_SERVICE_URL = "http://localhost:5000"

while True:
    question = input("\nAsk your study question (or 'quit'): ").strip()
    if question.lower() in ("quit", "exit"):
        break
    if not question:
        continue

    response = requests.post(f"{APPLICATION_SERVICE_URL}/ask", json={"question": question})
    data = response.json()

    print("\nAnswer:")
    print(data["answer"])

    print("\nSources used:")
    for s in data["sources"]:
        print(f"  {s['source']} (page {s['page']}) - similarity: {s['similarity']:.3f}")