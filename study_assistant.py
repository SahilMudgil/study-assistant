"""
Study Assistant - Exercise 1
-----------------------------
Flow: User -> Application (this script) -> API call -> Ollama -> Code Llama -> Response

Requirements:
  - Ollama installed and running
  - Code Llama pulled: ollama pull codellama
  - requests library: pip install requests
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama"


def ask_study_assistant(question: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": (
            "You are a helpful study assistant for a student. "
            "Answer the following question clearly and concisely.\n\n"
            f"Question: {question}"
        ),
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        return f"Error: Ollama returned status code {response.status_code}. " \
               f"Is Ollama running? Details: {response.text}"

    data = response.json()
    return data.get("response", "No response received.")


def main():
    print("=" * 50)
    print(" Study Assistant (Exercise 1) - powered by Code Llama")
    print(" Type 'quit' to exit")
    print("=" * 50)

    while True:
        question = input("\nAsk your study question: ").strip()

        if question.lower() in ("quit", "exit"):
            print("Goodbye! Happy studying.")
            break

        if not question:
            print("Please type a question.")
            continue

        print("\nThinking...\n")
        answer = ask_study_assistant(question)
        print("Assistant:", answer)


if __name__ == "__main__":
    main()