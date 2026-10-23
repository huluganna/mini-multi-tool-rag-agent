
import ollama

response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in one simple sentence."
        }
    ]
)

print(response["message"]["content"])
