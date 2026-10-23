
from pathlib import Path
import ollama

# Read the policy document
text = Path("documents/leave_policy.txt").read_text(
    encoding="utf-8"
)

# Generate an embedding
response = ollama.embed(
    model="nomic-embed-text:latest",
    input=text
)

embedding = response["embeddings"][0]

print("Embedding generated successfully!")
print("Vector length:", len(embedding))
print("First 5 values:", embedding[:5])
