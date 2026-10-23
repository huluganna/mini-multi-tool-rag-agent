
from pathlib import Path
import ollama
import chromadb

text = Path("documents/leave_policy.txt").read_text(encoding="utf-8")

chunks = text.split("\n\n")

ids = []
documents = []
embeddings = []

for i, chunk in enumerate(chunks):
    chunk = chunk.strip()

    if not chunk:
        continue

    response = ollama.embed(
        model="nomic-embed-text:latest",
        input=chunk
    )
    embedding = response["embeddings"][0]

    ids.append(f"leave-chunk-{i}")
    documents.append(chunk)
    embeddings.append(embedding)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="company_policies")
collection.delete(ids=["leave-policy-001"])

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Total records:", collection.count())
print("Stored chunk IDs:", ids)
print("Stored chunk count:", len(ids))