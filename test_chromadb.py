
from pathlib import Path
import ollama
import chromadb

# 1. Read the policy document
text = Path("documents/leave_policy.txt").read_text(
    encoding="utf-8"
)

# 2. Create a persistent ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="company_policies"
)

# 3. Generate the embedding
response = ollama.embed(
    model="nomic-embed-text:latest",
    input=text
)
embedding = response["embeddings"][0]

# 4. Store the text and its embedding
collection.upsert(
    ids=["leave-policy-001"],
    documents=[text],
    embeddings=[embedding]
)

print("Policy stored successfully!")
print("Total records:", collection.count())
