
import ollama
import chromadb

# Connect to our existing database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="company_policies"
)

# Ask a question using different wording
question = "What should an employee do when taking sick leave?"

# Convert the question into an embedding
response = ollama.embed(
    model="nomic-embed-text:latest",
    input=question
)
query_embedding = response["embeddings"][0]

# Search for the most relevant policy
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("Retrieved policy:")
print("Number of retrieved documents:", len(results["documents"][0]))
for document in results["documents"][0]:
    print(document)
    print("-" * 40)