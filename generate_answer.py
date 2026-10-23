
import ollama
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="company_policies")

question = input("Ask a question about company policy: ")

# Step 1: Convert the question into an embedding
response = ollama.embed(
    model="nomic-embed-text:latest",
    input=question
)
query_embedding = response["embeddings"][0]

# Step 2: Retrieve relevant chunks from ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
retrieved_chunks = results["documents"][0]

# Step 3: Combine the retrieved chunks
context = "\n\n".join(retrieved_chunks)

# Step 4: Build the prompt
prompt = f"""
Answer the employee's question using only the policy below.
If the answer is not in the policy, say that the policy does not specify it.

Policy:
{context}

Question:
{question}
"""

# Step 5: Ask Llama 3.2 to generate an answer
response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Step 6: Print the answer
answer = response["message"]["content"]
print("\nRAG Agent Answer:")
print(answer)
