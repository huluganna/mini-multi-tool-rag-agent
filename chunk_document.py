
from pathlib import Path

file_path = Path("documents/leave_policy.txt")
text = file_path.read_text(encoding="utf-8")

# Split the document into paragraphs
chunks = [
    paragraph.strip()
    for paragraph in text.split("\n\n")
    if paragraph.strip()
]

print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print("-" * 40)
