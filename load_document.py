
from pathlib import Path

# Locate the policy document
file_path = Path("documents/leave_policy.txt")

# Read the document
text = file_path.read_text(encoding="utf-8")

# Display the content
print("Policy document loaded successfully!\n")
print(text)
