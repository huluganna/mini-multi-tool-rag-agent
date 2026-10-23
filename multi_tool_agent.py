
import re
import ollama
import chromadb
from calculator_tool import calculate_remaining_leave


# 1. Connect to the existing ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="company_policies"
)


# 2. Tool 1: Search company policy
def search_company_policy(question):
    response = ollama.embed(
        model="nomic-embed-text:latest",
        input=question
    )

    query_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results["documents"][0] if results["documents"] else []

    if not documents:
        return "No matching policy documents were found."

    return "\n\n".join(documents)


# 3. Tool 2: Calculate remaining leave
def calculate_leave_from_question(question):
    numbers = re.findall(r"\d+", question)

    if len(numbers) < 2:
        return None

    total_days = int(numbers[0])
    used_days = int(numbers[1])

    if used_days > total_days:
        return None

    return calculate_remaining_leave(total_days, used_days)


# 4. Ask the employee for a question
question = input("Ask the company assistant: ").strip()

if not question:
    print("Please enter a question.")
    raise SystemExit


# 5. Decide which tools are needed
q = question.lower()

calculation_requested = (
    any(word in q for word in [
        "remaining", "remain", "left", "used",
        "subtract", "minus", "calculate"
    ])
    and bool(re.search(r"\d", q))
)

policy_requested = any(phrase in q for phrase in [
    "policy",
    "annual leave",
    "sick leave",
    "planned leave",
    "request leave",
    "request planned",
    "manager",
    "carry forward",
    "advance",
    "days in advance"
])

# A question with both policy and calculation intent
# will execute both tools.
if not policy_requested and not calculation_requested:
    policy_requested = True


# 6. Execute the required tools
tool_results = []

if calculation_requested:
    calculation_result = calculate_leave_from_question(question)

    if calculation_result is not None:
        numbers = re.findall(r"\d+", question)
        total_days = int(numbers[0])
        used_days = int(numbers[1])

        result_text = (
            f"Total annual leave: {total_days} days.\n"
            f"Leave used: {used_days} days.\n"
            f"Leave remaining: {calculation_result} days."
        )

        tool_results.append(("Calculator", result_text))
        print("\nTool used: calculator")
    else:
        tool_results.append((
            "Calculator",
            "Could not safely calculate. Please clarify the total and used days."
        ))

if policy_requested:
    policy_result = search_company_policy(question)
    tool_results.append(("Policy search", policy_result))
    print("Tool used: search_company_policy")


# 7. Build evidence for the final response
evidence = "\n\n".join(
    f"{tool_name} result:\n{result}"
    for tool_name, result in tool_results
)

# Display tool results so you can verify the evidence
print("\n--- Verified tool results ---")
print(evidence)



# 8. Generate the final answer safely

if calculation_requested and policy_requested:
    notice_match = re.search(
        r"at least\s+(\d+)\s+days in advance",
        policy_result,
        re.IGNORECASE
    )

    numbers = re.findall(r"\d+", question)
    total_days = int(numbers[0])
    used_days = int(numbers[1])

    remaining_days = calculate_remaining_leave(
        total_days, used_days
    )

    print("\n--- Assistant answer ---")
    print(
        f"You have {remaining_days} annual leave days remaining."
    )

    if notice_match:
        notice_days = int(notice_match.group(1))
        print(
            "Planned leave must be requested at least "
            f"{notice_days} days in advance, according "
            "to the retrieved company policy."
        )
    else:
        print(
            "I couldn't verify the required notice period "
            "from the retrieved policy."
        )

else:
    prompt = f"""
Answer the employee using the evidence below.

Use the retrieved policy for company rules.
Do not invent policy facts or confuse unrelated numbers.
If the evidence does not contain the answer, say so.

Employee question:
{question}

Verified tool results:
{evidence}
"""

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n--- Assistant answer ---")
    print(response["message"]["content"])
