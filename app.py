
import re

import chromadb
import gradio as gr
import ollama

from calculator_tool import calculate_remaining_leave


# --------------------------------------------------
# 1. Connect to the existing ChromaDB database
# --------------------------------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="company_policies"
)


# --------------------------------------------------
# 2. Search company policy using RAG
# --------------------------------------------------

def search_company_policy(question):
    """Retrieve relevant policy chunks from ChromaDB."""

    response = ollama.embed(
        model="nomic-embed-text:latest",
        input=question,
    )

    query_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    documents = (
        results["documents"][0]
        if results["documents"]
        else []
    )

    if not documents:
        return "No matching policy documents were found."

    return "\n\n".join(documents)


# --------------------------------------------------
# 3. Extract and calculate remaining annual leave
# --------------------------------------------------

def calculate_leave_from_question(question):
    """Calculate remaining leave from total and used days."""

    numbers = re.findall(r"\d+", question)

    if len(numbers) < 2:
        return None

    total_days = int(numbers[0])
    used_days = int(numbers[1])

    if total_days <= 0 or used_days < 0:
        return None

    if used_days > total_days:
        return None

    return total_days - used_days


# --------------------------------------------------
# 4. Main assistant logic
# --------------------------------------------------

def answer_question(question):

    if not question or not question.strip():
        return "Please enter a question.", "No tools used."

    question = question.strip()
    q = question.lower()

    # Detect calculation requests
    calculation_requested = (
        any(word in q for word in [
            "remaining",
            "remain",
            "left",
            "used",
            "subtract",
            "minus",
            "calculate",
        ])
        and bool(re.search(r"\d", question))
    )

    # Detect policy-related requests
    policy_requested = any(phrase in q for phrase in [
        "policy",
        "annual leave",
        "annual allowance",
        "sick leave",
        "planned leave",
        "request leave",
        "manager",
        "carry forward",
        "advance",
        "days in advance",
        "leave entitlement",
        "leave days",
    ])

    # Treat other questions as policy questions
    if not policy_requested and not calculation_requested:
        policy_requested = True

    tool_results = []
    tools_used = []
    policy_result = ""

    # Run calculator if requested
    if calculation_requested:
        remaining = calculate_leave_from_question(question)

        if remaining is not None:
            numbers = re.findall(r"\d+", question)
            total_days = int(numbers[0])
            used_days = int(numbers[1])

            tool_results.append(
                f"Total annual leave: {total_days} days.\n"
                f"Leave used: {used_days} days.\n"
                f"Leave remaining: {remaining} days."
            )

            tools_used.append("Calculator")

        else:
            tool_results.append(
                "Could not safely calculate remaining leave. "
                "Please provide the total allowance and days used."
            )

    # Retrieve policy if requested
    if policy_requested:
        policy_result = search_company_policy(question)

        tool_results.append(
            f"Retrieved company policy:\n{policy_result}"
        )

        tools_used.append("ChromaDB policy search")

    # --------------------------------------------------
    # 5. Produce the answer
    # --------------------------------------------------

    # Combined calculation + policy question
    if calculation_requested and policy_requested:

        remaining = calculate_leave_from_question(question)

        if remaining is None:
            answer = (
                "I couldn't safely calculate the remaining leave. "
                "Please clarify your total allowance and days used."
            )

        else:
            answer = (
                f"You have **{remaining} annual leave days remaining.**"
            )

            notice_match = re.search(
                r"at least\s+(\d+)\s+days in advance",
                policy_result,
                re.IGNORECASE | re.DOTALL,
            )

            if notice_match:
                notice_days = int(notice_match.group(1))

                answer += (
                    f"\n\nPlanned leave must be requested at least "
                    f"**{notice_days} days in advance**, according "
                    "to the retrieved company policy."
                )

            else:
                answer += (
                    "\n\nI couldn't verify the planned-leave notice "
                    "period in the retrieved policy."
                )

    # Policy-only question
    elif policy_requested:

        # Sick leave policy
        if "sick leave" in q:

            sick_leave_match = re.search(
                r"Employees should inform their manager when\s*"
                r"taking sick leave",
                policy_result,
                re.IGNORECASE,
            )

            if sick_leave_match:
                answer = (
                    "According to the company leave policy, "
                    "employees should inform their manager "
                    "when taking sick leave."
                )

            else:
                answer = (
                    "I couldn't verify the sick leave rule "
                    "in the retrieved policy."
                )

        # Planned leave notice period
        elif "planned leave" in q or "advance" in q:

            notice_match = re.search(
                r"at least\s+(\d+)\s+days in advance",
                policy_result,
                re.IGNORECASE | re.DOTALL,
            )

            if notice_match:
                answer = (
                    "Employees must request planned leave at least "
                    f"**{notice_match.group(1)} days in advance.**"
                )

            else:
                answer = (
                    "I couldn't verify the planned-leave notice "
                    "period in the retrieved policy."
                )

        # Carry-forward policy
        elif "carry forward" in q or "carried forward" in q:

            carry_match = re.search(
                r"Unused annual leave may be carried forward,?\s*"
                r"subject to company rules",
                policy_result,
                re.IGNORECASE | re.DOTALL,
            )

            if carry_match:
                answer = (
                    "According to the policy, unused annual leave "
                    "may be carried forward, subject to company "
                    "rules. Please consult your company's detailed leave "
                    "policy for the applicable carry-forward conditions."
                )

            else:
                answer = (
                    "I couldn't verify the carry-forward rule "
                    "in the retrieved policy."
                )

        # Annual leave allowance
        elif any(phrase in q for phrase in [
            "annual allowance",
            "annual leave",
            "leave entitlement",
            "leave days",
            "per year",
        ]):

            allowance_match = re.search(
                r"Every employee receives\s+(\d+)\s+"
                r"annual leave days per year",
                policy_result,
                re.IGNORECASE | re.DOTALL,
            )

            if allowance_match:
                answer = (
                    "According to the company leave policy, "
                    "every employee receives "
                    f"**{allowance_match.group(1)} annual leave "
                    "days per year.**"
                )

            else:
                answer = (
                    "I couldn't verify the annual leave allowance "
                    "in the retrieved policy."
                )

        # Use Llama 3.2 for other policy questions
        else:
            evidence = "\n\n".join(tool_results)

            prompt = f"""
You are a company leave-policy assistant.

Answer using only the evidence provided below.
Do not invent company rules.
If the evidence does not answer the question, say so.
Keep the answer focused on the employee's question.

Employee question:
{question}

Verified evidence:
{evidence}
"""

            response = ollama.chat(
                model="llama3.2:latest",
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            answer = response["message"]["content"]

    # Fallback
    else:
        answer = (
            "I couldn't calculate the leave days. "
            "Please provide the total annual allowance and "
            "the number of days already used."
        )

    # --------------------------------------------------
    # 6. Always define tool_summary before returning
    # --------------------------------------------------

    if tools_used:
        tool_summary = "Tools used: " + ", ".join(tools_used)
    else:
        tool_summary = "No tools used."

    return answer, tool_summary


# --------------------------------------------------
# 7. Build the Gradio interface
# --------------------------------------------------

with gr.Blocks(title="Mini Multi-Tool RAG Agent") as demo:

    gr.Markdown(
        """
        # Mini Multi-Tool RAG Agent

        Ask questions about company leave policies or calculate
        your remaining annual leave.

        **Powered by:** Ollama · Llama 3.2 · ChromaDB · Nomic Embeddings
        """
    )

    question_box = gr.Textbox(
        label="Your question",
        placeholder=(
            "Example: I have 20 days and used 6. "
            "How many remain, and how early must I request planned leave?"
        ),
        lines=3,
    )

    submit_button = gr.Button(
        "Ask Assistant",
        variant="primary",
    )

    answer_box = gr.Markdown(
        label="Assistant answer"
    )

    tools_box = gr.Textbox(
        label="Tool activity",
        interactive=False,
    )

    submit_button.click(
        fn=answer_question,
        inputs=question_box,
        outputs=[answer_box, tools_box],
    )

    question_box.submit(
        fn=answer_question,
        inputs=question_box,
        outputs=[answer_box, tools_box],
    )

    gr.Examples(
        examples=[
            ["What is the sick leave policy?"],
            ["How early must I request planned leave?"],
            ["What is the annual leave allowance?"],
            [
                "My annual allowance is 20 days and I used 6. "
                "How many remain, and how early must I request planned leave?"
            ],
            ["Can unused annual leave be carried forward?"],
        ],
        inputs=question_box,
    )


if __name__ == "__main__":
    demo.launch()
