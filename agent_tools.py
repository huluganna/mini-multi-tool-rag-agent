
import ollama
from calculator_tool import calculate_remaining_leave

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_remaining_leave",
            "description": "Calculate remaining annual leave days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_days": {
                        "type": "integer",
                        "description": "Total annual leave days"
                    },
                    "used_days": {
                        "type": "integer",
                        "description": "Annual leave days already used"
                    }
                },
                "required": ["total_days", "used_days"]
            }
        }
    }
]

question = "I have 20 annual leave days and have used 7. How many remain?"

response = ollama.chat(
    model="llama3.2:latest",
    messages=[{"role": "user", "content": question}],
    tools=tools
)

message = response["message"]

if message.tool_calls:
    for tool_call in message.tool_calls:
        if tool_call.function.name == "calculate_remaining_leave":
            arguments = tool_call.function.arguments

            result = calculate_remaining_leave(
                int(arguments["total_days"]),
                int(arguments["used_days"])
            )

            print("Calculator result:", result)

            tool_result = {
                "role": "tool",
                "name": "calculate_remaining_leave",
                "content": str(result)
            }

            final_response = ollama.chat(
                model="llama3.2:latest",
                messages=[
                    {"role": "user", "content": question},
                    message,
                    tool_result
                ],
                tools=tools
            )

            print("Agent answer:", final_response["message"]["content"])
else:
    print(message.content)
