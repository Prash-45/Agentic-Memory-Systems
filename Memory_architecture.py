import ollama

MODEL = "qwen3:8b"

# Request 1: Giving out some information
response_1 = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "My secret project code is ORBIT-731. "
                       "Reply only with: Information received."
        }
    ],
    options={"temperature": 0}
)

print("Request 1:", response_1.message.content)


# Request 2: New fresh conversation
response_2 = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "What is my secret project code? "
                       "Answer only if you know."
        }
    ],
    options={"temperature": 0}
)

print("\nRequest 2:", response_2.message.content)
