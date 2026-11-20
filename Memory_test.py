import ollama

MODEL = "qwen3:8b"

# 1. Initializing conversation memory
conversation_history = []


# 2. Fxn to chat with memory
def chat_with_memory(user_message, history):

    # Adding user's message to conversation history
    history.append({
        "role": "user",
        "content": user_message
    })

    # returning the complete conversation history to Ollama
    response = ollama.chat(
        model=MODEL,
        messages=history
    )

    # Extract the model's response
    model_response = response.message.content

    # Adding the assistant's response to history
    history.append({
        "role": "assistant",
        "content": model_response
    })

    return model_response, history


# 3. First interaction
print("\n--- First Interaction ---")

response, conversation_history = chat_with_memory(
    "Hello, my name is Alex. Nice to meet you!",
    conversation_history
)

print("Assistant:", response)


# 4. Second interaction
print("\n--- Second Interaction ---")

response, conversation_history = chat_with_memory(
    "What is my name?",
    conversation_history
)

print("Assistant:", response)


# 5. Inspect stored memory
print("\n--- Conversation History ---")

for message in conversation_history:
    print(message["role"], ":", message["content"])
