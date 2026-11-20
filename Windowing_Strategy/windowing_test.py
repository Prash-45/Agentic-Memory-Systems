
import ollama

MODEL = "qwen3:8b"
WINDOW_SIZE = 4

# Toy conversation: 10 turns (20 messages)
conversation_history = [
    {"role": "user", "content": "My name is Alex."},
    {"role": "assistant", "content": "Nice to meet you, Alex."},

    {"role": "user", "content": "I am an AI student."},
    {"role": "assistant", "content": "That's interesting!"},

    {"role": "user", "content": "I study Python."},
    {"role": "assistant", "content": "Python is great for AI."},

    {"role": "user", "content": "I am learning machine learning."},
    {"role": "assistant", "content": "Great choice!"},

    {"role": "user", "content": "I built a CNN model."},
    {"role": "assistant", "content": "What dataset did you use?"},

    {"role": "user", "content": "Dogs vs Cats."},
    {"role": "assistant", "content": "That's a classic project."},

    {"role": "user", "content": "I also know PyTorch."},
    {"role": "assistant", "content": "Excellent framework for deep learning."},

    {"role": "user", "content": "I am learning MCP."},
    {"role": "assistant", "content": "MCP connects models with external tools."},

    {"role": "user", "content": "I use Ollama locally."},
    {"role": "assistant", "content": "That lets you run local models."},

    {"role": "user", "content": "What am I currently learning?"},
    {"role": "assistant", "content": "You're learning MCP."}
]

# Apply windowing: retain the latest 4 turns
windowed_history = conversation_history[-(WINDOW_SIZE * 2):]

print("Total messages:", len(conversation_history))
print("Messages retained:", len(windowed_history))

print("\n--- Retained Conversation ---")
for message in windowed_history:
    print(message["role"], ":", message["content"])

# New question 
windowed_history.append({
    "role": "user",
    "content": "What is my name?"
})

response = ollama.chat(
    model=MODEL,
    messages=windowed_history
)

print("\nAssistant:", response.message.content)