import ollama

MODEL = "qwen3:8b"


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

WINDOW_SIZES = [2, 4, 6]

question = "Which model architecture did I build?"

for n in WINDOW_SIZES:

    # Keep the latest N turns (2 messages per turn)
    windowed_history = conversation_history[-(n * 2):]

    # Add the new question
    test_history = windowed_history + [
        {"role": "user", "content": question}
    ]

    response = ollama.chat(
        model=MODEL,
        messages=test_history,
        options={"temperature": 0}
    )

    print(f"\nWindow Size: {n}")
    print("Retained messages:", len(windowed_history))
    print("Answer:", response.message.content)
    