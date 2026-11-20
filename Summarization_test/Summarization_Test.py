
import ollama

MODEL = "qwen3:8b"
RECENT_TURNS = 5

#conversation (30 messages)
conversation_history = [
    {"role": "user", "content": "I am building a backend application."},
    {"role": "assistant", "content": "What technology are you considering?"},

    {"role": "user", "content": "I want to use FastAPI."},
    {"role": "assistant", "content": "FastAPI is a good choice for APIs."},

    {"role": "user", "content": "I have decided to use PostgreSQL."},
    {"role": "assistant", "content": "PostgreSQL will work well as your database."},

    {"role": "user", "content": "I will use SQLAlchemy as my ORM."},
    {"role": "assistant", "content": "SQLAlchemy can manage database operations."},

    {"role": "user", "content": "Our team has four members."},
    {"role": "assistant", "content": "You can divide the backend work among them."},

    {"role": "user", "content": "Our project deadline is March."},
    {"role": "assistant", "content": "Let's plan milestones before then."},

    {"role": "user", "content": "We will implement JWT authentication."},
    {"role": "assistant", "content": "JWT can provide stateless authentication."},

    {"role": "user", "content": "JWT tokens should expire after 15 minutes."},
    {"role": "assistant", "content": "We can configure a 15-minute expiry."},

    {"role": "user", "content": "I also want role-based access control."},
    {"role": "assistant", "content": "We can define roles and permissions."},

    {"role": "user", "content": "I will create separate API routers."},
    {"role": "assistant", "content": "That will keep the API organized."},

    {"role": "user", "content": "I want to write unit tests."},
    {"role": "assistant", "content": "Pytest is suitable for that."},

    {"role": "user", "content": "I will use GitHub for version control."},
    {"role": "assistant", "content": "You can manage branches and pull requests."},

    {"role": "user", "content": "I also need API documentation."},
    {"role": "assistant", "content": "FastAPI provides OpenAPI documentation."},

    {"role": "user", "content": "I plan to use Docker for deployment."},
    {"role": "assistant", "content": "Docker can package your application."},

    {"role": "user", "content": "I am currently working on the authentication module."},
    {"role": "assistant", "content": "Let's focus on authentication first."}
]


# STEP 1: Spliting older and recent conversations

split_index = RECENT_TURNS * 2

older_history = conversation_history[:-split_index]
recent_history = conversation_history[-split_index:]


# STEP 2: Generating a summary of older conversations

older_text = "\n".join(
    f"{message['role']}: {message['content']}"
    for message in older_history
)

summary_response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. "
                "Compress the conversation into a concise summary. "
                "Preserve important facts, decisions, names, "
                "technologies, numbers, and deadlines. "
                "Do not invent information."
            )
        },
        {
            "role": "user",
            "content": older_text
        }
    ]
)

rolling_summary = summary_response.message.content


# STEP 3: Construct compressed conversation context

compressed_history = [
    {
        "role": "system",
        "content": (
            "Conversation summary from earlier turns:\n"
            + rolling_summary
        )
    }
] + recent_history


# STEP 4: Comparing full history vs summarized history

question = (
    "Which database and ORM did we choose, "
    "what is the JWT expiry, and what am I currently working on?"
)

# Full conversation
full_response = ollama.chat(
    model=MODEL,
    messages=conversation_history + [
        {"role": "user", "content": question}
    ]
)

# Summarized conversation
summary_response = ollama.chat(
    model=MODEL,
    messages=compressed_history + [
        {"role": "user", "content": question}
    ]
)


# STEP 5: Display results

print("\n========== GENERATED SUMMARY ==========")
print(rolling_summary)

print("\n========== FULL HISTORY RESPONSE ==========")
print(full_response.message.content)

print("\n========== SUMMARIZED HISTORY RESPONSE ==========")
print(summary_response.message.content)

print("\n========== TOKEN COMPARISON ==========")

full_tokens = full_response.prompt_eval_count
compressed_tokens = summary_response.prompt_eval_count

print("Full history prompt tokens:", full_tokens)
print("Summarized prompt tokens:", compressed_tokens)

if full_tokens and compressed_tokens:
    reduction = (
        (full_tokens - compressed_tokens) / full_tokens
    ) * 100

    print(f"Prompt token reduction: {reduction:.2f}%")