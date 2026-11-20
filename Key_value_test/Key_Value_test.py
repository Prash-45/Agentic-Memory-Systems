import ollama
import json

MODEL = "qwen3:8b"

# STEP 1:conversation history

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

    {"role": "user", "content": "Our project deadline is March 2027."},
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


# STEP 2: Convert conversation into readable text

conversation_text = "\n".join(
    f"{msg['role']}: {msg['content']}"
    for msg in conversation_history
)


# STEP 3: Extract important information as JSON

extraction_response = ollama.chat(
    model=MODEL,
    format="json",
    messages=[
        {
            "role": "system",
            "content": """
            You are an information extraction system.

            Extract important facts from the conversation.

            Return valid JSON containing these keys:
            project, framework, database, orm, team_size,
            deadline, authentication, jwt_expiry,
            access_control, testing, version_control,
            documentation, deployment, current_task.

            Rules:
            1. Extract only information supported by the conversation.
            2. Do not invent or assume missing information.
            3. Use null for unknown values.
            4. Return JSON only.
            """
        },
        {
            "role": "user",
            "content": conversation_text
        }
    ]
)


# STEP 4: Parse JSON into a Python dictionary

memory = json.loads(
    extraction_response.message.content
)

print("\n========== EXTRACTED MEMORY ==========")
print(json.dumps(memory, indent=4))


# STEP 5: Retrieve a specific value

print("\n========== MEMORY RETRIEVAL ==========")

print("Database:", memory["database"])
print("Framework:", memory["framework"])
print("Team size:", memory["team_size"])
print("JWT expiry:", memory["jwt_expiry"])


# STEP 6: Compare token consumption

question = "Which database and ORM are we using?"

# A. Full conversation approach
baseline_answer = ollama.chat(
    model=MODEL,
    messages=conversation_history + [
        {"role": "user", "content": question}
    ]
)

# B. Key-value memory approach
memory_answer = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "Answer using this extracted memory:\n"
                       + json.dumps(memory)
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

# Get actual prompt token counts
full_tokens = baseline_answer.prompt_eval_count
memory_tokens = memory_answer.prompt_eval_count

# Calculate reduction
reduction = (
    (full_tokens - memory_tokens) / full_tokens
) * 100

print("\n========== TOKEN COMPARISON ==========")

print("Full history tokens:", full_tokens)
print("Key-value memory tokens:", memory_tokens)

print("Tokens saved:", full_tokens - memory_tokens)
print(f"Token reduction: {reduction:.2f}%")

# Also show extraction overhead
print("\n========== EXTRACTION COST ==========")
print("Extraction input tokens:",
      extraction_response.prompt_eval_count)

print("Extraction output tokens:",
      extraction_response.eval_count)
