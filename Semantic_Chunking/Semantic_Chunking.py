import ollama
import math

EMBED_MODEL = "nomic-embed-text"
 
# STEP 1: Conversation history

conversation = [
    {
        "user": "I am learning Python programming.",
        "assistant": "Python is a versatile programming language."
    },
    {
        "user": "I have started learning Python OOP and classes.",
        "assistant": "OOP uses classes and objects to organize code."
    },
    {
        "user": "I want to understand inheritance and polymorphism.",
        "assistant": "Both are important concepts in object-oriented programming."
    },
    {
        "user": "Now I am learning PostgreSQL and database design.",
        "assistant": "PostgreSQL is a relational database system."
    },
    {
        "user": "I want to understand SQL joins and foreign keys.",
        "assistant": "Joins combine records from related database tables."
    },
    {
        "user": "How do I connect PostgreSQL with Python?",
        "assistant": "You can use libraries such as SQLAlchemy."
    },
    {
        "user": "I am building an authentication system.",
        "assistant": "Authentication verifies a user's identity."
    },
    {
        "user": "I want to implement JWT tokens.",
        "assistant": "JWT can carry signed authentication claims."
    },
    {
        "user": "How can I add role-based access control?",
        "assistant": "You can assign roles and permissions to users."
    },
    {
        "user": "I want to deploy my application using Docker.",
        "assistant": "Docker packages applications into containers."
    },
    {
        "user": "How do Docker images and containers differ?",
        "assistant": "An image is a template; a container is its running instance."
    },
    {
        "user": "Can I use Docker Compose for multiple services?",
        "assistant": "Yes, Compose manages multi-container applications."
    }
]


# STEP 2: Combining each user-assistant exchange

turns = [
    f"User: {item['user']}\nAssistant: {item['assistant']}"
    for item in conversation
]


# STEP 3: Generate embeddings

def get_embedding(text):

    response = ollama.embed(
        model=EMBED_MODEL,
        input=text
    )

    return response.embeddings[0]


embeddings = [
    get_embedding(turn)
    for turn in turns
]


# STEP 4: Calculating cosine similarity

def cosine_similarity(a, b):

    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# STEP 5: Group semantically related turns

THRESHOLD = 0.70

chunks = []
current_chunk = [turns[0]]

for i in range(1, len(turns)):

    similarity = cosine_similarity(
        embeddings[i - 1],
        embeddings[i]
    )

    print(
        f"Turn {i} vs Turn {i + 1}: "
        f"{similarity:.3f}"
    )

    if similarity >= THRESHOLD:

        current_chunk.append(turns[i])

    else:

        chunks.append(current_chunk)
        current_chunk = [turns[i]]


# Add the final chunk
if current_chunk:
    chunks.append(current_chunk)


# STEP 6: Display semantic chunks

print("\n========== SEMANTIC CHUNKS ==========")

for index, chunk in enumerate(chunks, start=1):

    print(f"\nCHUNK {index}")
    print("-" * 40)

    for turn in chunk:
        print(turn)
        print()


# STEP 7: Embedding for each chunk

def average_embedding(chunk):

    vectors = [
        embeddings[turns.index(turn)]
        for turn in chunk
    ]

    return [
        sum(values) / len(values)
        for values in zip(*vectors)
    ]


chunk_embeddings = [
    average_embedding(chunk)
    for chunk in chunks
]


# STEP 8: Receive a user query

question = "Which database am I using?"


# STEP 9: Generate query embedding

query_embedding = get_embedding(question)


# STEP 10: Compare query with every chunk

scores = []

for index, chunk_embedding in enumerate(chunk_embeddings):

    similarity = cosine_similarity(
        query_embedding,
        chunk_embedding
    )

    scores.append((index, similarity))


# STEP 11: Retrieve the most relevant chunk

scores.sort(
    key=lambda item: item[1],
    reverse=True
)

top_k = 1

retrieved_chunks = [
    chunks[index]
    for index, score in scores[:top_k]
]

retrieved_context = "\n\n".join(
    turn for chunk in retrieved_chunks
    for turn in chunk
)


print("\n========== RETRIEVAL RESULTS ==========")

for index, score in scores:
    print(f"Chunk {index + 1}: {score:.4f}")

print("\nRetrieved context:")
print(retrieved_context)


# STEP 12: Qwen3 answering using retrieved memory

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": "Answer using the provided conversation context."
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{retrieved_context}\n\n"
                f"Question: {question}"
            )
        }
    ]
)

print("\n========== LLM ANSWER ==========")
print(response.message.content)


# STEP 13: Compare token consumption

full_context = "\n\n".join(turns)

baseline = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": "Answer using the provided conversation context."
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{full_context}\n\n"
                f"Question: {question}"
            )
        }
    ]
)

full_tokens = baseline.prompt_eval_count
retrieved_tokens = response.prompt_eval_count

saved_tokens = full_tokens - retrieved_tokens

reduction = (
    saved_tokens / full_tokens
) * 100 if full_tokens else 0

print("\n========== TOKEN COMPARISON ==========")
print("Full conversation:", full_tokens)
print("Retrieved context:", retrieved_tokens)
print("Tokens saved:", saved_tokens)
print(f"Reduction: {reduction:.2f}%")
