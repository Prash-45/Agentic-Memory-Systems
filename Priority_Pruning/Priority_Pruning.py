from datetime import datetime, timedelta
import ollama
import json


# STEP 1: Define memory retention policies

MEMORY_TIERS = {
    "eternal": {
        "priority": 4,
        "expiry_hours": None
    },
    "stable": {
        "priority": 3,
        "expiry_hours": 90 * 24
    },
    "evolving": {
        "priority": 2,
        "expiry_hours": 14 * 24
    },
    "experimental": {
        "priority": 1,
        "expiry_hours": 6
    }
}


# STEP 2: Create toy memory records

now = datetime.now()

memory_store = [
    {
        "key": "user_name",
        "value": "Alex",
        "tier": "eternal",
        "created_at": now - timedelta(days=100)
    },
    {
        "key": "core_language",
        "value": "Python",
        "tier": "eternal",
        "created_at": now - timedelta(days=50)
    },
    {
        "key": "project_convention",
        "value": "Use type hints",
        "tier": "stable",
        "created_at": now - timedelta(days=20)
    },
    {
        "key": "team_standard",
        "value": "Use GitHub pull requests",
        "tier": "stable",
        "created_at": now - timedelta(days=10)
    },
    {
        "key": "current_task",
        "value": "Building MCP memory architecture",
        "tier": "evolving",
        "created_at": now - timedelta(days=2)
    },
    {
        "key": "recent_decision",
        "value": "Use Qwen3 with Ollama",
        "tier": "evolving",
        "created_at": now - timedelta(days=1)
    },
    {
        "key": "active_blocker",
        "value": "CPU inference is slow",
        "tier": "evolving",
        "created_at": now - timedelta(days=20)
    },
    {
        "key": "test_hypothesis",
        "value": "Try a different prompt",
        "tier": "experimental",
        "created_at": now - timedelta(hours=1)
    },
    {
        "key": "temporary_debug",
        "value": "Check JSON parsing",
        "tier": "experimental",
        "created_at": now - timedelta(hours=8)
    }
]


# STEP 3: Check whether a memory has expired

def is_expired(memory):

    tier = MEMORY_TIERS[memory["tier"]]

    expiry = tier["expiry_hours"]

    # Eternal memories never expire
    if expiry is None:
        return False

    age = datetime.now() - memory["created_at"]

    return age > timedelta(hours=expiry)


# STEP 4: Apply priority pruning

def prune_memory(memories, max_memory=7):

    # Remove expired memories first
    valid_memories = [
        memory for memory in memories
        if not is_expired(memory)
    ]

    # Sorting by priority, then recency
    valid_memories.sort(
        key=lambda memory: (
            MEMORY_TIERS[memory["tier"]]["priority"],
            memory["created_at"]
        ),
        reverse=True
    )

    # Retain only the highest-priority memories
    retained = valid_memories[:max_memory]

    return retained


# STEP 5: Execute pruning

print("\n========== BEFORE PRUNING ==========")

for item in memory_store:
    print(item["key"], ":", item["tier"])

retained_memory = prune_memory(
    memory_store,
    max_memory=7
)

print("\n========== AFTER PRUNING ==========")

for item in retained_memory:
    print(item["key"], ":", item["tier"])

print("\nRemoved memories:")

retained_keys = {
    item["key"] for item in retained_memory
}

for item in memory_store:
    if item["key"] not in retained_keys:
        print(item["key"], ":", item["tier"])

# Convert memories into JSON text
before_memory = json.dumps(memory_store, default=str)
after_memory = json.dumps(retained_memory, default=str)

# Measure tokens before pruning
before = ollama.chat(
    model="qwen3:8b",
    messages=[
        {"role": "user", "content": before_memory}
    ]
)

# Measure tokens after pruning
after = ollama.chat(
    model="qwen3:8b",
    messages=[
        {"role": "user", "content": after_memory}
    ]
)

before_tokens = before.prompt_eval_count
after_tokens = after.prompt_eval_count

reduction = (
    (before_tokens - after_tokens) / before_tokens
) * 100

print("\n========== TOKEN REDUCTION ==========")
print("Before:", before_tokens)
print("After:", after_tokens)
print("Tokens saved:", before_tokens - after_tokens)
print(f"Reduction: {reduction:.2f}%")
