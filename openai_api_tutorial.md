![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_API-412991?style=flat-square&logo=openai&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Topics](https://img.shields.io/badge/Topics-16-blueviolet?style=flat-square)

A comprehensive reference guide for building production applications with the OpenAI API, Hugging Face Transformers, embeddings, semantic search, and vector databases (ChromaDB and Pinecone).

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [OpenAI API — Chat Completions](#2-openai-api--chat-completions)
3. [Key Parameters Reference](#3-key-parameters-reference)
4. [Tokenization & Cost Calculation](#4-tokenization--cost-calculation)
5. [Shot Prompting](#5-shot-prompting)
6. [Chat Roles and System Messages](#6-chat-roles-and-system-messages)
7. [Multi-Turn Conversations](#7-multi-turn-conversations)
8. [Prompt Formatting with F-Strings](#8-prompt-formatting-with-f-strings)
9. [Chain of Thought & Self-Consistency Prompting](#9-chain-of-thought--self-consistency-prompting)
10. [Structured Output (JSON Mode)](#10-structured-output-json-mode)
11. [Error Handling & Retries](#11-error-handling--retries)
12. [Batching Requests](#12-batching-requests)
13. [Token Management with tiktoken](#13-token-management-with-tiktoken)
14. [Function Calling (Tool Use)](#14-function-calling-tool-use)
15. [Streaming Responses](#15-streaming-responses)
16. [Hugging Face — The Pipeline API](#16-hugging-face--the-pipeline-api)
17. [Hugging Face — Inference Providers](#17-hugging-face--inference-providers)
18. [Hugging Face — Datasets](#18-hugging-face--datasets)
19. [Text Classification](#19-text-classification)
20. [Text Summarization](#20-text-summarization)
21. [Auto Models and Tokenizers](#21-auto-models-and-tokenizers)
22. [Document Q&A](#22-document-qa)
23. [OpenAI Embeddings](#23-openai-embeddings)
24. [Semantic Search with Embeddings](#24-semantic-search-with-embeddings)
25. [Recommendation Systems](#25-recommendation-systems)
26. [Embeddings for Classification](#26-embeddings-for-classification)
27. [Vector Databases — ChromaDB](#27-vector-databases--chromadb)
28. [ChromaDB — Querying and Filtering](#28-chromadb--querying-and-filtering)
29. [Vector Databases — Pinecone](#29-vector-databases--pinecone)
30. [Pinecone — Performance Tuning](#30-pinecone--performance-tuning)
31. [Semantic Search with Pinecone](#31-semantic-search-with-pinecone)
32. [Quick Reference Cheat Sheet](#32-quick-reference-cheat-sheet)

---

## 1. Installation & Setup

Before writing any code, install the required libraries. It is best practice to work inside a virtual environment (`python -m venv .venv && source .venv/bin/activate`) so dependencies do not collide across projects.

```bash
pip install openai
pip install huggingface_hub transformers datasets
pip install chromadb
pip install pinecone
pip install tiktoken
pip install tenacity
pip install pypdf
pip install scipy
pip install python-dotenv
```

Or install everything in one shot:

```bash
pip install openai huggingface_hub transformers datasets chromadb pinecone tiktoken tenacity pypdf scipy python-dotenv
```

### Setting API Keys

Never hardcode API keys in source code. Use environment variables instead. There are two common patterns.

**Pattern 1 — Set keys directly in your shell session:**

```bash
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."
export PINECONE_API_KEY="pcsk_..."
```

**Pattern 2 — Use a `.env` file with `python-dotenv` (recommended for projects):**

Create a `.env` file in your project root (add it to `.gitignore` immediately):

```bash
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
PINECONE_API_KEY=pcsk_...
```

Then load it at the top of your Python script:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env and populates os.environ

openai_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")
pinecone_key = os.getenv("PINECONE_API_KEY")
```

`os.getenv` returns `None` silently if a variable is missing; `os.environ["KEY"]` raises a `KeyError`, which is useful when the key is mandatory.

---

## 2. OpenAI API — Chat Completions

The Chat Completions endpoint (`/v1/chat/completions`) is the primary interface for interacting with GPT-4o, GPT-4, GPT-3.5-turbo, and other OpenAI chat models. You send a list of messages — each with a role and content — and receive a model-generated reply.

The `messages` array is the core of every request. It acts as the full conversation context the model sees. Roles determine how the model interprets each message:

- **system** — Sets the model's behavior, persona, or constraints. Processed first.
- **user** — The human turn. This is what the user typed or what you inject programmatically.
- **assistant** — A previous model response. Used to give the model memory of what it already said.

Here is a complete, annotated example of a chat completion call:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",                  # model identifier
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant who explains concepts clearly."
        },
        {
            "role": "user",
            "content": "What is the difference between RAM and storage?"
        }
    ],
    temperature=0.7,                 # 0 = deterministic, 2 = very random
    max_tokens=500,                  # hard cap on output length
    top_p=1.0,                       # nucleus sampling threshold
    frequency_penalty=0.0,           # penalize repeated tokens (-2 to 2)
    presence_penalty=0.0,            # penalize tokens already used (-2 to 2)
)

# The model's reply is in the first choice's message content
answer = response.choices[0].message.content
print(answer)
```

The response object also contains useful metadata:

```python
print(response.model)                         # which model was used
print(response.usage.prompt_tokens)           # tokens consumed by your input
print(response.usage.completion_tokens)       # tokens consumed by the output
print(response.usage.total_tokens)            # total tokens billed
```

---

## 3. Key Parameters Reference

Use this table as a quick lookup when tuning your API calls. Each parameter influences cost, creativity, and output length.

| Parameter | Type | Range / Options | Description |
|---|---|---|---|
| `model` | string | `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`, etc. | Which model to use. Affects capability and cost. |
| `messages` | list | Array of role/content dicts | The full conversation history the model sees. |
| `temperature` | float | 0.0 – 2.0 | Controls randomness. Lower = more focused and deterministic. Higher = more creative and varied. |
| `max_tokens` | int | 1 – model limit | Maximum number of tokens to generate. The call stops when this limit is hit. |
| `max_completion_tokens` | int | 1 – model limit | Newer alias for `max_tokens` used in the Responses API. Preferred over `max_tokens` on newer models. |
| `top_p` | float | 0.0 – 1.0 | Nucleus sampling. Only tokens whose cumulative probability reaches `top_p` are considered. 0.1 = very conservative vocabulary. |
| `frequency_penalty` | float | -2.0 – 2.0 | Positive values reduce the likelihood of repeating the same words. Useful for avoiding repetitive text. |
| `presence_penalty` | float | -2.0 – 2.0 | Positive values encourage the model to introduce new topics. Increases diversity of content. |
| `stop` | string or list | Any string(s) | The model stops generating when it produces one of these sequences. |
| `n` | int | 1+ | Number of independent completions to generate. Each costs tokens. |
| `stream` | bool | `True` / `False` | If `True`, returns tokens as a stream instead of waiting for the full response. |
| `response_format` | dict | `{"type": "json_object"}` | Forces the model to return valid JSON. Requires instructing the model to output JSON in the system message. |
| `seed` | int | Any integer | Makes output more reproducible when combined with a fixed temperature. Not guaranteed to be deterministic. |

---

## 4. Tokenization & Cost Calculation

Tokens are the units the model reads and writes. A token is roughly 4 characters or 0.75 words in English. Knowing how to count tokens lets you stay within context limits and accurately forecast API costs.

The `tiktoken` library is OpenAI's official tokenizer. It encodes text into the exact token IDs used by each model family.

```python
import tiktoken

# Load the tokenizer for a specific model
enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, how are you doing today?"
tokens = enc.encode(text)

print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")
```

### Cost Calculation Example

Different models have different per-token prices. Here is a simple helper that estimates cost before you make a call:

```python
import tiktoken

def estimate_cost(text: str, model: str = "gpt-4o") -> dict:
    """Estimate the token count and approximate cost for a prompt."""
    enc = tiktoken.encoding_for_model(model)
    token_count = len(enc.encode(text))

    # Prices in USD per 1,000 tokens (check platform.openai.com for current rates)
    price_per_1k = {
        "gpt-4o": 0.005,
        "gpt-4": 0.03,
        "gpt-3.5-turbo": 0.0015,
    }

    rate = price_per_1k.get(model, 0.005)
    estimated_cost = (token_count / 1000) * rate

    return {
        "token_count": token_count,
        "estimated_cost_usd": round(estimated_cost, 6),
        "model": model,
    }

result = estimate_cost("Explain the concept of neural networks in simple terms.", "gpt-4o")
print(result)
```

### Token Budgeting Pattern

When your input might be long, truncate it to stay within the model's context window:

```python
import tiktoken

def truncate_to_token_limit(text: str, max_tokens: int = 3000, model: str = "gpt-4o") -> str:
    """Trim text so it does not exceed max_tokens when encoded."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return enc.decode(tokens)
    return text
```

---

## 5. Shot Prompting

"Shot prompting" refers to how many examples you include in your prompt to guide the model's behavior.

### Zero-Shot Prompting

You provide no examples — just an instruction. The model relies entirely on its pre-trained knowledge.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a sentiment analysis assistant."},
        {"role": "user", "content": "Classify the sentiment: 'The battery life on this laptop is absolutely terrible.'"}
    ]
)
print(response.choices[0].message.content)
```

### One-Shot Prompting

You provide a single example to anchor the format or style you expect.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You classify customer reviews as Positive, Negative, or Neutral."},
        {"role": "user", "content": "Review: 'Fast shipping, exactly what I ordered.' → Positive\n\nReview: 'The zipper broke after one week.' →"}
    ]
)
print(response.choices[0].message.content)
```

### Few-Shot Prompting

You provide multiple examples to firmly establish a pattern. This is especially useful for tasks with specific output formats.

```python
few_shot_prompt = """
Translate English to SQL WHERE clauses.

English: users older than 30
SQL: WHERE age > 30

English: orders placed in January
SQL: WHERE MONTH(order_date) = 1

English: products with price under 50 dollars
SQL: WHERE price < 50

English: customers from New York
SQL:"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert SQL assistant."},
        {"role": "user", "content": few_shot_prompt}
    ],
    temperature=0.0  # deterministic output for code generation
)
print(response.choices[0].message.content)
```

---

## 6. Chat Roles and System Messages

Understanding roles is fundamental to prompt engineering. Each role shapes how the model interprets a message.

### System Role — Setting Behavior and Persona

The system message is processed before any user input and sets the model's identity, tone, constraints, and knowledge scope. It is the most powerful lever for controlling model behavior.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are Aria, a senior financial advisor at a wealth management firm. "
                "You speak with confidence, use precise financial terminology, and always "
                "remind users to consult a licensed advisor before making investment decisions. "
                "You never speculate about specific stock prices."
            )
        },
        {
            "role": "user",
            "content": "Should I invest in index funds or individual stocks?"
        }
    ]
)
print(response.choices[0].message.content)
```

### User Role

The user role represents the human side of the conversation. In application code, this is often populated dynamically from user input or injected data.

```python
user_question = "What are the main symptoms of vitamin D deficiency?"

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a knowledgeable health information assistant."},
        {"role": "user", "content": user_question}
    ]
)
print(response.choices[0].message.content)
```

### Assistant Role — Providing Conversation History

When you want the model to remember what it already said in a previous turn, you replay its earlier responses using the `assistant` role. This is how you implement memory in a stateless API.

```python
# Simulating a multi-turn conversation where the model needs to remember context
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful cooking assistant."
        },
        {
            "role": "user",
            "content": "I want to make a pasta dish. What ingredients do I need?"
        },
        {
            "role": "assistant",
            "content": (
                "For a classic pasta dish you'll need: pasta (spaghetti or penne work great), "
                "olive oil, garlic, canned tomatoes, onion, salt, pepper, and fresh basil. "
                "You can add ground beef or Italian sausage for a meat sauce."
            )
        },
        {
            "role": "user",
            "content": "Great — can you give me step-by-step cooking instructions?"
        }
    ]
)
# The model now has full context of what ingredients were already discussed
print(response.choices[0].message.content)
```

### Guardrails via System Message

You can use the system message to restrict what the model will and will not do, creating a safety boundary for your application:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a customer support agent for AcmeSoftware. "
                "Only answer questions related to AcmeSoftware products. "
                "If the user asks about competitors or unrelated topics, politely explain "
                "that you can only help with AcmeSoftware-related questions."
            )
        },
        {"role": "user", "content": "Can you compare your product to CompetitorX?"}
    ]
)
print(response.choices[0].message.content)
```

---

## 7. Multi-Turn Conversations

A multi-turn conversation is simply a `messages` list that grows with each exchange. Since the API is stateless, your application is responsible for maintaining and passing the full history on every call.

The following pattern shows how to build a conversational loop where the history is preserved across turns:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat(messages: list, user_input: str, model: str = "gpt-4o") -> str:
    """Add the user message, call the API, append the reply, return the reply text."""
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply


# Initialize with a system prompt
conversation_history = [
    {"role": "system", "content": "You are a concise and helpful travel planning assistant."}
]

# Turn 1
reply1 = chat(conversation_history, "I want to visit Japan in April.")
print(f"Assistant: {reply1}\n")

# Turn 2 — the model remembers Japan and April from Turn 1
reply2 = chat(conversation_history, "What cities would you recommend starting with?")
print(f"Assistant: {reply2}\n")

# Turn 3 — the model has full context
reply3 = chat(conversation_history, "How many days should I allocate to Tokyo?")
print(f"Assistant: {reply3}\n")

print(f"Total messages in history: {len(conversation_history)}")
```

### Managing History Length

As conversations grow, the prompt eventually exceeds the model's context window. A simple mitigation is to keep only the system message and the most recent N turns:

```python
def trim_history(messages: list, max_turns: int = 10) -> list:
    """Keep the system message and the most recent max_turns user/assistant pairs."""
    system_messages = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    # Each turn = 1 user + 1 assistant message = 2 entries
    trimmed = non_system[-(max_turns * 2):]
    return system_messages + trimmed
```

---

## 8. Prompt Formatting with F-Strings

F-strings are the cleanest way to inject dynamic data into prompts. This keeps your prompt templates readable and your data handling separate from the prompt logic.

### Story Completion Example

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

genre = "science fiction"
setting = "a deserted Mars colony in the year 2147"
protagonist = "a geologist named Dr. Reyes"

prompt = (
    f"Write the opening paragraph of a {genre} story. "
    f"The story is set in {setting}. "
    f"The main character is {protagonist}, who has just discovered something unexpected underground."
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a creative fiction writer."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.9,
    max_tokens=300
)

print(response.choices[0].message.content)
```

### Injecting External Text (e.g., from a file or database)

When you have long external content such as a document or a retrieved passage, inject it cleanly with clear delimiters so the model understands where your instructions end and the content begins:

```python
def summarize_document(document_text: str) -> str:
    """Summarize an arbitrary document using a structured prompt."""
    prompt = f"""Please summarize the following document in 3-5 bullet points.
Focus on the key findings, decisions, or recommendations.

---
{document_text}
---

Summary:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert document summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content
```

---

## 9. Chain of Thought & Self-Consistency Prompting

Chain of Thought (CoT) prompting asks the model to reason step by step before producing a final answer. This dramatically improves accuracy on math, logic, and multi-step reasoning tasks.

### Basic Chain of Thought

Adding "Let's think step by step" or "Reason through this carefully" to your prompt triggers the model to show its work:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

problem = """
A train leaves City A at 9:00 AM traveling at 80 km/h toward City B.
Another train leaves City B at 10:00 AM traveling at 100 km/h toward City A.
The distance between the cities is 540 km.
At what time do the trains meet?
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a math tutor. Always reason step by step before giving a final answer."
        },
        {
            "role": "user",
            "content": f"{problem}\n\nLet's think through this step by step."
        }
    ],
    temperature=0.0  # low temperature for deterministic math
)

print(response.choices[0].message.content)
```

### Self-Consistency Pattern

Self-consistency generates multiple reasoning paths and picks the most common answer, reducing the chance of a fluke error:

```python
def self_consistent_answer(question: str, n_samples: int = 3) -> str:
    """Generate multiple reasoning paths and return the most frequent answer."""
    answers = []

    for i in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Reason step by step, then provide a final numeric or one-word answer on the last line prefixed with 'Answer:'."
                },
                {"role": "user", "content": question}
            ],
            temperature=0.7  # some variance to get different reasoning paths
        )
        full_text = response.choices[0].message.content
        # Extract the final answer line
        for line in full_text.splitlines():
            if line.strip().startswith("Answer:"):
                answers.append(line.replace("Answer:", "").strip())
                break

    # Return the most common answer
    if answers:
        return max(set(answers), key=answers.count)
    return "No consistent answer found."

result = self_consistent_answer("If you have 3 dozen eggs and use 14, how many are left?")
print(f"Self-consistent answer: {result}")
```

---

## 10. Structured Output (JSON Mode)

JSON mode forces the model to always return a syntactically valid JSON object. This is essential for pipelines where the output is parsed programmatically. You must also instruct the model to produce JSON in the system or user message — the parameter alone is not sufficient.

### Basic JSON Mode

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a data extraction assistant. "
                "Always respond with valid JSON only, no markdown or extra text."
            )
        },
        {
            "role": "user",
            "content": "Extract the name, age, and city from: 'Maria Gonzalez, 34 years old, lives in Austin, Texas.'"
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.0
)

data = json.loads(response.choices[0].message.content)
print(data)
# Expected: {"name": "Maria Gonzalez", "age": 34, "city": "Austin"}
```

### Structuring an API Call for a Specific Schema

When you need the JSON to conform to a particular schema, describe the schema explicitly in the prompt:

```python
import json

schema_description = """
Return a JSON object with this exact structure:
{
  "product_name": "string",
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": float between 0.0 and 1.0,
  "key_phrases": ["array", "of", "strings"]
}
"""

review_text = "I absolutely love this keyboard. The tactile feedback is satisfying and it has held up for two years."

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": f"You are a product review analyzer. {schema_description}"
        },
        {
            "role": "user",
            "content": f"Analyze this review: {review_text}"
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.0
)

result = json.loads(response.choices[0].message.content)
print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})")
print(f"Key phrases: {result['key_phrases']}")
```

---

## 11. Error Handling & Retries

The OpenAI API can return errors due to rate limits, network issues, invalid authentication, or server-side problems. Robust production code must handle these gracefully.

The `tenacity` library provides a clean decorator-based retry mechanism with exponential backoff — meaning it waits increasingly longer between retries to avoid hammering the API.

```python
import os
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=60),  # wait 2s, 4s, 8s... up to 60s
    stop=stop_after_attempt(5)
)
def call_with_retry(messages: list, model: str = "gpt-4o") -> str:
    """Call the API with automatic retry on rate limit errors."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content


def safe_chat(messages: list) -> str:
    """Wraps the API call with comprehensive error handling."""
    try:
        return call_with_retry(messages)

    except openai.AuthenticationError:
        print("Authentication failed. Check that OPENAI_API_KEY is set correctly.")
        raise

    except openai.RateLimitError:
        print("Rate limit exceeded even after retries. Consider request queuing or upgrading your plan.")
        raise

    except openai.BadRequestError as e:
        print(f"Bad request — likely a content policy violation or invalid parameter: {e}")
        raise

    except openai.OpenAIError as e:
        # Catch-all for any other OpenAI API error
        print(f"An unexpected OpenAI error occurred: {e}")
        raise


# Usage
messages = [{"role": "user", "content": "What is the speed of light?"}]
answer = safe_chat(messages)
print(answer)
```

**Exponential backoff** means: first retry after 2 seconds, second after 4, third after 8, and so on. This prevents your application from overwhelming the API when it is temporarily overloaded and respects rate limit reset windows.

---

## 12. Batching Requests

When you need to process many items independently, you can send them in a single prompt (pseudo-batching) or use the OpenAI Batch API for true asynchronous batching. The simplest approach is to include all items in one prompt and parse the structured response.

This pattern is useful for classification, translation, or data extraction tasks applied to a list of inputs:

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

countries = ["Brazil", "Germany", "Japan", "Nigeria", "Australia"]

# Build a single prompt that processes all items at once
batch_prompt = f"""For each country in the list below, provide:
- capital city
- continent
- official language

Countries: {', '.join(countries)}

Respond with a JSON array where each element has the keys: country, capital, continent, language."""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a geography expert. Respond with valid JSON only."},
        {"role": "user", "content": batch_prompt}
    ],
    response_format={"type": "json_object"},
    temperature=0.0
)

# The model returns a JSON object; adjust key name based on actual output
raw = json.loads(response.choices[0].message.content)
results = raw.get("countries", raw)  # handle varying root key names

for item in results:
    print(f"{item['country']}: {item['capital']}, {item['continent']}, {item['language']}")
```

**When to use this pattern:** Batching into a single prompt reduces API call overhead and latency when items are small. For very large batches (hundreds to thousands of items), use the official OpenAI Batch API which processes requests asynchronously at a discounted rate.

---

## 13. Token Management with tiktoken

Managing tokens precisely lets you avoid truncation errors, optimize costs, and implement sliding-window context strategies. Here is a full example combining token counting with conditional behavior:

```python
import os
import tiktoken
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return the exact token count for a string using the model's tokenizer."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def count_messages_tokens(messages: list, model: str = "gpt-4o") -> int:
    """Count total tokens across a messages list, including role overhead."""
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for message in messages:
        # Each message has ~4 tokens of overhead for formatting
        total += 4
        for key, value in message.items():
            total += len(enc.encode(str(value)))
    total += 2  # reply priming overhead
    return total


def smart_completion(user_input: str, system_prompt: str, model: str = "gpt-4o", token_limit: int = 4000) -> str:
    """
    Only call the API if the prompt fits within token_limit.
    Truncates user input if needed to fit.
    """
    enc = tiktoken.encoding_for_model(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    token_count = count_messages_tokens(messages, model)

    if token_count > token_limit:
        print(f"Prompt too long ({token_count} tokens). Truncating user input.")
        # Calculate how many tokens to shave off
        overage = token_count - token_limit
        user_tokens = enc.encode(user_input)
        trimmed_tokens = user_tokens[:max(1, len(user_tokens) - overage)]
        user_input = enc.decode(trimmed_tokens)
        messages[1]["content"] = user_input
        print(f"Truncated to {count_messages_tokens(messages, model)} tokens.")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content


# Example usage
result = smart_completion(
    user_input="Summarize the history of the Roman Empire in detail.",
    system_prompt="You are a concise historian.",
    token_limit=200  # intentionally low to demonstrate truncation
)
print(result)
```

---

## 14. Function Calling (Tool Use)

Function calling lets the model trigger external tools by outputting a structured JSON payload instead of plain text. Your application executes the actual function and returns the result. This is the foundation of AI agents.

The flow is: you define tools → the model decides whether to call one → you execute the function → you send the result back → the model gives a final answer.

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Define the tools available to the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'San Francisco'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit to return"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# Step 2: Send the initial request — the model may call a tool
messages = [{"role": "user", "content": "What's the weather like in Tokyo right now?"}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # let the model decide whether to call a tool
)

response_message = response.choices[0].message

# Step 3: Check if the model decided to call a tool
if response_message.tool_calls:
    tool_call = response_message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    print(f"Model wants to call: {function_name}")
    print(f"With arguments: {function_args}")

    # Step 4: Execute your actual function (mocked here)
    def get_current_weather(city: str, unit: str = "celsius") -> dict:
        """Mock weather function — replace with a real API call."""
        return {"city": city, "temperature": 18, "unit": unit, "condition": "Partly cloudy"}

    function_result = get_current_weather(**function_args)

    # Step 5: Send the result back to the model to get a natural language answer
    messages.append(response_message)  # append the model's tool_call message
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(function_result)
    })

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )

    print(final_response.choices[0].message.content)
```

---

## 15. Streaming Responses

Streaming returns tokens to your application as they are generated rather than waiting for the complete response. This dramatically improves perceived latency in user-facing applications and allows you to start rendering output immediately.

Enable streaming by setting `stream=True`. The API then returns a generator of `ChatCompletionChunk` objects:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def stream_response(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Stream the response token by token and return the full text."""
    full_response = []

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        stream=True  # enable streaming
    )

    print("Response: ", end="", flush=True)
    for chunk in stream:
        # Each chunk may or may not contain a delta
        delta = chunk.choices[0].delta
        if delta.content is not None:
            print(delta.content, end="", flush=True)
            full_response.append(delta.content)

    print()  # newline after stream ends
    return "".join(full_response)


full_text = stream_response("Explain how photosynthesis works in three sentences.")
print(f"\nTotal characters received: {len(full_text)}")
```

### Streaming with Token Counting

You can track token usage after the stream completes by checking the final chunk:

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List five programming languages."}],
    stream=True,
    stream_options={"include_usage": True}  # request usage stats in final chunk
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.usage:
        print(f"\n\nTokens used — prompt: {chunk.usage.prompt_tokens}, completion: {chunk.usage.completion_tokens}")
```

> **When to use:** Use streaming for chat interfaces, code editors, and any user-facing feature where perceived responsiveness matters. Avoid streaming in batch processing pipelines where you need the complete response before proceeding.

---

## 16. Hugging Face — The Pipeline API

The `pipeline` function from Hugging Face Transformers is the highest-level abstraction for running NLP tasks locally. It downloads a pretrained model, handles tokenization, runs inference, and post-processes output — all in one call.

### Available Pipeline Tasks

```python
from transformers import pipeline

# Sentiment analysis
sentiment = pipeline("sentiment-analysis")
result = sentiment("I had an amazing time at the concert last night!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation
generator = pipeline("text-generation", model="gpt2")
output = generator("Once upon a time in a small mountain village,", max_length=60, num_return_sequences=1)
print(output[0]["generated_text"])

# Named entity recognition
ner = pipeline("ner", grouped_entities=True)
entities = ner("Barack Obama was born in Honolulu, Hawaii.")
for entity in entities:
    print(f"{entity['word']} → {entity['entity_group']} (score: {entity['score']:.3f})")

# Question answering
qa = pipeline("question-answering")
context = "The Eiffel Tower is located in Paris and was completed in 1889 for the World's Fair."
answer = qa(question="When was the Eiffel Tower completed?", context=context)
print(answer["answer"])

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
long_text = """
The Amazon rainforest, often referred to as the lungs of the Earth, spans over 5.5 million
square kilometers across nine countries in South America. It produces 20% of the world's oxygen
and is home to 10% of all species on the planet. Deforestation has accelerated over the past
decade due to agricultural expansion, logging, and infrastructure development.
"""
summary = summarizer(long_text, max_length=60, min_length=20, do_sample=False)
print(summary[0]["summary_text"])
```

> **When to use:** Use the pipeline API for rapid prototyping, local inference without API costs, or when data privacy requirements prevent sending text to external APIs. For production scale, consider using Hugging Face Inference Endpoints or the Inference API.

---

## 17. Hugging Face — Inference Providers

Hugging Face Inference Providers let you run models hosted on the Hugging Face Hub via API calls, eliminating the need to download and run models locally. You authenticate with your HF token and choose a provider (e.g., `hf-inference`, `together`, `replicate`).

Note that `os.environ` is a dictionary — use bracket notation `os.environ["KEY"]` to read from it, not parentheses.

```python
import os
from huggingface_hub import InferenceClient

# Correct: os.environ is a dict, use bracket notation to access a key
hf_token = os.environ["HF_TOKEN"]

client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token,
)

# Text generation via the inference API
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "user", "content": "Explain the concept of entropy in thermodynamics."}
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)
```

### Using Different Providers

You can swap providers to access different model deployments:

```python
# Using Together AI as the inference provider
together_client = InferenceClient(
    provider="together",
    api_key=os.environ["HF_TOKEN"],
)

response = together_client.chat.completions.create(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
    messages=[{"role": "user", "content": "What are the key differences between Python 2 and Python 3?"}],
    max_tokens=400,
)

print(response.choices[0].message.content)
```

> **When to use:** Use Inference Providers when you want access to open-source models (Llama, Mistral, Falcon) without managing GPU infrastructure. Ideal for comparing multiple model families or when OpenAI's models do not fit your licensing requirements.

---

## 18. Hugging Face — Datasets

The `datasets` library provides access to thousands of public datasets and a consistent API for loading, slicing, and preprocessing them. Datasets are memory-efficient thanks to Apache Arrow under the hood.

```python
from datasets import load_dataset

# Load the IMDb movie review dataset
dataset = load_dataset("imdb")

print(dataset)
# DatasetDict with train (25000) and test (25000) splits

# Access a specific split
train_data = dataset["train"]
print(f"Training examples: {len(train_data)}")
print(f"Features: {train_data.features}")

# Access a single example
example = train_data[0]
print(f"Label: {example['label']}")  # 0 = negative, 1 = positive
print(f"Text preview: {example['text'][:200]}")
```

### Slicing and Filtering

```python
# Take a slice of the dataset
sliced = train_data.select(range(10))
print(sliced[0]["text"])  # access by index, then column name

# Filter to only positive reviews
positive_reviews = train_data.filter(lambda example: example["label"] == 1)
print(f"Positive reviews: {len(positive_reviews)}")

# Map a transformation over the dataset
def preprocess(example):
    example["text_length"] = len(example["text"].split())
    return example

processed = train_data.map(preprocess)
print(f"Average word count: {sum(processed['text_length']) / len(processed):.0f}")
```

### Loading Custom Datasets

```python
import pandas as pd
from datasets import Dataset

# Convert a pandas DataFrame to a Hugging Face Dataset
df = pd.DataFrame({
    "text": ["Great product", "Terrible experience", "Average quality"],
    "label": [1, 0, 1]
})

custom_dataset = Dataset.from_pandas(df)
print(custom_dataset[0])
```

---

## 19. Text Classification

Text classification assigns predefined categories to input text. Hugging Face supports both standard classification (model trained on specific labels) and zero-shot classification (generalizes to arbitrary labels).

### Standard Sentiment Classification

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "The new update completely broke my workflow.",
    "Best purchase I've made all year!",
    "It's okay, nothing special."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Text: {text[:50]}...")
    print(f"Label: {result['label']}, Score: {result['score']:.4f}\n")
```

### Zero-Shot Classification

Zero-shot classification uses a model trained on natural language inference to classify text into categories it was never explicitly trained on. You provide the candidate labels at inference time.

```python
from transformers import pipeline

zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "Scientists have discovered a new exoplanet with conditions potentially suitable for liquid water."

candidate_labels = ["astronomy", "politics", "sports", "technology", "entertainment"]

result = zero_shot(text, candidate_labels=candidate_labels)

print(f"Top Label: {result['labels'][0]} with score: {result['scores'][0]:.4f}")

for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label}: {score:.4f}")
```

---

## 20. Text Summarization

Summarization condenses long documents into shorter versions. There are two approaches: extractive (pulls key sentences directly from the source) and abstractive (generates new text that captures the meaning). Hugging Face's BART and T5 models use the abstractive approach.

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
Electric vehicles (EVs) are rapidly gaining market share globally, driven by falling battery costs,
government incentives, and growing consumer awareness of climate change. In 2023, EV sales surpassed
14 million units worldwide, representing nearly 18% of all new car sales. Major automakers including
Ford, GM, Volkswagen, and Toyota have announced multi-billion dollar investments to transition their
production lines. However, challenges remain: charging infrastructure is still sparse in rural areas,
range anxiety persists among potential buyers, and the electricity grid in many regions is not yet
clean enough to make EVs truly zero-emission throughout their lifecycle.
"""

summary = summarizer(
    article,
    max_length=80,
    min_length=30,
    do_sample=False  # deterministic summarization
)

print("Original length:", len(article.split()), "words")
print("Summary:", summary[0]["summary_text"])
print("Summary length:", len(summary[0]["summary_text"].split()), "words")
```

### Handling Long Documents

Most summarization models have a token limit (typically 1024). For longer documents, chunk and summarize recursively:

```python
def summarize_long_document(text: str, chunk_size: int = 800) -> str:
    """Split long text into chunks, summarize each, then summarize the summaries."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    words = text.split()

    # Split into chunks
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    chunk_summaries = []

    for chunk in chunks:
        result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        chunk_summaries.append(result[0]["summary_text"])

    # If multiple chunks, summarize the combined summaries
    if len(chunk_summaries) > 1:
        combined = " ".join(chunk_summaries)
        final = summarizer(combined, max_length=150, min_length=50, do_sample=False)
        return final[0]["summary_text"]

    return chunk_summaries[0]
```

---

## 21. Auto Models and Tokenizers

`AutoModelForSequenceClassification` and `AutoTokenizer` are flexible classes that automatically select the correct model architecture and tokenizer based on the checkpoint name. Use these when you need more control than the `pipeline` API provides — for example, when fine-tuning or when you need raw logits.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "The movie had brilliant cinematography but a confusing plot."

# Tokenize the input
inputs = tokenizer(
    text,
    return_tensors="pt",       # PyTorch tensors
    truncation=True,
    padding=True,
    max_length=512
)

# Run inference without computing gradients
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
probabilities = torch.softmax(outputs.logits, dim=-1)
predicted_class_id = probabilities.argmax().item()
label = model.config.id2label[predicted_class_id]
confidence = probabilities[0][predicted_class_id].item()

print(f"Prediction: {label} ({confidence:.4f} confidence)")
print(f"All scores: { {model.config.id2label[i]: f'{p:.4f}' for i, p in enumerate(probabilities[0].tolist())} }")
```

### Loading Different Model Types

```python
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer
)

# For text generation (GPT-style)
gen_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gen_model = AutoModelForCausalLM.from_pretrained("gpt2")

# For summarization (encoder-decoder)
sum_tokenizer = AutoTokenizer.from_pretrained("t5-small")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# For named entity recognition
ner_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
```

---

## 22. Document Q&A

Document Q&A extracts answers directly from a provided context passage. The model finds the span of text in the context that best answers the question — it does not generate new information.

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """
Python was created by Guido van Rossum and first released in 1991. It is designed to be
readable and concise, emphasizing code readability with the use of significant indentation.
Python supports multiple programming paradigms, including structured, object-oriented, and
functional programming. It is dynamically typed and garbage-collected, and its comprehensive
standard library is often referred to as one of its greatest strengths.
"""

questions = [
    "Who created Python?",
    "When was Python first released?",
    "What type of typing does Python use?",
    "What is Python's standard library known for?"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']} (confidence: {result['score']:.3f})\n")
```

### Q&A Over a PDF Document

```python
import os
from pypdf import PdfReader
from transformers import pipeline

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())


def answer_from_pdf(pdf_path: str, question: str) -> dict:
    """Answer a question using text extracted from a PDF."""
    context = extract_pdf_text(pdf_path)
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return qa(question=question, context=context[:3000])  # truncate to model limit
```

---

## 23. OpenAI Embeddings

Embeddings are dense numeric vectors that represent the semantic meaning of text. Texts with similar meanings have vectors that are close together in high-dimensional space, regardless of the specific words used.

The OpenAI Embeddings API converts text into these vectors using models like `text-embedding-3-small` and `text-embedding-3-large`. You can use embeddings to build semantic search engines, recommendation systems, clustering pipelines, and classification tools.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Return the embedding vector for a single text string."""
    text = text.replace("\n", " ")  # newlines can degrade embedding quality
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed a list of texts in a single API call (more efficient than looping)."""
    cleaned = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=cleaned, model=model)
    return [item.embedding for item in response.data]


# Single embedding
vector = get_embedding("The quick brown fox jumps over the lazy dog.")
print(f"Embedding dimensions: {len(vector)}")  # 1536 for text-embedding-3-small

# Batch embeddings
texts = ["Machine learning", "Deep neural networks", "Soccer match results"]
vectors = get_embeddings_batch(texts)
print(f"Embedded {len(vectors)} texts, each with {len(vectors[0])} dimensions.")
```

**Embedding model comparison:**

| Model | Dimensions | Best For |
|---|---|---|
| `text-embedding-3-small` | 1536 | Cost-efficient general use |
| `text-embedding-3-large` | 3072 | Highest accuracy requirements |
| `text-embedding-ada-002` | 1536 | Legacy — use 3-small instead |

> **When to use:** Use embeddings when you need semantic similarity rather than keyword matching. They excel at tasks like "find documents related to this query" where exact word overlap is insufficient.

---

## 24. Semantic Search with Embeddings

Semantic search uses embedding similarity to find the most relevant items for a query — even if they share no words in common. The standard similarity metric is cosine similarity, which measures the angle between two vectors.

The following example builds a small in-memory semantic search engine:

```python
import os
import json
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Embed a single text string."""
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model=model
    )
    return response.data[0].embedding


# Build an enriched embedding store with metadata
documents = [
    "Python is a versatile programming language used for web development, data science, and automation.",
    "The Eiffel Tower stands 330 meters tall and was built for the 1889 World's Fair in Paris.",
    "Neural networks are inspired by the structure of biological brains and learn from data.",
    "The Amazon River flows through South America and discharges into the Atlantic Ocean.",
    "Machine learning algorithms improve their performance automatically through experience.",
]

# Embed all documents once and store them
print("Building embedding index...")
embedded_documents = []
for index, doc in enumerate(documents):
    embedding = get_embedding(doc)
    embedded_documents.append({
        "id": index,
        "text": doc,
        "embedding": embedding
    })

print(f"Indexed {len(embedded_documents)} documents.\n")


def find_n_closest(query: str, embedded_docs: list, n: int = 3) -> list:
    """Find the n most semantically similar documents to a query."""
    query_embedding = get_embedding(query)
    distances = []

    for doc in embedded_docs:
        distance = cosine(query_embedding, doc["embedding"])
        distances.append({
            "id": doc["id"],
            "text": doc["text"],
            "distance": distance,
            "similarity": 1 - distance  # convert to similarity score
        })

    # Sort by ascending distance (lower = more similar), then return top n
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    return distances_sorted[0:n]


# Run a semantic search query
query = "What programming language is good for AI projects?"
results = find_n_closest(query, embedded_documents, n=3)

print(f"Query: {query}\n")
for rank, result in enumerate(results, 1):
    print(f"  {rank}. (similarity: {result['similarity']:.3f}) {result['text'][:80]}...")
```

---

## 25. Recommendation Systems

A recommendation system built on embeddings surfaces items semantically related to a given input. Each item is embedded once at indexing time; at query time, the input is embedded and compared against the stored vectors.

```python
import os
from openai import OpenAI
from scipy.spatial.distance import cosine

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# Article catalog with metadata
articles = [
    {"id": 1, "headline": "OpenAI Releases GPT-5 with Enhanced Reasoning", "topic": "AI", "keywords": "OpenAI, GPT, language model"},
    {"id": 2, "headline": "Scientists Discover New Species in Deep Ocean", "topic": "Science", "keywords": "ocean, biology, discovery"},
    {"id": 3, "headline": "Stock Markets Hit Record Highs Amid Tech Rally", "topic": "Finance", "keywords": "stocks, market, technology"},
    {"id": 4, "headline": "Machine Learning Accelerates Drug Discovery", "topic": "AI", "keywords": "AI, healthcare, drug, ML"},
    {"id": 5, "headline": "New Coral Reef Found Near Australian Coast", "topic": "Science", "keywords": "ocean, reef, environment"},
    {"id": 6, "headline": "Python Overtakes JavaScript in Developer Surveys", "topic": "Tech", "keywords": "Python, programming, developer"},
]

# Build the embedding index
print("Indexing articles...")
for article in articles:
    combined_text = f"{article['headline']} {article['keywords']}"
    article["embedding"] = get_embedding(combined_text)

print(f"Indexed {len(articles)} articles.\n")


def recommend_articles(query: str, catalog: list, n: int = 3) -> list:
    """Return the top n most relevant articles for a query."""
    query_embedding = get_embedding(query)

    scored = []
    for article in catalog:
        similarity = 1 - cosine(query_embedding, article["embedding"])
        scored.append({**article, "similarity": similarity})

    return sorted(scored, key=lambda x: x["similarity"], reverse=True)[:n]


user_interest = "advances in artificial intelligence and neural networks"
hits = recommend_articles(user_interest, articles, n=3)

print(f"Recommendations for: '{user_interest}'\n")
print("Here are some recommendations:")
for hit in hits:
    print(f"  - {hit['headline']} (topic: {hit['topic']}, keywords: {hit['keywords']}, score: {hit['similarity']:.3f})")
```

---

## 26. Embeddings for Classification

You can classify text without a trained classifier by embedding both the text and each candidate label, then finding the label whose embedding is closest to the text embedding. This is a lightweight, zero-shot approach that requires no labeled training data.

```python
import os
from openai import OpenAI
from scipy.spatial.distance import cosine

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def find_closest_label(text: str, labels: list) -> dict:
    """Return the label whose embedding is closest to the text embedding."""
    text_embedding = get_embedding(text)
    scores = []

    for label in labels:
        label_embedding = get_embedding(label["label"])
        similarity = 1 - cosine(text_embedding, label_embedding)
        scores.append({"label": label["label"], "similarity": similarity})

    return sorted(scores, key=lambda x: x["similarity"], reverse=True)[0]


# Define category labels (no semicolons — clean Python dicts)
categories = [
    {"label": "Tech"},
    {"label": "Sports"},
    {"label": "Politics"},
    {"label": "Entertainment"},
    {"label": "Finance"},
    {"label": "Science"},
]

# Classify a set of headlines
test_headlines = [
    "Congress passes new infrastructure spending bill",
    "Tesla stock surges after record quarterly earnings",
    "NASA announces new Mars rover mission for 2030",
    "The Lakers win their third consecutive championship",
    "New AI chip outperforms GPUs in benchmark tests",
]

print("Embedding-based classification results:\n")
for headline in test_headlines:
    result = find_closest_label(headline, categories)
    print(f"  [{result['label']}] {headline}")
    print(f"    Similarity: {result['similarity']:.4f}\n")
```

---

## 27. Vector Databases — ChromaDB

ChromaDB is an open-source, embedded vector database designed for AI applications. It stores documents alongside their embeddings and metadata, enabling fast similarity search with optional metadata filtering. ChromaDB runs in-process (no server required) for development and can persist to disk or run as a server in production.

### Setting Up ChromaDB

```python
import chromadb

# In-memory client (ephemeral — data lost on process exit)
client = chromadb.Client()

# Persistent client (data saved to disk)
persistent_client = chromadb.PersistentClient(path="./chroma_storage")
```

### Creating a Collection and Adding Documents

A collection in ChromaDB is analogous to a table in a relational database. You can let ChromaDB generate embeddings automatically (using its built-in embedding function) or supply your own.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_storage")

# Create or retrieve a collection
collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

# Add documents — ChromaDB embeds them automatically
documents = [
    "Python is a high-level programming language known for its readability.",
    "The Great Wall of China stretches over 21,000 kilometers.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Bitcoin was the first decentralized cryptocurrency, created in 2009.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[
        {"source": "programming", "category": "tech"},
        {"source": "history", "category": "geography"},
        {"source": "biology", "category": "science"},
        {"source": "finance", "category": "crypto"},
        {"source": "biology", "category": "science"},
    ]
)

print(f"Collection contains {collection.count()} documents.")
```

### Querying the Collection

```python
# Query by natural language
results = collection.query(
    query_texts=["how do plants make food?"],
    n_results=2,
    include=["documents", "distances", "metadatas"]
)

for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
    print(f"Result {i+1} (distance: {distance:.4f}): {doc}")
    print(f"  Metadata: {results['metadatas'][0][i]}\n")
```

> **When to use:** Use ChromaDB for local development, prototypes, and small-to-medium scale production applications (up to a few million vectors). Its in-process nature means zero infrastructure overhead. For massive scale or multi-region deployments, consider Pinecone or Weaviate.

---

## 28. ChromaDB — Querying and Filtering

ChromaDB supports rich filtering on metadata using a MongoDB-style query syntax. You can combine similarity search with exact metadata filters to narrow results.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_or_create_collection("knowledge_base")

# Filter by exact metadata match
science_results = collection.query(
    query_texts=["cellular biology processes"],
    n_results=3,
    where={"category": "science"},          # metadata filter
    include=["documents", "distances", "metadatas"]
)

print("Science category results:")
for doc, distance in zip(science_results["documents"][0], science_results["distances"][0]):
    print(f"  {doc[:70]}... (distance: {distance:.4f})")

# Filter using operators
tech_or_crypto = collection.query(
    query_texts=["digital currency"],
    n_results=5,
    where={"category": {"$in": ["tech", "crypto"]}},  # match multiple values
    include=["documents", "metadatas"]
)

print("\nTech or crypto results:")
for doc, meta in zip(tech_or_crypto["documents"][0], tech_or_crypto["metadatas"][0]):
    print(f"  [{meta['category']}] {doc[:70]}...")
```

### Updating and Deleting Documents

```python
# Update document metadata
collection.update(
    ids=["doc_0"],
    metadatas=[{"source": "programming", "category": "tech", "reviewed": True}]
)

# Delete a document
collection.delete(ids=["doc_4"])
print(f"Collection now contains {collection.count()} documents.")

# Retrieve a document by ID without a query
retrieved = collection.get(
    ids=["doc_0"],
    include=["documents", "metadatas"]
)
print(retrieved["documents"][0])
```

### Getting All Documents

```python
# Retrieve all documents (use with caution on large collections)
all_docs = collection.get(
    include=["documents", "metadatas", "embeddings"]
)
print(f"Total documents: {len(all_docs['documents'])}")
```

---

## 29. Vector Databases — Pinecone

Pinecone is a managed, serverless vector database built for production scale. Unlike ChromaDB, Pinecone is a cloud service — you interact with it via API. It handles indexing, sharding, replication, and scaling automatically.

### Setup and Index Creation

```python
import os
from pinecone import Pinecone, ServerlessSpec

# Correct: assign the Pinecone instance to pc
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "semantic-search-index"

# Create the index if it does not already exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,          # must match your embedding model's output dimension
        metric="cosine",         # distance metric: cosine, euclidean, or dotproduct
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created index: {index_name}")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index
index = pc.Index(index_name)
print(index.describe_index_stats())
```

### Upserting Vectors

Upsert inserts a vector if its ID does not exist, or updates it if it does. Each vector requires a unique ID, the embedding values, and optional metadata.

```python
import os
from openai import OpenAI
from pinecone import Pinecone

oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("semantic-search-index")

def embed(text: str) -> list[float]:
    response = oai_client.embeddings.create(
        input=text.replace("\n", " "),
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

documents = [
    {"id": "doc-001", "text": "Quantum computing uses qubits to perform calculations.", "category": "science"},
    {"id": "doc-002", "text": "The Python requests library simplifies HTTP calls.", "category": "tech"},
    {"id": "doc-003", "text": "The Federal Reserve controls US monetary policy.", "category": "finance"},
]

vectors_to_upsert = []
for doc in documents:
    embedding = embed(doc["text"])
    vectors_to_upsert.append({
        "id": doc["id"],
        "values": embedding,
        "metadata": {"text": doc["text"], "category": doc["category"]}
    })

index.upsert(vectors=vectors_to_upsert)
print(f"Upserted {len(vectors_to_upsert)} vectors.")
```

### Querying Pinecone

```python
query = "central banking and interest rates"
query_embedding = embed(query)

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True    # correct parameter name (not include_metadatas)
)

print(f"Query: {query}\n")
for match in results["matches"]:
    print(f"  ID: {match['id']}, Score: {match['score']:.4f}")
    print(f"  Text: {match['metadata']['text']}\n")
```

---

## 30. Pinecone — Performance Tuning

When working with large document collections (tens of thousands to millions of vectors), batching upserts and parallelizing requests are essential for acceptable ingestion throughput.

### Chunking Large Documents

Long documents should be split into smaller chunks before embedding. This improves retrieval precision because the model returns the specific relevant passage rather than an entire document.

```python
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


# Example: chunk and embed a long document
long_doc = " ".join(["word"] * 600)  # simulated 600-word document
chunks = chunk_text(long_doc, chunk_size=200, overlap=50)
print(f"Created {len(chunks)} chunks from the document.")
```

### Batch Upsert with Parallel Processing

```python
import os
import concurrent.futures
from openai import OpenAI
from pinecone import Pinecone

oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("semantic-search-index")

def embed(text: str) -> list[float]:
    response = oai_client.embeddings.create(
        input=text.replace("\n", " "),
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def upsert_batch(batch: list) -> str:
    """Upsert a single batch of vectors. Called by each thread."""
    index.upsert(vectors=batch)
    return f"Upserted batch of {len(batch)}"


def parallel_upsert(texts: list[str], batch_size: int = 100, max_workers: int = 5):
    """Embed texts and upsert to Pinecone in parallel batches."""
    # Embed all texts (consider batching embedding calls too for large volumes)
    vectors = []
    for i, text in enumerate(texts):
        embedding = embed(text)
        vectors.append({
            "id": f"vec-{i:06d}",
            "values": embedding,
            "metadata": {"text": text[:500]}  # store truncated text as metadata
        })

    # Split into batches
    batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
    print(f"Upserting {len(vectors)} vectors across {len(batches)} batches...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upsert_batch, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            print(f"  {future.result()}")

    print("Parallel upsert complete.")


# Count total vectors to upsert
sample_texts = [f"Document number {i} about various interesting topics." for i in range(250)]
parallel_upsert(sample_texts, batch_size=100, max_workers=3)
```

---

## 31. Semantic Search with Pinecone

Combining Pinecone with OpenAI embeddings gives you a production-ready semantic search pipeline that scales to millions of documents. The following is a complete, end-to-end example.

```python
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "article-search"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def get_embedding(text: str) -> list[float]:
    response = oai_client.embeddings.create(
        input=text.replace("\n", " "),
        model=EMBED_MODEL
    )
    return response.data[0].embedding


def setup_index() -> object:
    """Create the Pinecone index if it does not exist, then return it."""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)


def index_articles(index, articles: list[dict]):
    """Embed and upsert a list of article dicts."""
    vectors = []
    for article in articles:
        text = f"{article['title']} {article['body']}"
        embedding = get_embedding(text)
        vectors.append({
            "id": article["id"],
            "values": embedding,
            "metadata": {
                "title": article["title"],
                "category": article["category"],
                "body": article["body"][:400]
            }
        })
    index.upsert(vectors=vectors)
    print(f"Indexed {len(vectors)} articles.")


def semantic_search(index, query: str, top_k: int = 5, category_filter: str = None) -> list:
    """Search the index with an optional metadata filter."""
    query_embedding = get_embedding(query)

    filter_dict = {}
    if category_filter:
        filter_dict = {"category": {"$eq": category_filter}}

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    return results["matches"]


# --- Main workflow ---

index = setup_index()

sample_articles = [
    {"id": "a1", "title": "Advances in Transformer Architecture", "body": "Researchers have introduced sparse attention mechanisms that reduce memory usage by 60% while maintaining accuracy.", "category": "AI"},
    {"id": "a2", "title": "Federal Reserve Raises Interest Rates Again", "body": "The Fed increased rates by 25 basis points in response to persistent inflation figures.", "category": "Finance"},
    {"id": "a3", "title": "New Vaccine Shows 94% Efficacy in Trials", "body": "Clinical trials of a novel mRNA vaccine demonstrated strong protection against multiple strains.", "category": "Health"},
    {"id": "a4", "title": "SpaceX Launches 60 New Starlink Satellites", "body": "The latest launch brings total Starlink constellation to over 5,000 active satellites.", "category": "Space"},
    {"id": "a5", "title": "Python 3.13 Released with Performance Improvements", "body": "The new Python release includes a faster interpreter and improved memory management.", "category": "Tech"},
]

index_articles(index, sample_articles)

# Broad semantic search
print("\n--- Broad Search ---")
query = "machine learning breakthroughs"
matches = semantic_search(index, query, top_k=3)
for m in matches:
    print(f"  [{m['score']:.3f}] {m['metadata']['title']}")

# Filtered semantic search
print("\n--- Filtered Search (Finance only) ---")
finance_matches = semantic_search(index, "monetary policy and banking", top_k=3, category_filter="Finance")
for m in finance_matches:
    print(f"  [{m['score']:.3f}] {m['metadata']['title']}")
```

---

## 32. Quick Reference Cheat Sheet

A consolidated reference for the most common operations covered in this guide.

| Operation | Code Pattern |
|---|---|
| **OpenAI Chat Completion** | `client.chat.completions.create(model=..., messages=[...])` |
| **Get response text** | `response.choices[0].message.content` |
| **Check token usage** | `response.usage.total_tokens` |
| **JSON mode** | `response_format={"type": "json_object"}` |
| **Enable streaming** | `stream=True` then `for chunk in response:` |
| **Function calling** | `tools=[{"type": "function", "function": {...}}]` |
| **OpenAI embedding** | `client.embeddings.create(input=text, model="text-embedding-3-small")` |
| **HF Pipeline (sentiment)** | `pipeline("sentiment-analysis")` |
| **HF Pipeline (zero-shot)** | `pipeline("zero-shot-classification", model="facebook/bart-large-mnli")` |
| **HF Pipeline (summarize)** | `pipeline("summarization", model="facebook/bart-large-cnn")` |
| **HF Pipeline (QA)** | `pipeline("question-answering")` |
| **HF Inference Provider** | `InferenceClient(provider="hf-inference", api_key=os.environ["HF_TOKEN"])` |
| **Load HF Dataset** | `load_dataset("imdb")` |
| **ChromaDB in-memory** | `chromadb.Client()` |
| **ChromaDB persistent** | `chromadb.PersistentClient(path="./storage")` |
| **ChromaDB add docs** | `collection.add(documents=[...], ids=[...], metadatas=[...])` |
| **ChromaDB query** | `collection.query(query_texts=[...], n_results=N, include_metadata=True)` |
| **ChromaDB filter** | `where={"category": "science"}` |
| **Pinecone init** | `pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))` |
| **Pinecone create index** | `pc.create_index(name=..., dimension=1536, metric="cosine", spec=ServerlessSpec(...))` |
| **Pinecone upsert** | `index.upsert(vectors=[{"id": ..., "values": [...], "metadata": {...}}])` |
| **Pinecone query** | `index.query(vector=[...], top_k=5, include_metadata=True)` |
| **Count tokens** | `tiktoken.encoding_for_model("gpt-4o").encode(text)` |
| **Retry on rate limit** | `@retry(retry=retry_if_exception_type(openai.RateLimitError), wait=wait_exponential(...))` |
| **Auth error handling** | `except openai.AuthenticationError:` |
| **Load .env file** | `from dotenv import load_dotenv; load_dotenv()` |
| **Read env variable** | `os.getenv("OPENAI_API_KEY")` or `os.environ["OPENAI_API_KEY"]` |
| **Semantic similarity** | `1 - cosine(embedding_a, embedding_b)` (requires `scipy`) |

---

*This guide covers the core patterns for production AI development with OpenAI and Hugging Face. For the latest model names, pricing, and API changes, always refer to [platform.openai.com/docs](https://platform.openai.com/docs) and [huggingface.co/docs](https://huggingface.co/docs).*
