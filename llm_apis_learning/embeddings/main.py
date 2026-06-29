from google import genai
from google.genai import types

from dotenv import load_dotenv
import os

import math

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embedding_call():
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="What is the capital of France?",
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    ); return result.embeddings[0].values

# print(embedding_call())

# ---

def embed_docs(docs: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=docs,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    ); return [e.values for e in result.embeddings]

def cosine_similarlity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def search(query: str, docs: list[str], doc_embeddings: list[list[float]], top_k: int = 3):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    ); query_embedding = result.embeddings[0].values # Find embedding value of user's query

    scored = [] # (score, doc)
    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarlity(query_embedding, doc_emb)
        # ^ Compare embedding value of query w/ every doc's embedding individually for simiarlity
        scored.append((score, docs[i]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k] # Return top k results that match embedding values best

# ---

docs = [
    "Python is a high-level programming language known for readability.",
    "Machine learning models learn patterns from training data.",
    "Next.js is a React framework for building full-stack web applications.",
    "Supabase is an open-source Firebase alternative built on PostgreSQL.",
    "Neural networks are composed of layers of interconnected nodes.",
    "FastAPI is a modern Python web framework for building APIs.",
    "Embeddings represent the semantic meaning of text as vectors.",
    "PostgreSQL supports vector similarity search via the pgvector extension.",
    "Backpropagation computes gradients by applying the chain rule.",
    "RAG combines retrieval with generation to ground LLM responses in facts.",
]

print("Embedding docs...")
doc_embeddings = embed_docs(docs)
print(f"Embedded {len(docs)} documents, vector size: {len(doc_embeddings[0])}")
# ^ Index doesn't matter since every document should have the same vector size length

while True:
    query = input("\n Search: ").strip()
    if query.lower() in ("quit", "exit"): break

    results = search(query, docs, doc_embeddings, 5)
    print("\n Top Results: ")
    for score, doc in results:
        print(f"[{score:.4f}] {doc}")

# The two retrieval types tell the model how the text will be used, but dimensionality and vector is the same structure all throughout

# Docs embedded before search & querying.
# Docs embedd once > stored in a vector DB > query is embedded on each search multiple times

# In real scenarios, instead of comparing with every single document, it searches for nearest neighbors