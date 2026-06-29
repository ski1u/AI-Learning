from google import genai
from google.genai import types

import math
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

embed_dim = 768 # Must match Supabase's database column

# ---

def normalize(vec: list[float]) -> list[float]:
    # Because Gemini normalizes the 3072 default dimension, truncating it to 768 messes up the values,
    # so normalization is needed here
    n = math.sqrt(sum(x * x for x in vec))
    return [x/n for x in vec] if n > 0 else vec

def embed_docs(docs: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=docs,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=embed_dim
        )
    )
    if result.embeddings is None: return []
    # return [v for e in result.embeddings if (v := e.values) is not None] # Without normalization
    return [normalize(v) for e in result.embeddings if (v := e.values) is not None]

# v = embed_docs(["The first planet from the sun"])[0]
# print(len(v)) # 768
# print(round(sum(x * x for x in v) ** 0.5, 4))
# ^ Should show 1.0. Without normalization, it would be a greater value like 30.0