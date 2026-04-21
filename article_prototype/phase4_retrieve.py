import os
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL = config["models"]["embedding"]
LLM_MODEL = config["models"]["llm"]
CROSS_ENCODER_MODEL = config["models"]["cross_encoder"]
COLLECTION_NAME = "article_chunks"

# 1. Initialize Clients
print("Loading clients and models... (This may take a moment for Cross-Encoder)")
qdrant = QdrantClient(path="qdrant_db")
openai_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

import warnings

warnings.filterwarnings("ignore")

cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)


def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def generate_answer(query: str, contexts: list[dict]) -> str:
    # Build context blocks detailing the modality and text
    context_str = ""
    for i, ctx in enumerate(contexts):
        modality_badge = (
            f"[{ctx['modality'].upper()}]" if "modality" in ctx else "[TEXT]"
        )
        page = ctx.get("page", "?")
        context_str += (
            f"--- Chunk {i + 1} (Page {page}) {modality_badge} ---\n{ctx['text']}\n\n"
        )

    prompt = f"""You are an advanced AI assistant powered by Multimodal Structural RAG.
Use the provided context chunks optimally to answer the user's question. Take into account that some chunks are textual descriptions generated directly from images, diagrams, or tables extracted structurally from the source PDF.

Context:
{context_str}

User Question: {query}

Answer:"""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content


def interactive_search():
    print(f"\n--- Multimodal RAG with Cross-Encoder Reranking ---")
    print("Database: Qdrant")
    print(f"Reranker: {CROSS_ENCODER_MODEL}")
    while True:
        try:
            query = input("\nEnter your query (or 'quit' to exit): ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue

            print("\n1. Dense Retrieval (Fetching top 20 candidates)...")
            q_emb = embed_text(query)
            # Stage 1: Fast Vector Search
            results = qdrant.search(
                collection_name=COLLECTION_NAME, query_vector=q_emb, limit=20
            )

            if not results:
                print("No results found in Qdrant.")
                continue

            # Stage 2: Cross-Encoder Reranking
            print(
                f"2. Re-Ranking (Scoring {len(results)} candidates against the query)..."
            )
            # Pair each chunk text with the user query
            pairs = [[query, hit.payload.get("text", "")] for hit in results]

            # Predict scores
            scores = cross_encoder.predict(pairs)

            # Reattach scores to results and sort
            for hit, score in zip(results, scores):
                hit.score = float(
                    score
                )  # Overwrite dense cosine score with Cross-Encoder score

            # Sort by new Cross-Encoder score descending
            results.sort(key=lambda x: x.score, reverse=True)

            # Take top 4
            top_k = 4
            best_hits = results[:top_k]

            print(f"\n=> Top {top_k} structurally-aware chunks isolated!")
            for i, hit in enumerate(best_hits):
                modality = hit.payload.get("modality", "text")
                print(
                    f"  {i + 1}. [Score: {hit.score:.2f}] (Type: {modality}, Page: {hit.payload.get('page')}) -> {hit.payload.get('text')[:60]}..."
                )

            print("\n3. Synthesizing Answer with LLM...")
            answer = generate_answer(query, [h.payload for h in best_hits])

            print(
                f"\n========== ANSWER ==========\n{answer}\n============================"
            )

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    interactive_search()
