import yaml
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
import warnings

warnings.filterwarnings("ignore")

with open(Path(__file__).parent / "config.yaml") as f:
    config = yaml.safe_load(f)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL = config["models"]["embedding"]
LLM_MODEL = config["models"]["llm"]
CROSS_ENCODER_MODEL = config["models"]["cross_encoder"]
COLLECTION_NAME = "article_chunks"

qdrant_path = config.get("qdrant", {}).get("path", "qdrant_db")
qdrant = QdrantClient(path=qdrant_path)
openai_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)


def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def generate_answer(query: str, contexts: list[dict]) -> str:
    context_str = ""
    for i, ctx in enumerate(contexts):
        modality_badge = f"[{ctx['modality'].upper()}]" if "modality" in ctx else "[TEXT]"
        page = ctx.get("page", "?")
        context_str += f"--- Chunk {i + 1} (Page {page}) {modality_badge} ---\n{ctx['text']}\n\n"

    prompt = f"""You are an advanced AI assistant powered by Multimodal Structural RAG.
Use the provided context chunks optimally to answer the user's question.

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
            results = qdrant.search(
                collection_name=COLLECTION_NAME, query_vector=q_emb, limit=20
            )

            if not results:
                print("No results found in Qdrant.")
                continue

            visual_keywords = {"diagram", "flowchart", "figure", "image", "chart", "visual", "illustration", "picture", "encoder", "decoder"}
            query_words = set(query.lower().split())
            is_visual_query = bool(query_words & visual_keywords)
            if is_visual_query:
                print("   [Visual query detected - boosting image modality]")
                for hit in results:
                    if hit.payload.get("modality") == "image":
                        hit.score *= 1.35

            if is_visual_query:
                print("   Skipping cross-encoder for visual query (boosting sufficient)")
            else:
                print(f"2. Re-Ranking ({len(results)} candidates)...")
                pairs = [[query, hit.payload.get("text", "")] for hit in results]
                scores = cross_encoder.predict(pairs)
                for hit, score in zip(results, scores):
                    hit.score = float(score)
                results.sort(key=lambda x: x.score, reverse=True)

            top_k = 4
            best_hits = results[:top_k]

            print(f"\n=> Top {top_k} chunks isolated!")
            for i, hit in enumerate(best_hits):
                modality = hit.payload.get("modality", "text")
                print(f"  {i + 1}. [Score: {hit.score:.2f}] ({modality}, Page {hit.payload.get('page')}) -> {hit.payload.get('text')[:60]}...")

            print("\n3. Synthesizing Answer with LLM...")
            answer = generate_answer(query, [h.payload for h in best_hits])

            print(f"\n========== ANSWER ==========\n{answer}\n============================")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    interactive_search()