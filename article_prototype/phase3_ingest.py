import json
import yaml
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import uuid

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# We will use Ollama's local OpenAI-compatible endpoint with an available model
OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL = config["models"]["embedding"]  # Get embedding model from config
COLLECTION_NAME = "article_chunks"


def embed_text(client: OpenAI, text: str) -> list[float]:
    """Generates an embedding vector using Ollama's OpenAI compat API."""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def main():
    input_file = Path("output/enriched_chunks.json")
    if not input_file.exists():
        print(f"File {input_file} not found. Run phase2_enrich.py first.")
        return

    with open(input_file, "r") as f:
        chunks = json.load(f)

    # Connect to local Qdrant
    qdrant = QdrantClient(path="qdrant_db")
    openai_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    # Getting embedding dimension
    print("Initializing embedding test to find dimension...")
    test_emb = embed_text(openai_client, "test")
    emb_dim = len(test_emb)
    print(f"Embedding dimension is {emb_dim}")

    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE),
    )

    points = []
    print(f"Embedding {len(chunks)} chunks and uploading to Qdrant...")
    for i, chunk in enumerate(chunks):
        # We prefer the raw text, but if caption is present, we boost it.
        # Images/Tables have their markdown text embedded
        text_content = chunk.get("text", "")
        if not text_content:
            continue

        vector = embed_text(openai_client, text_content)

        # Save structural metadata but exclude large base64 strings so they don't bloat memory
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "element_types": chunk["element_types"],
            "source_file": chunk["source_file"],
            "modality": chunk.get("modality", "text"),
        }

        # If it's an image, keep the base64 reference for retrieval rendering
        if chunk.get("image_base64"):
            metadata["has_image"] = True
            # Optional: store truncated or full image base64 here if needed by UI
            metadata["image_base64"] = chunk["image_base64"]

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text_content, **metadata},
            )
        )
        print(
            f"Embedded chunk {i + 1}/{len(chunks)} (modality: {chunk.get('modality')})"
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(
        f"Successfully ingested {len(points)} chunks into Qdrant index '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    main()
