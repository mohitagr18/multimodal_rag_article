import json
import yaml
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import uuid

with open(Path(__file__).parent / "config.yaml") as f:
    config = yaml.safe_load(f)

RESULTS_DIR = Path(config.get("directories", {}).get("results", "results"))
OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL = config["models"]["embedding"]
COLLECTION_NAME = "article_chunks"


def embed_text(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def main():
    input_file = RESULTS_DIR / "enriched_chunks.json"
    if not input_file.exists():
        print(f"File {input_file} not found. Run phase2_enrich.py first.")
        raise SystemExit(1)

    with open(input_file, "r") as f:
        chunks = json.load(f)

    qdrant_path = config.get("qdrant", {}).get("path", "qdrant_db")
    qdrant = QdrantClient(path=qdrant_path)
    openai_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

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
        text_content = chunk.get("text", "")
        if not text_content:
            continue

        vector = embed_text(openai_client, text_content)
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "element_types": chunk["element_types"],
            "source_file": chunk["source_file"],
            "modality": chunk.get("modality", "text"),
        }

        if chunk.get("image_base64"):
            metadata["has_image"] = True
            metadata["image_base64"] = chunk["image_base64"]

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text_content, **metadata},
            )
        )
        print(f"Embedded chunk {i + 1}/{len(chunks)} (modality: {chunk.get('modality')})")

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully ingested {len(points)} chunks into Qdrant index '{COLLECTION_NAME}'.")
    return qdrant, openai_client


if __name__ == "__main__":
    main()