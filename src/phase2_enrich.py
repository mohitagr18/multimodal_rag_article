import json
import base64
import yaml
from pathlib import Path
import fitz
import requests
from PIL import Image

with open(Path(__file__).parent / "config.yaml") as f:
    config = yaml.safe_load(f)

RESULTS_DIR = Path(config.get("directories", {}).get("results", "results"))
INPUT_DIR = Path(config.get("directories", {}).get("input", "input"))
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = config["models"]["vlm"]


def get_base64_from_fitz_rect(page: fitz.Page, bbox: list[float]) -> str:
    """Crop PDF region and return as base64 PNG."""
    w, h = page.rect.width, page.rect.height
    x1, y1, x2, y2 = bbox
    rect = fitz.Rect(x1 * w / 1000.0, y1 * h / 1000.0, x2 * w / 1000.0, y2 * h / 1000.0)
    pix = page.get_pixmap(clip=rect, dpi=150)
    img_data = pix.tobytes("png")
    return base64.b64encode(img_data).decode("utf-8")


def caption_image(base64_img: str) -> str:
    """Generate VLM caption via Ollama."""
    prompt = "Please provide a very brief, rich description of the contents of this image or chart. Focus on the core message."
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [base64_img],
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Failed to generate caption: {e}")
        return ""


def main():
    input_file = RESULTS_DIR / "structured_chunks.json"
    output_file = RESULTS_DIR / "enriched_chunks.json"

    if not input_file.exists():
        print(f"Input file {input_file} not found.")
        return

    with open(input_file, "r") as f:
        chunks = json.load(f)

    docs = {}
    print(f"Enriching {len(chunks)} chunks...")
    for idx, chunk in enumerate(chunks):
        if chunk.get("bbox") and chunk.get("modality") in ["image", "table", "formula"]:
            pdf_path = INPUT_DIR / chunk["source_file"]
            if not pdf_path.exists():
                pdf_path = Path(chunk["source_file"])
            if not pdf_path.exists():
                print(f"Source file {pdf_path} not found for chunk {chunk['chunk_id']}.")
                continue

            if pdf_path not in docs:
                docs[pdf_path] = fitz.open(pdf_path)

            doc = docs[pdf_path]
            page_num = chunk["page"] - 1
            if page_num < 0 or page_num >= len(doc):
                continue

            page = doc[page_num]
            print(f"[{idx + 1}/{len(chunks)}] Extracting base64 for chunk {chunk['chunk_id']}...")
            b64_image = get_base64_from_fitz_rect(page, chunk["bbox"])
            chunk["image_base64"] = b64_image

            if chunk["modality"] == "image":
                print(f"[{idx + 1}/{len(chunks)}] Generating VLM caption via {MODEL_NAME}...")
                caption = caption_image(b64_image)
                chunk["caption"] = caption
                if caption:
                    chunk["text"] = f"[IMAGE CAPTION] {caption}\n\n[ORIGINAL TEXT] {chunk['text']}"

    for doc in docs.values():
        doc.close()

    with open(output_file, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Successfully saved enriched chunks to {output_file}")


if __name__ == "__main__":
    main()