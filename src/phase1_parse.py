import argparse
import json
import logging
from pathlib import Path
from dataclasses import asdict
import fitz
from glmocr import GlmOcr
import yaml

from schemas import ParsedElement, ParseResult, PageResult
from chunker import structure_aware_chunking

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

with open(Path(__file__).parent / "config.yaml") as f:
    config = yaml.safe_load(f)

RESULTS_DIR = Path(config.get("directories", {}).get("results", "results"))
INPUT_DIR = Path(config.get("directories", {}).get("input", "input"))

def naive_baseline(pdf_path: Path):
    """Extracts text naively using PyMuPDF and creates plain text chunks."""
    logger.info("Running Naive Extraction Baseline...")
    doc = fitz.open(pdf_path)
    naive_chunks = []
    chunk_idx = 0
    max_tokens = 512

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        current_text = []
        current_tokens = 0
        
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            tokens = int(len(text.split()) * 1.3)
            if current_tokens + tokens > max_tokens and current_text:
                naive_chunks.append({
                    "chunk_id": f"{pdf_path.name}_{page_num}_{chunk_idx}",
                    "text": "\n\n".join(current_text),
                    "modality": "text",
                    "is_atomic": False
                })
                chunk_idx += 1
                current_text = []
                current_tokens = 0
            current_text.append(text)
            current_tokens += tokens

        if current_text:
            naive_chunks.append({
                "chunk_id": f"{pdf_path.name}_{page_num}_{chunk_idx}",
                "text": "\n\n".join(current_text),
                "modality": "text",
                "is_atomic": False
            })
            chunk_idx += 1

    return naive_chunks


class MockGlmOcr:
    """Mock of GlmOcr layout detection."""
    def parse(self, pdf_path: str, **kwargs):
        json_result = [[
            {"label": "document_title", "content": "Deep Learning for Document Understanding", "bbox_2d": [100,50,900,100], "index": 0},
            {"label": "paragraph_title", "content": "Abstract", "bbox_2d": [100,120,900,140], "index": 1},
            {"label": "paragraph", "content": "We propose a new multimodal document representation pipeline.", "bbox_2d": [100,150,900,200], "index": 2},
            {"label": "paragraph_title", "content": "1. Introduction", "bbox_2d": [100,220,900,240], "index": 3},
            {"label": "paragraph", "content": "Most RAG approaches treat documents as a flat sequence of words.", "bbox_2d": [100,250,900,300], "index": 4},
            {"label": "paragraph_title", "content": "2. Proposed Architecture", "bbox_2d": [100,320,900,340], "index": 5},
            {"label": "figure_title", "content": "Figure 1: Performance comparison.", "bbox_2d": [100,350,900,370], "index": 6},
            {"label": "image", "content": "", "bbox_2d": [100,380,500,580], "index": 7},
            {"label": "paragraph_title", "content": "3. Results", "bbox_2d": [100,600,900,620], "index": 8},
            {"label": "table", "content": "| Method | F1 Score |\n| --- | --- |\n| Naive | 65.2 |\n| Ours | 94.6 |", "bbox_2d": [100,630,900,730], "index": 9},
            {"label": "paragraph", "content": "Table 1: Quantitative results.", "bbox_2d": [100,740,900,760], "index": 10}
        ]]
        class _MockResult:
            pass
        res = _MockResult()
        res.json_result = json_result
        res.markdown_result = "mock markdown"
        return res


def structure_aware_pipeline(pdf_path: Path, engine: str):
    """Extracts structure using PPDocLayout/GLM-OCR."""
    if engine == "mock":
        if "sample_document" not in str(pdf_path):
            logger.warning("Mock engine is hardcoded to sample_document.pdf!")
        logger.info("Running Structure-Aware Parsing (mock mode)...")
        parser = MockGlmOcr()
    else:
        logger.info("Running Structure-Aware Parsing using PP-DocLayout and GLM...")
        parser = GlmOcr(config_path=str(Path(__file__).parent / "config.yaml"), api_key=None)

    raw_result = parser.parse(str(pdf_path), save_layout_visualization=False)
    pages = []
    raw_pages = getattr(raw_result, "json_result", [])
    
    for page_idx, raw_elements in enumerate(raw_pages):
        page_num = page_idx + 1
        elements = []
        for el in raw_elements:
            bbox_2d = el.get("bbox_2d", [0,0,1,1])
            elements.append(ParsedElement(
                label=el.get("label", "paragraph"),
                text=el.get("content", ""),
                bbox=[float(v) for v in bbox_2d],
                score=1.0,
                reading_order=el.get("index", len(elements))
            ))
        pages.append(PageResult(page_num=page_num, elements=elements))
        
    result = ParseResult(source_file=pdf_path.name, pages=pages)
    structured_chunks = []
    for page in result.pages:
        page_chunks = structure_aware_chunking(
            elements=page.elements,
            source_file=result.source_file,
            page=page.page_num
        )
        structured_chunks.extend(page_chunks)
        
    return [asdict(c) for c in structured_chunks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--engine", type=str, choices=["mock", "real"], default="real",
                        help="Use 'real' for PP-DocLayout detection (default: real)")
    args = parser.parse_args()
    
    if not args.pdf.exists():
        alt = INPUT_DIR / args.pdf.name
        if alt.exists():
            args.pdf = alt
            logger.info(f"Using input directory: {args.pdf}")
        else:
            logger.error(f"PDF file not found: {args.pdf}")
            return
    
    args.output_dir.mkdir(exist_ok=True)
    
    naive_out = naive_baseline(args.pdf)
    with open(args.output_dir / "naive_chunks.json", "w") as f:
        json.dump(naive_out, f, indent=2)
        
    structured_out = structure_aware_pipeline(args.pdf, args.engine)
    with open(args.output_dir / "structured_chunks.json", "w") as f:
        json.dump(structured_out, f, indent=2)
        
    logger.info("Saved chunks to %s/", args.output_dir)
    print("\n--- COMPARISON SUMMARY ---")
    print(f"Naive baseline produced {len(naive_out)} text-only chunks.")
    
    modalities = [c["modality"] for c in structured_out]
    images = modalities.count("image")
    tables = modalities.count("table")
    print(f"Structure-aware produced {len(structured_out)} chunks ({images} image(s), {tables} table(s)).")


if __name__ == "__main__":
    main()