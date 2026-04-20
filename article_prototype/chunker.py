from typing import List
from schemas import ParsedElement, Chunk

ATOMIC_LABELS = {"table", "formula", "image", "figure"}
TITLE_LABELS = {"document_title", "paragraph_title", "figure_title"}

def infer_modality(labels: List[str]) -> str:
    types = set(labels)
    if types & {"image", "figure"}: return "image"
    if types & {"table"}: return "table"
    if types & {"formula", "inline_formula"}: return "formula"
    return "text"

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

def structure_aware_chunking(elements: List[ParsedElement], source_file: str, page: int, max_tokens: int = 512) -> List[Chunk]:
    chunks = []
    chunk_idx = 0
    current_texts = []
    current_labels = []
    current_tokens = 0
    pending_title = None
    pending_title_label = None

    def flush():
        nonlocal current_texts, current_labels, current_tokens, chunk_idx, pending_title, pending_title_label
        if not current_texts and not pending_title:
            return
        
        texts = []
        labels = []
        if pending_title:
            texts.append(pending_title)
            labels.append(pending_title_label)
            pending_title = None
            pending_title_label = None
        
        texts.extend(current_texts)
        labels.extend(current_labels)
        
        if not texts:
            return

        chunks.append(Chunk(
            text="\n\n".join(texts),
            chunk_id=f"{source_file}_{page}_{chunk_idx}",
            page=page,
            element_types=labels,
            bbox=None,  # typically None for multi-element text chunks
            source_file=source_file,
            is_atomic=False,
            modality=infer_modality(labels)
        ))
        chunk_idx += 1
        current_texts.clear()
        current_labels.clear()
        current_tokens = 0

    elements.sort(key=lambda x: x.reading_order)

    for el in elements:
        text = el.text.strip()
        label = el.label
        
        if label in ATOMIC_LABELS:
            flush()
            caption_text = None
            if pending_title and pending_title_label == "figure_title":
                caption_text = pending_title
                pending_title = None
                pending_title_label = None
            
            atomic_text = f"{caption_text}\n\n{text}" if caption_text and text else (text or caption_text or f"[{label}]")
            labels = ["figure_title", label] if caption_text else [label]
            
            chunks.append(Chunk(
                text=atomic_text,
                chunk_id=f"{source_file}_{page}_{chunk_idx}",
                page=page,
                element_types=labels,
                bbox=el.bbox,
                source_file=source_file,
                is_atomic=True,
                modality=infer_modality(labels)
            ))
            chunk_idx += 1
            continue

        if not text:
            continue

        if label in TITLE_LABELS:
            if current_texts or (pending_title and not current_texts):
                flush()
            pending_title = text
            pending_title_label = label
            continue

        tokens = estimate_tokens(text)
        pending_tokens = estimate_tokens(pending_title) if pending_title else 0

        if current_texts and (current_tokens + tokens + pending_tokens > max_tokens):
            flush()

        if pending_title:
            current_texts.append(pending_title)
            current_labels.append(pending_title_label or "paragraph_title")
            current_tokens += estimate_tokens(pending_title)
            pending_title = None
            pending_title_label = None

        current_texts.append(text)
        current_labels.append(label)
        current_tokens += tokens

        if current_tokens >= max_tokens:
            flush()

    flush()
    return chunks
