#!/usr/bin/env python3
"""
Visualize PP-DocLayout element detections on the test PDF.

Renders each page of the PDF as a high-resolution image, then draws colored
bounding boxes around every detected element (text, tables, images, formulas,
titles, footnotes, etc.) with a labeled legend.

Usage:
    python visualize_layout.py                     # Defaults: input/test.pdf
    python visualize_layout.py --pdf path/to.pdf   # Custom PDF
    python visualize_layout.py --dpi 200           # Higher resolution
"""
import argparse
import json
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont
import yaml

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

with open(PROJECT_ROOT / "src" / "config.yaml") as f:
    config = yaml.safe_load(f)

INPUT_DIR = PROJECT_ROOT / config.get("directories", {}).get("input", "input")
RESULTS_DIR = PROJECT_ROOT / config.get("directories", {}).get("results", "results")

# ── Color palette: one vibrant color per element label ──────────────────────
LABEL_COLORS = {
    "text":              (41, 128, 185),   # Blue
    "paragraph_title":   (142, 68, 173),   # Purple
    "document_title":    (155, 89, 182),   # Amethyst
    "figure_title":      (243, 156, 18),   # Amber / Orange
    "image":             (46, 204, 113),   # Emerald green
    "table":             (231, 76, 60),    # Red
    "formula":           (26, 188, 156),   # Teal
    "footnote":          (149, 165, 166),  # Gray
    "reference":         (127, 140, 141),  # Dark gray
    "reference_content": (127, 140, 141),  # Dark gray
    "header":            (189, 195, 199),  # Silver
    "footer":            (189, 195, 199),  # Silver
    "abstract":          (52, 152, 219),   # Light blue
    "chart":             (241, 196, 15),   # Yellow
    "seal":              (211, 84, 0),     # Pumpkin
    "algorithm":         (22, 160, 133),   # Green sea
    "aside_text":        (44, 62, 80),     # Midnight blue
    "content":           (52, 73, 94),     # Wet asphalt
    "vision_footnote":   (108, 122, 137),  # Blue gray
    "formula_number":    (39, 174, 96),    # Nephritis
    "number":            (100, 100, 100),  # Neutral gray
    "inline_formula":    (22, 160, 133),   # Green sea
    "doc_title":         (155, 89, 182),   # Amethyst
}
FALLBACK_COLOR = (200, 200, 200)
BOX_WIDTH = 3
LABEL_FONT_SIZE = 14


def _get_color(label: str):
    return LABEL_COLORS.get(label, FALLBACK_COLOR)


def _try_load_font(size: int):
    """Try to load a readable font, falling back to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def run_layout_detection(pdf_path: str):
    """Run PP-DocLayout + GLM-OCR and return raw element dicts per page."""
    from glmocr import GlmOcr

    print("Running layout detection (PP-DocLayout + GLM-OCR)...")
    parser = GlmOcr(
        config_path=str(PROJECT_ROOT / "src" / "config.yaml"),
        api_key=None,
    )
    result = parser.parse(pdf_path, save_layout_visualization=False)
    raw_pages = getattr(result, "json_result", [])
    print(f"  Detected {sum(len(p) for p in raw_pages)} elements across {len(raw_pages)} pages.")
    return raw_pages


def render_page(doc: fitz.Document, page_idx: int, dpi: int = 150) -> Image.Image:
    """Render a single PDF page to a PIL Image."""
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def draw_boxes(img: Image.Image, elements: list, page_w: float, page_h: float):
    """
    Draw bounding boxes on the image for each element.
    
    Bounding boxes from the layout engine are in 0-1000 normalized coordinates.
    """
    draw = ImageDraw.Draw(img)
    font = _try_load_font(LABEL_FONT_SIZE)
    img_w, img_h = img.size

    for el in elements:
        label = el.get("label", "unknown")
        bbox = el.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue

        # Convert 0-1000 normalized coords → pixel coords
        x1 = bbox[0] * img_w / 1000.0
        y1 = bbox[1] * img_h / 1000.0
        x2 = bbox[2] * img_w / 1000.0
        y2 = bbox[3] * img_h / 1000.0

        color = _get_color(label)

        # Draw rectangle
        for offset in range(BOX_WIDTH):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Draw label tag
        tag_text = label.upper().replace("_", " ")
        text_bbox = font.getbbox(tag_text)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        pad = 4

        # Position tag at top-left of box
        tag_x = x1
        tag_y = y1 - th - 2 * pad
        if tag_y < 0:
            tag_y = y1 + 2  # If no room above, put inside

        # Semi-transparent background for label
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [tag_x, tag_y, tag_x + tw + 2 * pad, tag_y + th + 2 * pad],
            fill=(*color, 180),
        )
        img_rgba = img.convert("RGBA")
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        img.paste(img_rgba.convert("RGB"))

        # Re-create draw context after paste
        draw = ImageDraw.Draw(img)
        draw.text(
            (tag_x + pad, tag_y + pad),
            tag_text,
            fill=(255, 255, 255),
            font=font,
        )

    return img


def draw_legend(img: Image.Image, labels_used: set) -> Image.Image:
    """Add a color legend bar at the bottom of the image."""
    font = _try_load_font(LABEL_FONT_SIZE)
    sorted_labels = sorted(labels_used)

    # Calculate legend dimensions
    pad = 12
    swatch_size = 16
    line_height = swatch_size + 8
    col_width = 220
    cols = max(1, img.width // col_width)
    rows = (len(sorted_labels) + cols - 1) // cols
    legend_height = rows * line_height + 2 * pad + 30  # extra for title

    # Create new image with legend area
    new_img = Image.new("RGB", (img.width, img.height + legend_height), (255, 255, 255))
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    title_font = _try_load_font(LABEL_FONT_SIZE + 2)
    y_start = img.height + pad
    draw.text((pad, y_start), "ELEMENT LEGEND", fill=(50, 50, 50), font=title_font)
    y_start += 24

    for idx, label in enumerate(sorted_labels):
        col = idx % cols
        row = idx // cols
        x = pad + col * col_width
        y = y_start + row * line_height
        color = _get_color(label)
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=color)
        draw.text(
            (x + swatch_size + 6, y),
            label.replace("_", " ").title(),
            fill=(40, 40, 40),
            font=font,
        )

    return new_img


def main():
    parser = argparse.ArgumentParser(description="Visualize layout detection bounding boxes")
    parser.add_argument("--pdf", type=str, default="test.pdf", help="PDF file (default: test.pdf)")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Output directory")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to cached JSON of raw layout results (skip re-detection)")
    args = parser.parse_args()

    pdf_path = INPUT_DIR / args.pdf
    if not pdf_path.exists():
        pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Get raw layout elements ─────────────────────────────────────
    if args.cache and Path(args.cache).exists():
        print(f"Loading cached layout from {args.cache}")
        with open(args.cache) as f:
            raw_pages = json.load(f)
    else:
        raw_pages = run_layout_detection(str(pdf_path))
        # Cache for future runs
        cache_path = output_dir / "raw_layout_elements.json"
        with open(cache_path, "w") as f:
            json.dump(raw_pages, f, indent=2)
        print(f"  Cached raw elements to {cache_path}")

    # ── Step 2: Render and annotate each page ───────────────────────────────
    doc = fitz.open(pdf_path)
    all_labels = set()
    output_paths = []

    for page_idx, elements in enumerate(raw_pages):
        page_num = page_idx + 1
        print(f"\nPage {page_num}: {len(elements)} elements")

        for el in elements:
            lbl = el.get("label", "unknown")
            all_labels.add(lbl)
            bbox = el.get("bbox_2d", [])
            content_preview = el.get("content", "")[:50]
            print(f"  [{lbl:20s}] bbox={bbox}  \"{content_preview}...\"")

        # Render page
        page_img = render_page(doc, page_idx, dpi=args.dpi)

        # Get page dimensions (PDF points)
        page = doc[page_idx]
        page_w = page.rect.width
        page_h = page.rect.height

        # Draw bounding boxes
        annotated = draw_boxes(page_img, elements, page_w, page_h)

        # Add legend
        annotated = draw_legend(annotated, all_labels)

        # Save
        out_path = output_dir / f"layout_page_{page_num}.png"
        annotated.save(out_path, "PNG")
        output_paths.append(out_path)
        print(f"  → Saved: {out_path}")

    doc.close()

    print(f"\n{'='*50}")
    print(f"Generated {len(output_paths)} annotated page images:")
    for p in output_paths:
        print(f"  {p}")
    print(f"\nLabels found: {sorted(all_labels)}")


if __name__ == "__main__":
    main()
