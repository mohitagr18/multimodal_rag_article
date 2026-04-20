from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ParsedElement:
    label: str
    text: str
    bbox: List[float]
    score: float
    reading_order: int

@dataclass
class PageResult:
    page_num: int
    elements: List[ParsedElement] = field(default_factory=list)
    markdown: str = ""

@dataclass
class ParseResult:
    source_file: str
    pages: List[PageResult] = field(default_factory=list)
    total_elements: int = 0
    full_markdown: str = ""

@dataclass
class Chunk:
    text: str
    chunk_id: str
    page: int
    element_types: List[str]
    bbox: Optional[List[float]]
    source_file: str
    is_atomic: bool
    modality: str = "text"
    image_base64: Optional[str] = None
    caption: Optional[str] = None
