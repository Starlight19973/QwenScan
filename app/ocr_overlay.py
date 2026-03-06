"""Searchable PDF: OCR через Qwen3-VL + невидимый текстовый слой (PyMuPDF)."""

import json
import logging
import time
from typing import Callable

import fitz  # PyMuPDF

from app.vlm_client import call_vlm, call_vlm_chat, parse_vlm_json
from app.prompts import (
    OCR_FULLTEXT_SYSTEM, OCR_FULLTEXT_USER,
    TAGS_EXTRACT_SYSTEM, TAGS_EXTRACT_USER,
)
from app.ocr_hint_registry import get_ocr_hint
from app.config import OCR_DPI, OCR_MAX_TOKENS, TAGS_MAX_TOKENS, VLLM_CHAT_URL, MODEL_NAME, TEMPERATURE

log = logging.getLogger(__name__)

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _parse_ocr_response(raw: dict) -> list[dict]:
    """Parse VLM response into list of {text, bbox, type}.

    Handles: normal JSON, truncated JSON (partial lines array),
    and complete parse failures (raw_text fallback).
    """
    if raw.get("parse_error") and "raw_text" in raw:
        text = raw["raw_text"].strip()
        if not text:
            return []
        # Try to salvage truncated JSON: extract complete line objects
        lines = _salvage_truncated_json(text)
        if lines:
            return lines
        # Fallback: split raw text into lines (strip JSON artifacts)
        clean = _strip_json_wrapper(text)
        if clean:
            result = []
            for i, ln in enumerate(clean.split("\n")):
                ln = ln.strip()
                if ln:
                    result.append({"text": ln, "bbox": [20, 20 + i * 30, 980, 50 + i * 30], "type": "text"})
            return result
        return [{"text": text[:500], "bbox": [20, 20, 980, 980], "type": "text"}]

    lines = raw.get("lines")
    if not isinstance(lines, list):
        if isinstance(raw, dict) and "text" in raw:
            return [{"text": str(raw["text"]), "bbox": [20, 20, 980, 980], "type": "text"}]
        return []

    return _normalize_lines(lines)


def _normalize_lines(lines: list) -> list[dict]:
    """Normalize a list of line dicts."""
    result = []
    for item in lines:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            bbox = [20, 20, 980, 980]
        else:
            try:
                bbox = [float(b) for b in bbox[:4]]
            except (TypeError, ValueError):
                bbox = [20, 20, 980, 980]
        line_type = item.get("type", "text")
        if line_type not in ("text", "table_row"):
            line_type = "text"
        result.append({"text": text, "bbox": bbox, "type": line_type})
    return result


def _salvage_truncated_json(text: str) -> list[dict] | None:
    """Try to extract complete line objects from truncated JSON response."""
    import re as _re
    # Find all complete {"text": ..., "bbox": ..., "type": ...} objects
    pattern = r'\{\s*"text"\s*:\s*"([^"]*)"\s*,\s*"bbox"\s*:\s*\[(\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*)\]\s*,\s*"type"\s*:\s*"(text|table_row)"\s*\}'
    matches = _re.findall(pattern, text)
    if not matches or len(matches) < 2:
        return None
    result = []
    for txt, bbox_str, line_type in matches:
        try:
            bbox = [float(x.strip()) for x in bbox_str.split(",")]
        except ValueError:
            bbox = [20, 20, 980, 980]
        result.append({"text": txt, "bbox": bbox, "type": line_type})
    return result


def _strip_json_wrapper(text: str) -> str:
    """Strip JSON wrapper from raw text, leaving just the content."""
    import re as _re
    # Remove {"lines": [ ... wrapper
    text = _re.sub(r'^\s*\{\s*"lines"\s*:\s*\[', '', text)
    # Remove trailing ]} or incomplete
    text = _re.sub(r'\]\s*\}\s*$', '', text)
    # Remove individual line JSON objects, keep just text values
    # Extract text values from "text": "..." patterns
    texts = _re.findall(r'"text"\s*:\s*"([^"]*)"', text)
    if texts:
        return "\n".join(texts)
    return ""


async def ocr_page(image_bytes: bytes, max_tokens: int = OCR_MAX_TOKENS, mode: str = "universal") -> list[dict]:
    """Run OCR on a single page image via Qwen3-VL. Returns [{text, bbox}, ...]."""
    hint = get_ocr_hint(mode)
    system = OCR_FULLTEXT_SYSTEM + hint

    raw = await call_vlm(
        image_bytes,
        system,
        OCR_FULLTEXT_USER,
        max_tokens=max_tokens,
    )
    return _parse_ocr_response(raw)


async def extract_tags(full_text: str) -> dict:
    """Извлечь теги из OCR текста (текстовый запрос, без картинки)."""
    import httpx
    from app.vlm_client import _get_client

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": TAGS_EXTRACT_SYSTEM},
            {"role": "user", "content": TAGS_EXTRACT_USER + "\n\n" + full_text[:8000]},
        ],
        "max_tokens": TAGS_MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    client = _get_client()
    response = await client.post(VLLM_CHAT_URL, json=payload)
    result = response.json()

    if "error" in result:
        log.warning(f"Tags extraction error: {result['error']}")
        return {"names": [], "dates": [], "documents": []}

    if "choices" not in result:
        log.warning(f"Tags extraction unexpected response: {str(result)[:200]}")
        return {"names": [], "dates": [], "documents": []}

    content = result["choices"][0]["message"]["content"]
    parsed = parse_vlm_json(content)

    # Handle both {tags: {names, dates, documents}} and {names, dates, documents}
    if "tags" in parsed and isinstance(parsed["tags"], dict):
        tags = parsed["tags"]
    else:
        tags = parsed

    return {
        "names": tags.get("names", []) if isinstance(tags.get("names"), list) else [],
        "dates": tags.get("dates", []) if isinstance(tags.get("dates"), list) else [],
        "documents": tags.get("documents", []) if isinstance(tags.get("documents"), list) else [],
    }


def build_searchable_pdf(
    src_pdf_bytes: bytes,
    ocr_results: list[list[dict]],
) -> bytes:
    """Build searchable PDF: copy original pages + add invisible text overlay.

    Key difference from naive approach: we do NOT re-render pages.
    We copy original page content (preserving JPEG compression) and only
    add an invisible text layer on top. Result is ~same size as original.
    """
    src_doc = fitz.open(stream=src_pdf_bytes, filetype="pdf")
    out_doc = fitz.open()

    for page_idx in range(len(src_doc)):
        src_page = src_doc[page_idx]
        page_width_pts = src_page.rect.width
        page_height_pts = src_page.rect.height

        # Copy original page as-is (preserves original JPEG/image compression)
        out_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
        new_page = out_doc[out_doc.page_count - 1]

        # Add invisible text overlay
        if page_idx < len(ocr_results):
            lines = ocr_results[page_idx]
            for line in lines:
                text = line.get("text", "")
                bbox = line.get("bbox", [0, 0, 1000, 1000])
                if not text:
                    continue

                try:
                    pt_x1 = (bbox[0] / 1000.0) * page_width_pts
                    pt_y1 = (bbox[1] / 1000.0) * page_height_pts
                    pt_x2 = (bbox[2] / 1000.0) * page_width_pts
                    pt_y2 = (bbox[3] / 1000.0) * page_height_pts

                    line_height = pt_y2 - pt_y1
                    font_size = max(4.0, line_height * 0.80)

                    # Baseline position: near bottom of bbox
                    baseline_y = pt_y2 - line_height * 0.15

                    new_page.insert_text(
                        fitz.Point(pt_x1, baseline_y),
                        text,
                        fontsize=font_size,
                        fontfile=_FONT_PATH,
                        fontname="f0",
                        render_mode=3,  # INVISIBLE
                    )
                except Exception as e:
                    log.warning(f"insert_text failed page={page_idx}: {e}")
                    continue

    # Garbage-collect unused objects and compress
    result = out_doc.tobytes(garbage=4, deflate=True)
    out_doc.close()
    src_doc.close()
    return result


async def process_pdf_to_searchable(
    pdf_bytes: bytes,
    dpi: int = OCR_DPI,
    on_progress: Callable[[int, int], None] | None = None,
    mode: str = "universal",
) -> bytes:
    """Full pipeline: PDF -> OCR each page -> searchable PDF.

    Args:
        pdf_bytes: Input PDF bytes
        dpi: DPI for rendering pages (used only for OCR input, not for output)
        on_progress: Optional callback(page_num, total_pages)
        mode: OCR mode (universal, text, table, handwritten)

    Returns:
        Searchable PDF bytes
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    ocr_results: list[list[dict]] = []

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    for page_idx in range(total_pages):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        t0 = time.perf_counter()
        lines = await ocr_page(img_bytes, mode=mode)
        elapsed = time.perf_counter() - t0
        print(f"[OCR] page {page_idx + 1}/{total_pages}: {len(lines)} lines, {elapsed:.1f}s", flush=True)

        ocr_results.append(lines)

        if on_progress:
            on_progress(page_idx + 1, total_pages)

    doc.close()

    pdf_result = build_searchable_pdf(pdf_bytes, ocr_results)
    return pdf_result, ocr_results
