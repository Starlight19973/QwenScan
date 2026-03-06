"""Утилиты для работы с PDF: конвертация страниц в PNG."""

import math
import fitz  # PyMuPDF

from pathlib import Path
from app.config import PDF_DPI, MAX_MODEL_LEN, EXTRACTION_MAX_TOKENS


# Qwen3-VL: tokens ≈ (W × H) / 784; для A4 ≈ 0.1233 × DPI²
_TOKENS_PER_DPI2 = 0.1233
_MIN_DPI = 100
_MAX_DPI = 260
# Резерв под системный/user промпты и ответ
_PROMPT_OVERHEAD = 5000


def get_document_dpi(num_pages: int) -> int:
    """Вычислить оптимальный DPI чтобы все страницы поместились в контекст.

    Формула: token_budget = (MAX_MODEL_LEN - EXTRACTION_MAX_TOKENS - overhead) / num_pages
              DPI = sqrt(token_budget / 0.1233), clamp [100, 260]
    """
    if num_pages <= 0:
        return PDF_DPI

    total_budget = MAX_MODEL_LEN - EXTRACTION_MAX_TOKENS - _PROMPT_OVERHEAD
    per_page_budget = total_budget / num_pages
    dpi = math.sqrt(per_page_budget / _TOKENS_PER_DPI2)
    return max(_MIN_DPI, min(_MAX_DPI, int(dpi)))


def pdf_to_images(source: Path | bytes, dpi: int = PDF_DPI) -> list[bytes]:
    """Конвертировать PDF в список PNG-картинок (bytes на страницу).

    Args:
        source: путь к PDF-файлу или сырые bytes PDF.
        dpi: разрешение рендеринга.

    Returns:
        Список bytes (PNG) — по одному на страницу.
    """
    if isinstance(source, bytes):
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        doc = fitz.open(source)

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    images: list[bytes] = []

    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(pix.tobytes("png"))

    doc.close()
    return images
