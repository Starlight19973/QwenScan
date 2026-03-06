"""Чат с VLM — свободный режим + документы, per-message изображения."""

import base64
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import CHAT_MAX_HISTORY, UPLOADS_DIR
from app.pdf_utils import pdf_to_images

# ── In-memory storage ──────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_page_cache: dict[tuple[str, int], bytes] = {}


CHAT_SYSTEM_PROMPT = (
    "Ты — эксперт по анализу документов. Перед тобой изображение страницы документа.\n\n"
    "Отвечай на вопросы пользователя о содержимом документа.\n"
    "Будь точным, цитируй данные из документа.\n"
    "Если информация не видна — честно скажи об этом.\n"
    "Не выдумывай данные. Отвечай на языке пользователя.\n"
    "/no_think"
)

CHAT_FREE_SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай точно и по делу.\n"
    "Если пользователь прикрепил изображение — внимательно анализируй его содержимое.\n"
    "Отвечай на языке пользователя.\n"
    "/no_think"
)


# ── Sessions ───────────────────────────────────────────────────────

def create_session(
    batch_id: str | None = None,
    filename: str | None = None,
    page: int | None = None,
    pdf_path: Path | None = None,
) -> str:
    """Create a new chat session, return chat_id (8 hex chars)."""
    chat_id = uuid.uuid4().hex[:8]
    mode = "document" if pdf_path else "free"
    _sessions[chat_id] = {
        "id": chat_id,
        "batch_id": batch_id,
        "filename": filename or "",
        "page": page or 0,
        "mode": mode,
        "messages": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdf_path": pdf_path,
    }
    return chat_id


def get_session(chat_id: str) -> dict | None:
    return _sessions.get(chat_id)


def add_message(
    chat_id: str,
    role: str,
    content: str,
    images: list[str] | None = None,
) -> dict:
    """Add a message to the session.

    Args:
        images: optional list of base64-encoded PNG strings (for user messages).
    """
    session = _sessions[chat_id]
    msg = {
        "role": role,
        "content": content,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if images:
        msg["images"] = images
    session["messages"].append(msg)
    return msg


def list_sessions() -> list[dict]:
    return [
        {
            "id": s["id"],
            "batch_id": s.get("batch_id"),
            "filename": s.get("filename", ""),
            "page": s.get("page", 0),
            "mode": s.get("mode", "document"),
            "messages_count": len(s["messages"]),
            "created_at": s["created_at"],
        }
        for s in _sessions.values()
    ]


# ── Page image cache ──────────────────────────────────────────────

def get_or_render_page(pdf_path: Path, page: int) -> bytes:
    """Return PNG bytes for the given page (0-based). Cached in memory."""
    key = (str(pdf_path), page)
    if key in _page_cache:
        return _page_cache[key]
    images = pdf_to_images(pdf_path)
    for i, img in enumerate(images):
        _page_cache[(str(pdf_path), i)] = img
    if page < 0 or page >= len(images):
        raise IndexError(f"Page {page} out of range (0-{len(images)-1})")
    return _page_cache[key]


def get_page_count(pdf_path: Path) -> int:
    """Return number of pages in a PDF."""
    import fitz
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# ── Resolve PDF path ──────────────────────────────────────────────

def resolve_pdf_path(batch_id: str | None, filename: str) -> Path | None:
    """Find the PDF file on disk."""
    if batch_id and batch_id != "direct":
        path = UPLOADS_DIR / f"{batch_id}_{filename}"
    else:
        path = UPLOADS_DIR / f"chat_{filename}"
    return path if path.exists() else None


# ── Build VLM messages ────────────────────────────────────────────

_MAX_IMAGE_MESSAGES = 3  # include images from at most last N user messages


def build_vlm_messages(
    session: dict,
    page_image_b64: str | None = None,
) -> list[dict]:
    """Build OpenAI-format messages for VLM.

    Each user message can carry its own images (stored in msg['images']).
    In document mode, the page image is attached to the first user message
    that has no explicit attachments (backward compat).

    To respect vLLM's --limit-mm-per-prompt, only include images from
    the last _MAX_IMAGE_MESSAGES user messages that have images.
    """
    is_document = session.get("mode") == "document"
    system_prompt = CHAT_SYSTEM_PROMPT if is_document else CHAT_FREE_SYSTEM_PROMPT

    messages = session["messages"]
    if len(messages) > CHAT_MAX_HISTORY:
        messages = messages[-CHAT_MAX_HISTORY:]

    # Find which user messages (by index) should keep their images
    img_msg_indices: set[int] = set()
    count = 0
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("images") and messages[idx]["role"] == "user":
            img_msg_indices.add(idx)
            count += 1
            if count >= _MAX_IMAGE_MESSAGES:
                break

    vlm_messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]

    page_image_used = False

    for idx, msg in enumerate(messages):
        role = msg["role"]
        text = msg["content"]
        msg_images = msg.get("images") if idx in img_msg_indices else None

        if role == "user" and msg_images:
            content_parts = []
            for img_b64 in msg_images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            content_parts.append({"type": "text", "text": text})
            vlm_messages.append({"role": "user", "content": content_parts})
        elif role == "user" and is_document and not page_image_used and page_image_b64:
            vlm_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{page_image_b64}"},
                    },
                    {"type": "text", "text": text},
                ],
            })
            page_image_used = True
        else:
            vlm_messages.append({"role": role, "content": text})

    return vlm_messages
