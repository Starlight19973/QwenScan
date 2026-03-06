"""Registry for editable OCR mode hints (stored in ocr_hints.json)."""

import json
from pathlib import Path

_HINTS_PATH = Path(__file__).parent / "ocr_hints.json"


def load_ocr_hints() -> list[dict]:
    """Read all OCR hints from JSON file."""
    with open(_HINTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ocr_hint(mode: str) -> str:
    """Return the hint string for a given OCR mode id."""
    for h in load_ocr_hints():
        if h["id"] == mode:
            return h.get("hint", "")
    return ""


def update_ocr_hint(mode: str, name: str, description: str, hint: str) -> dict:
    """Update a single OCR hint entry and persist to file. Returns the updated entry."""
    hints = load_ocr_hints()
    for h in hints:
        if h["id"] == mode:
            h["name"] = name
            h["description"] = description
            h["hint"] = hint
            with open(_HINTS_PATH, "w", encoding="utf-8") as f:
                json.dump(hints, f, ensure_ascii=False, indent=2)
            return h
    raise KeyError(f"OCR mode '{mode}' not found")
