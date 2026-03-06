"""Реестр шаблонов документов: загрузка, кеш, выбор промптов."""

from __future__ import annotations

import json
import re
import shutil
from threading import Lock
from pathlib import Path

from app.prompts import UNIFIED_SYSTEM, UNIFIED_USER, FLEXIBLE_SYSTEM
from app.config import EXTRACTION_MAX_TOKENS

_TEMPLATES_PATH = Path(__file__).parent / "templates.json"
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_cache: dict | None = None
_cache_lock = Lock()

_DEFAULT_FIELDS: list[dict] = [
    {"key": "тип_документа", "label": "Тип документа", "type": "string"},
    {"key": "номер_документа", "label": "Номер документа", "type": "string"},
    {"key": "дата_документа", "label": "Дата документа", "type": "date"},
    {"key": "контрагент1", "label": "Контрагент1", "type": "string"},
    {"key": "контрагент1_инн", "label": "ИНН", "type": "inn"},
    {"key": "контрагент1_кпп", "label": "КПП", "type": "kpp"},
    {"key": "контрагент1_адрес", "label": "Адрес", "type": "string"},
    {"key": "контрагент2", "label": "Контрагент2", "type": "string"},
    {"key": "контрагент2_инн", "label": "ИНН", "type": "inn"},
    {"key": "контрагент2_кпп", "label": "КПП", "type": "kpp"},
    {"key": "контрагент2_адрес", "label": "Адрес", "type": "string"},
    {"key": "ндс", "label": "НДС", "type": "number"},
    {"key": "сумма_с_ндс", "label": "Сумма с НДС", "type": "number"},
]


def _extract_fields_from_prompt(system_prompt: str) -> list[dict] | None:
    """Parse JSON schema from system_prompt and return fields list."""
    text = (system_prompt or "").strip()
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    snippet = text[start:end + 1]
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            obj = json.loads(fixed)
        except Exception:
            return None
    if not isinstance(obj, dict) or not obj:
        return None
    fields: list[dict] = []
    for key, type_hint in obj.items():
        key_s = key.strip()
        if not key_s or key_s.startswith("_"):
            continue
        hint_lower = str(type_hint).lower()
        key_lower = key_s.lower()
        if "инн" in key_lower or "inn" in key_lower:
            field_type = "inn"
        elif "кпп" in key_lower or "kpp" in key_lower:
            field_type = "kpp"
        elif "дд.мм" in hint_lower or "date" in hint_lower:
            field_type = "date"
        elif hint_lower in ("number", "число", "float", "int"):
            field_type = "number"
        else:
            field_type = "string"
        label = key_s.replace("_", " ").strip()
        label = label[0].upper() + label[1:] if label else key_s
        fields.append({"key": key_s, "label": label, "type": field_type})
    return fields if fields else None


def _generate_user_prompt_from_fields(fields: list[dict]) -> str:
    """Generate user_prompt matching the given fields."""
    names = [f["key"] for f in fields if not f["key"].startswith("_")]
    return "Извлеки из документа следующие поля: " + ", ".join(names) + "."



def _load_prompt_file(filepath: Path) -> str:
    """Read a prompt file, return empty string if missing."""
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


def _write_prompt_file(filepath: Path, content: str) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content.rstrip() + "\n", encoding="utf-8")


def _load_templates_list() -> list[dict]:
    with open(_TEMPLATES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("templates.json must contain a list")
    return data


def _save_templates_list(templates_list: list[dict]) -> None:
    tmp_path = _TEMPLATES_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(templates_list, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(_TEMPLATES_PATH)


def invalidate_cache() -> None:
    global _cache
    with _cache_lock:
        _cache = None
    # Also clear batch_processor schema cache
    try:
        from app.batch_processor import clear_schema_cache
        clear_schema_cache()
    except ImportError:
        pass


def _load() -> dict[str, dict]:
    global _cache
    with _cache_lock:
        if _cache is None:
            templates_list = _load_templates_list()
            _cache = {}
            for t in templates_list:
                tid = t["id"]
                # Load prompts from files if not inline
                if "system_prompt" not in t:
                    t["system_prompt"] = _load_prompt_file(_PROMPTS_DIR / tid / "system_prompt.md")
                if "user_prompt" not in t:
                    t["user_prompt"] = _load_prompt_file(_PROMPTS_DIR / tid / "user_prompt.md")
                _cache[tid] = t
    return _cache


_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def _slugify_template_id(name: str) -> str:
    text = (name or "").strip().lower()
    translit = "".join(_CYR_TO_LAT.get(ch, ch) for ch in text)
    translit = translit.replace("-", "_").replace(" ", "_")
    translit = re.sub(r"[^a-z0-9_]+", "_", translit)
    translit = re.sub(r"_+", "_", translit).strip("_")
    if translit and translit[0].isdigit():
        translit = f"t_{translit}"
    return translit or "template"


def _unique_template_id(base: str, existing_ids: set[str]) -> str:
    if base not in existing_ids:
        return base
    idx = 2
    while f"{base}_{idx}" in existing_ids:
        idx += 1
    return f"{base}_{idx}"


def _validate_template_id(template_id: str) -> str:
    tid = (template_id or "").strip()
    if not tid:
        raise ValueError("Template id is required")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", tid):
        raise ValueError("Invalid template id")
    return tid


def list_templates() -> list[dict]:
    """Список шаблонов (id, name, description, fields, page_range)."""
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "description": t["description"],
            # Поля — отдельная сущность от промпта. Для пользовательских шаблонов
            # по умолчанию используем фиксированный набор полей.
            "fields": t.get("fields") or _DEFAULT_FIELDS,
            "page_range": t.get("page_range", ""),
        }
        for t in _load().values()
    ]


def list_templates_with_system_prompts() -> list[dict]:
    """Список шаблонов для UI-редактора системных промптов."""
    items = []
    for t in _load().values():
        items.append(
            {
                "id": t["id"],
                "name": t["name"],
                "description": t.get("description", ""),
                "system_prompt": t.get("system_prompt", ""),
                "fields": t.get("fields") or _DEFAULT_FIELDS,
                "page_range": t.get("page_range", ""),
            }
        )
    return items


def get_template(template_id: str) -> dict | None:
    """Полный шаблон по id или None."""
    return _load().get(template_id)


def get_template_field_keys(template_id: str) -> list[str]:
    """Список допустимых ключей результата для данного шаблона.

    Важно: список атрибутов задаётся отдельно от промпта и используется для
    строгой фильтрации вывода (чтобы модель не \"придумывала\" поля).
    """
    tmpl = get_template(template_id) if template_id else None
    fields = (tmpl or {}).get("fields") or _DEFAULT_FIELDS
    keys: list[str] = []
    for f in fields:
        k = (f or {}).get("key")
        if isinstance(k, str) and k.strip():
            keys.append(k.strip())
    return keys or [f["key"] for f in _DEFAULT_FIELDS]


def update_template_system_prompt(
    template_id: str,
    system_prompt: str,
    name: str | None = None,
    description: str | None = None,
    fields: list[dict] | None = None,
    page_range: str | None = None,
) -> dict:
    """Обновить системный промпт и метаданные существующего шаблона."""
    template_id = _validate_template_id(template_id)
    system_prompt = (system_prompt or "").strip()
    if not system_prompt:
        raise ValueError("System prompt cannot be empty")

    templates_list = _load_templates_list()
    target = None
    for item in templates_list:
        if item.get("id") == template_id:
            target = item
            break
    if target is None:
        raise KeyError(f"Template '{template_id}' not found")

    if name is not None:
        name = name.strip()
        if not name:
            raise ValueError("Template name cannot be empty")
        target["name"] = name

    if description is not None:
        target["description"] = description.strip()

    if page_range is not None:
        target["page_range"] = page_range.strip()

    # Если fields переданы явно — используем их (пользователь задал вручную).
    # Иначе — автоэкстракция из system_prompt.
    if fields is not None:
        target["fields"] = fields
        user_prompt_gen = _generate_user_prompt_from_fields(fields)
        _write_prompt_file(_PROMPTS_DIR / template_id / "user_prompt.md", user_prompt_gen)
    else:
        parsed_fields = _extract_fields_from_prompt(system_prompt)
        if parsed_fields:
            target["fields"] = parsed_fields
            user_prompt_gen = _generate_user_prompt_from_fields(parsed_fields)
            _write_prompt_file(_PROMPTS_DIR / template_id / "user_prompt.md", user_prompt_gen)
    _save_templates_list(templates_list)
    _write_prompt_file(_PROMPTS_DIR / template_id / "system_prompt.md", system_prompt)
    invalidate_cache()

    updated = get_template(template_id)
    if not updated:
        raise RuntimeError(f"Template '{template_id}' not available after update")
    return updated


def create_template_with_system_prompt(
    name: str,
    system_prompt: str,
    description: str = "",
    fields: list[dict] | None = None,
    page_range: str = "",
) -> dict:
    """Создать новый тип документа и файл его системного промпта."""
    name = (name or "").strip()
    system_prompt = (system_prompt or "").strip()
    description = (description or "").strip()

    if not name:
        raise ValueError("Template name is required")
    if not system_prompt:
        raise ValueError("System prompt cannot be empty")

    templates_list = _load_templates_list()
    existing_ids = {str(t.get("id", "")) for t in templates_list}
    base_id = _slugify_template_id(name)
    template_id = _unique_template_id(base_id, existing_ids)

    # Если fields переданы явно — используем их, иначе автоэкстракция/дефолт.
    resolved_fields = fields if fields is not None else (
        _extract_fields_from_prompt(system_prompt) or _DEFAULT_FIELDS
    )

    new_template = {
        "id": template_id,
        "name": name,
        "description": description or f"Пользовательский шаблон: {name}",
        "fields": resolved_fields,
        "page_range": (page_range or "").strip(),
        "max_tokens": EXTRACTION_MAX_TOKENS,
        "postprocessing": [],
    }
    templates_list.append(new_template)

    user_prompt_gen = _generate_user_prompt_from_fields(resolved_fields)
    _write_prompt_file(_PROMPTS_DIR / template_id / "system_prompt.md", system_prompt)
    _write_prompt_file(_PROMPTS_DIR / template_id / "user_prompt.md", user_prompt_gen)
    _save_templates_list(templates_list)
    invalidate_cache()

    created = get_template(template_id)
    if not created:
        raise RuntimeError(f"Template '{template_id}' not available after create")
    return created


def delete_template_with_prompts(template_id: str) -> None:
    """Удалить шаблон из реестра и убрать его prompt-файлы."""
    template_id = _validate_template_id(template_id)

    templates_list = _load_templates_list()
    filtered = [t for t in templates_list if t.get("id") != template_id]
    if len(filtered) == len(templates_list):
        raise KeyError(f"Template '{template_id}' not found")

    _save_templates_list(filtered)
    prompt_dir = _PROMPTS_DIR / template_id
    if prompt_dir.exists():
        shutil.rmtree(prompt_dir)
    invalidate_cache()


def resolve_prompts(
    template_id: str = "",
    custom_prompt: str = "",
) -> tuple[str, str, int, list[str]]:
    """Определить промпты, max_tokens и шаги пост-обработки.

    | template_id | custom_prompt | Поведение                              |
    |-------------|---------------|----------------------------------------|
    | пусто       | пусто         | Универсальный (UNIFIED_SYSTEM/USER)    |
    | пусто       | есть          | Гибкий (FLEXIBLE_SYSTEM + custom)      |
    | есть        | пусто         | Промпты из шаблона                     |
    | есть        | есть          | system из шаблона + custom как user     |
    """
    if template_id:
        tmpl = get_template(template_id)
        if tmpl:
            system = tmpl["system_prompt"]
            user = custom_prompt if custom_prompt else tmpl["user_prompt"]
            max_tokens = tmpl.get("max_tokens", EXTRACTION_MAX_TOKENS)
            steps = tmpl.get("postprocessing", [])
            return system, user, max_tokens, steps

    if custom_prompt:
        return FLEXIBLE_SYSTEM, custom_prompt, EXTRACTION_MAX_TOKENS, []

    # Универсальный режим (обратная совместимость)
    return (
        UNIFIED_SYSTEM,
        UNIFIED_USER,
        EXTRACTION_MAX_TOKENS,
        [
            "split_inn_kpp", "validate_inn", "validate_kpp",
            "clean_address", "fix_doc_number",
        ],
    )


# ─── Continuation (pages 2+) optimization ───────────────────────────────

_CONTINUATION_MAX_TOKENS = 256  # Для стр. 2+ нужен минимальный ответ

# Шаблоны, для которых страницы 2+ содержат только продолжение таблицы и итоги
_CONTINUATION_TEMPLATES = {
    "upd", "schet_faktura", "akt", "torg12",
    "schet_na_oplatu", "schet_dogovor", "deklaratsiya", "doverennost",
}

_CONTINUATION_SYSTEM = """Ты — экспертная система извлечения данных из бухгалтерских документов.

На изображении — ПРОДОЛЖЕНИЕ (страница 2+) многостраничного документа.
Шапка документа (номер, дата, стороны, ИНН, КПП, адреса) уже извлечена с первой страницы.

Извлеки ТОЛЬКО итоговые суммы, если они есть на этой странице.

Верни СТРОГО JSON:
{
  "ндс": number | null,
  "сумма_с_ндс": number | null
}

ПРАВИЛА:
1. Верни ТОЛЬКО валидный JSON.
2. Суммы — числа, десятичный разделитель — точка, без пробелов.
3. Если итоговых сумм нет — null.
4. Если на странице ТОЛЬКО подписи/печати → {"_страница_без_данных": true}.
/no_think"""

_CONTINUATION_USER = "Это продолжение документа. Извлеки ТОЛЬКО итоговые суммы (НДС и сумму с НДС), если они есть на этой странице."


def resolve_continuation_prompts(
    template_id: str = "",
    custom_prompt: str = "",
) -> tuple[str, str, int, list[str]] | None:
    """Облегчённые промпты для страниц 2+ или None если оптимизация не применима."""
    if custom_prompt:
        return None  # Пользовательский промпт — не оптимизируем
    if not template_id or template_id not in _CONTINUATION_TEMPLATES:
        return None
    return _CONTINUATION_SYSTEM, _CONTINUATION_USER, _CONTINUATION_MAX_TOKENS, []
