"""Синхронный API: POST /api/attributes — извлечение атрибутов из PDF."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from starlette.requests import Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field


class AttributeIn(BaseModel):
    id: str = Field(..., description="UUID атрибута")
    value: str = Field(..., description="Название поля для извлечения")

class AttributeOut(BaseModel):
    id: str = Field(..., description="UUID атрибута")
    value: str | None = Field(None, description="Извлечённое значение")

class AttributesResponse(BaseModel):
    attributes: list[AttributeOut] = Field(..., description="Массив атрибутов с извлечёнными значениями")
    pages: int = Field(..., description="Количество страниц в PDF")
    warnings: list[str] = Field(default_factory=list, description="Предупреждения при обработке")

from app.pdf_utils import pdf_to_images
from app.pipeline import process_page, process_document
from app.batch_processor import merge_file_pages
from app.pdf_utils import get_document_dpi
from app.template_registry import get_template
from app.config import EXTRACTION_MAX_TOKENS, MAX_DOCUMENT_PAGES

sync_router = APIRouter(prefix="/api", tags=["Синхронный API"])

# Промпт по умолчанию для произвольного извлечения атрибутов
_DEFAULT_SYSTEM_PROMPT = (
    "Ты — экспертная система извлечения данных из документов.\n"
    "На изображении — отсканированный документ. Извлеки запрошенные поля.\n\n"
    "ПРАВИЛА:\n"
    "1. Верни СТРОГО валидный JSON — объект с запрошенными ключами.\n"
    "2. Если данные не найдены — значение null. НИКОГДА не выдумывай.\n"
    "3. Даты в формате ДД.ММ.ГГГГ.\n"
    "4. Суммы — числа, разделитель копеек — точка.\n"
    "5. Если на странице нет полезных данных → {\"_страница_без_данных\": true}.\n"
    "/no_think"
)


def _build_attribute_prompt(attributes: list[dict]) -> tuple[str, dict[str, str]]:
    """Построить user_prompt из списка атрибутов и вернуть маппинг value→id.

    Args:
        attributes: [{"id": "uuid1", "value": "Номер документа"}, ...]

    Returns:
        (user_prompt, value_to_id) — промпт и обратный маппинг.
    """
    value_to_id: dict[str, str] = {}
    fields: dict[str, str] = {}

    for attr in attributes:
        attr_id = attr["id"]
        attr_value = attr["value"]
        # Нормализуем ключ: нижний регистр, пробелы → подчёркивания
        key = attr_value.lower().replace(" ", "_")
        fields[key] = attr_value
        value_to_id[key] = attr_id

    fields_json = json.dumps(fields, ensure_ascii=False, indent=2)
    user_prompt = f"Извлеки следующие поля из документа:\n{fields_json}"

    return user_prompt, value_to_id


def _map_results(vlm_result: dict, attributes: list[dict]) -> list[dict]:
    """Маппинг результата VLM обратно на UUID атрибутов.

    VLM возвращает dict с ключами — нормализованными названиями полей.
    Нужно сопоставить их с исходными id атрибутов.

    Returns:
        [{"id": "uuid1", "value": "47893"}, ...]
    """
    # Строим маппинг: normalized_key → id
    key_to_id: dict[str, str] = {}
    for attr in attributes:
        key = attr["value"].lower().replace(" ", "_")
        key_to_id[key] = attr["id"]

    result = []
    matched_ids: set[str] = set()

    for key, value in vlm_result.items():
        if key.startswith("_") or key in ("filename", "page", "parse_error", "raw_text", "error"):
            continue
        attr_id = key_to_id.get(key)
        if attr_id:
            str_value = str(value) if value is not None else None
            result.append({"id": attr_id, "value": str_value})
            matched_ids.add(attr_id)

    # Добавить атрибуты без совпадения как null
    for attr in attributes:
        if attr["id"] not in matched_ids:
            result.append({"id": attr["id"], "value": None})

    return result


async def _process_pdf_sync(
    pdf_bytes: bytes,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    postprocessing_steps: list[str],
    template_id: str = "",
) -> tuple[dict, int, list[str]]:
    """Обработать PDF целиком за один вызов VLM (multi-image).

    Returns:
        (result, num_pages, warnings)
    """
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    raw_pages = len(doc)
    doc.close()

    num_pages = min(raw_pages, MAX_DOCUMENT_PAGES)
    dpi = get_document_dpi(num_pages)
    images = pdf_to_images(pdf_bytes, dpi=dpi)
    warnings: list[str] = []

    if not images:
        raise HTTPException(400, "PDF не содержит страниц")

    if len(images) > MAX_DOCUMENT_PAGES:
        warnings.append(f"PDF содержит {len(images)} страниц, обработаны первые {MAX_DOCUMENT_PAGES}")
        images = images[:MAX_DOCUMENT_PAGES]

    try:
        result = await process_document(
            images=images,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            postprocessing_steps=postprocessing_steps,
            template_id=template_id,
        )
    except Exception as e:
        warnings.append(f"Ошибка обработки: {e}")
        return {}, len(images), warnings

    result["filename"] = "upload"
    result["page"] = 0

    return result, len(images), warnings


@sync_router.post(
    "/attributes",
    response_model=AttributesResponse,
    summary="Извлечение атрибутов из PDF",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file", "attributes"],
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "PDF файл"
                            },
                            "attributes": {
                                "type": "string",
                                "description": 'JSON массив: [{"id": "uuid", "value": "Номер документа"}, ...]'
                            },
                            "template": {
                                "type": "string",
                                "default": "",
                                "description": "Пусто = универсальный | ID шаблона (schet_faktura) | JSON с system_prompt"
                            }
                        }
                    }
                }
            }
        }
    },
)
async def extract_attributes(request: Request):
    """Синхронное извлечение атрибутов из PDF.

    Args:
        file: PDF файл (multipart form field).
        attributes: JSON string — [{"id": "uuid1", "value": "Номер документа"}, ...]
        template: "" (универсальный) | "schet_faktura" (ID шаблона)
                  | '{"system_prompt": "...", ...}' (полный JSON)
    """
    # Парсим форму без лимита на размер файла
    async with request.form(max_part_size=1024 * 1024 * 1024) as form:
        file = form["file"]
        attributes_raw = form.get("attributes", "")
        template = form.get("template", "")

        # Валидация файла
        if not hasattr(file, "filename") or not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Файл должен быть PDF")

        # Читаем PDF
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(400, "Пустой файл")

    # Парсим attributes
    try:
        attrs = json.loads(attributes_raw)
    except json.JSONDecodeError:
        raise HTTPException(400, "Невалидный JSON в attributes")

    if not isinstance(attrs, list) or not attrs:
        raise HTTPException(400, "attributes должен быть непустым массивом")

    for attr in attrs:
        if not isinstance(attr, dict) or "id" not in attr or "value" not in attr:
            raise HTTPException(400, 'Каждый атрибут должен содержать "id" и "value"')

    # Определяем промпты из template (3 варианта)
    system_prompt = ""
    max_tokens = EXTRACTION_MAX_TOKENS
    postprocessing_steps: list[str] = []
    template_id = ""

    if not template:
        system_prompt = _DEFAULT_SYSTEM_PROMPT
        user_prompt, _ = _build_attribute_prompt(attrs)
    elif template.strip().startswith("{"):
        try:
            tmpl_data = json.loads(template)
        except json.JSONDecodeError:
            raise HTTPException(400, "Невалидный JSON в template")
        system_prompt = tmpl_data.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        user_prompt = tmpl_data.get("user_prompt", "")
        max_tokens = tmpl_data.get("max_tokens", EXTRACTION_MAX_TOKENS)
        postprocessing_steps = tmpl_data.get("postprocessing", [])
        if not user_prompt:
            user_prompt, _ = _build_attribute_prompt(attrs)
    else:
        tmpl = get_template(template)
        if not tmpl:
            raise HTTPException(400, f"Шаблон '{template}' не найден")
        template_id = template
        system_prompt = tmpl["system_prompt"]
        user_prompt = tmpl["user_prompt"]
        max_tokens = tmpl.get("max_tokens", EXTRACTION_MAX_TOKENS)
        postprocessing_steps = tmpl.get("postprocessing", [])

    # Обрабатываем PDF
    vlm_result, num_pages, warnings = await _process_pdf_sync(
        pdf_bytes=pdf_bytes,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        postprocessing_steps=postprocessing_steps,
        template_id=template_id,
    )

    # Маппим результат обратно на UUID атрибутов
    mapped = _map_results(vlm_result, attrs)

    return JSONResponse({
        "attributes": mapped,
        "pages": num_pages,
        "warnings": warnings,
    })
