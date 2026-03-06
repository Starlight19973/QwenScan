"""Одноэтапная обработка: классификация + извлечение за один вызов VLM + пост-обработка."""

import re

from app.vlm_client import call_vlm, call_vlm_document
from app.prompts import UNIFIED_SYSTEM, UNIFIED_USER, FLEXIBLE_SYSTEM
from app.validators import validate_requisites, validate_flexible_requisites
from app.config import EXTRACTION_MAX_TOKENS
from app.postprocessors import run_postprocessing


def _split_inn_kpp(value: str) -> tuple[str | None, str | None]:
    """Разделить ИНН/КПП из объединённой строки."""
    if not value:
        return None, None
    if "/" in value:
        parts = value.split("/", 1)
        inn = re.sub(r"\D", "", parts[0].strip())
        kpp = re.sub(r"\D", "", parts[1].strip()) if len(parts) > 1 else ""
        inn = inn if len(inn) in (10, 12) else None
        kpp = kpp if len(kpp) == 9 else None
        return inn, kpp
    digits = re.sub(r"\D", "", value)
    if len(digits) == 19:
        return digits[:10], digits[10:]
    if len(digits) == 21:
        return digits[:12], digits[12:]
    return value, None


def _clean_address(addr: str) -> str | None:
    """Убрать банковские реквизиты и телефоны из адреса."""
    if not addr:
        return None
    cleaned = re.split(
        r'[,;]?\s*(?:р/с|р\.с\.|Р/С|расч[её]тный|р/сч|'
        r'тел\.?\s*[:.]?\s*\+?\d|тел\s*\(|Тел\.?\s*[:.]?\s*\+?\d|факс|Факс|'
        r'т\.\s*\d{3}|т\s*\d{3}|'
        r'\+7\s*\(|\+7\(|8[\s-]?\(?\d{3}\)?[\s-]?\d{3}|'
        r'к/с|корр[.]?\s*сч|К/С|БИК|бик|Бик|ИНН\s+банка|в\s+банке\s|'
        r'ПАО\s+С[бБ]ербанк|АКЦИОНЕРНОЕ\s+ОБЩЕСТВО.*БАНК|Банк\s+[А-Я]|'
        r'e-?mail|www\.|http)',
        addr,
        maxsplit=1,
    )[0]
    cleaned = cleaned.rstrip(" ,;.")
    return cleaned if cleaned else None


def _is_valid_kpp(value: str) -> bool:
    if not value:
        return False
    digits = re.sub(r"\D", "", value)
    if len(digits) != 9:
        return False
    region = int(digits[:2])
    return 1 <= region <= 99


def _fix_document_number(num: str) -> str:
    """Исправить буквы О на цифры 0 в номерах документов."""
    if not num or not isinstance(num, str):
        return num
    fixed = re.sub(r'(\d)\s+(\d)', r'\1\2', num)
    fixed = re.sub(r'^[ОоO](?=[A-Za-z]{1,3}[/\\])', '0', fixed)
    fixed = re.sub(r'(?<=\d)[ОоO](?=\d)', '0', fixed)
    return fixed


def _postprocess_act(data: dict) -> dict:
    """Пост-обработка для актов: очистить контрагентов."""
    buyer = data.get("покупатель")
    seller = data.get("продавец")
    if not buyer and seller:
        data["покупатель"] = seller
        data["покупатель_инн"] = data.get("продавец_инн")
        data["покупатель_кпп"] = data.get("продавец_кпп")
        data["покупатель_адрес"] = data.get("продавец_адрес")
    data["продавец"] = None
    data["продавец_инн"] = None
    data["продавец_кпп"] = None
    data["продавец_адрес"] = None
    return data


def postprocess_template(data: dict) -> dict:
    """Пост-обработка результата шаблонного режима (deprecated — используйте run_postprocessing)."""
    return run_postprocessing(
        data,
        [
            "fix_doc_number", "postprocess_act", "split_inn_kpp",
            "validate_inn", "validate_kpp", "clean_address",
            "check_nds_vs_summa",
        ],
    )


async def process_page(
    image_bytes: bytes,
    system_prompt: str = "",
    user_prompt: str = "",
    max_tokens: int = 0,
    postprocessing_steps: list[str] | None = None,
    template_id: str = "",
    custom_prompt: str = "",
) -> dict:
    """Полный цикл обработки одной страницы.

    Новая сигнатура: resolved промпты передаются вызывающим кодом.
    Обратная совместимость: если system_prompt пуст — legacy логика через custom_prompt.
    """
    if postprocessing_steps is None:
        postprocessing_steps = []

    # Legacy fallback
    if not system_prompt:
        if custom_prompt:
            system_prompt = FLEXIBLE_SYSTEM
            user_prompt = custom_prompt
            max_tokens = max_tokens or EXTRACTION_MAX_TOKENS
        else:
            system_prompt = UNIFIED_SYSTEM
            user_prompt = UNIFIED_USER
            max_tokens = max_tokens or EXTRACTION_MAX_TOKENS
            postprocessing_steps = [
                "fix_doc_number", "postprocess_act", "split_inn_kpp",
                "validate_inn", "validate_kpp", "clean_address",
                "check_nds_vs_summa", "validate_requisites",
            ]

    max_tokens = max_tokens or EXTRACTION_MAX_TOKENS

    extracted = await call_vlm(
        image_bytes=image_bytes,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )

    if extracted.get("_страница_без_данных"):
        return extracted

    # Пост-обработка
    if postprocessing_steps:
        extracted = run_postprocessing(extracted, postprocessing_steps)
    elif not template_id and not custom_prompt:
        warnings = validate_requisites(extracted)
        if warnings:
            if "проблемы" not in extracted:
                extracted["проблемы"] = []
            extracted["проблемы"].extend(warnings)
    elif not template_id and custom_prompt:
        warnings = validate_flexible_requisites(extracted)
        if warnings:
            extracted["_предупреждения"] = warnings

    return extracted


async def process_document(
    images: list[bytes],
    system_prompt: str = "",
    user_prompt: str = "",
    max_tokens: int = 0,
    postprocessing_steps: list[str] | None = None,
    template_id: str = "",
    custom_prompt: str = "",
) -> dict:
    """Полный цикл обработки ВСЕХ страниц документа за один вызов VLM.

    Все страницы отправляются в одном запросе — модель видит весь контекст.
    """
    if postprocessing_steps is None:
        postprocessing_steps = []

    if not system_prompt:
        if custom_prompt:
            system_prompt = FLEXIBLE_SYSTEM
            user_prompt = custom_prompt
            max_tokens = max_tokens or EXTRACTION_MAX_TOKENS
        else:
            system_prompt = UNIFIED_SYSTEM
            user_prompt = UNIFIED_USER
            max_tokens = max_tokens or EXTRACTION_MAX_TOKENS
            postprocessing_steps = [
                "fix_doc_number", "postprocess_act", "split_inn_kpp",
                "validate_inn", "validate_kpp", "clean_address",
                "check_nds_vs_summa", "validate_requisites",
            ]

    max_tokens = max_tokens or EXTRACTION_MAX_TOKENS

    extracted = await call_vlm_document(
        images=images,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )

    if extracted.get("_страница_без_данных"):
        return extracted

    if postprocessing_steps:
        extracted = run_postprocessing(extracted, postprocessing_steps)
    elif not template_id and not custom_prompt:
        warnings = validate_requisites(extracted)
        if warnings:
            if "проблемы" not in extracted:
                extracted["проблемы"] = []
            extracted["проблемы"].extend(warnings)
    elif not template_id and custom_prompt:
        warnings = validate_flexible_requisites(extracted)
        if warnings:
            extracted["_предупреждения"] = warnings

    return extracted
