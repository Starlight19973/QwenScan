"""Реестр шагов пост-обработки данных, извлечённых VLM."""

import re

from app.validators import validate_requisites, validate_flexible_requisites


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


def _is_valid_kpp(value: str) -> bool:
    if not value:
        return False
    digits = re.sub(r"\D", "", value)
    if len(digits) != 9:
        return False
    region = int(digits[:2])
    return 1 <= region <= 99


def _clean_address(addr: str) -> str | None:
    """Убрать банковские реквизиты, телефоны, email из адреса."""
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


# --- Шаги пост-обработки ---


def _split_inn_kpp_step(data: dict) -> dict:
    """Разделить слитые ИНН/КПП для всех *_инн полей."""
    inn_keys = [k for k in data if k.endswith("_инн")]
    for inn_key in inn_keys:
        kpp_key = inn_key.replace("_инн", "_кпп")
        inn_val = data.get(inn_key)
        if isinstance(inn_val, str):
            inn_digits = re.sub(r"\D", "", inn_val)
            needs_split = (
                "/" in inn_val
                or len(inn_digits) == 19
                or len(inn_digits) == 21
                or len(inn_digits) > 12
            )
            if needs_split:
                new_inn, new_kpp = _split_inn_kpp(inn_val)
                if new_inn:
                    data[inn_key] = new_inn
                if new_kpp and not _is_valid_kpp(str(data.get(kpp_key, ""))):
                    data[kpp_key] = new_kpp
    return data


def _validate_inn_step(data: dict) -> dict:
    """Валидация формата ИНН: только цифры, 10 или 12 знаков."""
    inn_keys = [k for k in data if k.endswith("_инн")]
    for inn_key in inn_keys:
        inn_val = data.get(inn_key)
        if isinstance(inn_val, str):
            digits = re.sub(r"\D", "", inn_val)
            if digits and len(digits) in (10, 12):
                data[inn_key] = digits
            elif digits:
                data[inn_key] = None
    return data


def _validate_kpp_step(data: dict) -> dict:
    """Валидация КПП: 9 цифр, не почтовый индекс, у ИП нет КПП."""
    kpp_keys = [k for k in data if k.endswith("_кпп")]
    for kpp_key in kpp_keys:
        # У ИП не бывает КПП
        name_key = kpp_key.replace("_кпп", "")
        name_val = data.get(name_key, "")
        if isinstance(name_val, str) and re.match(
            r'(?i)^ИП\s|индивидуальный\s+предприниматель', name_val
        ):
            data[kpp_key] = None
            continue

        kpp_val = data.get(kpp_key)
        if kpp_val is not None:
            kpp_digits = re.sub(r"\D", "", str(kpp_val))
            if len(kpp_digits) == 6:
                data[kpp_key] = None
            elif len(kpp_digits) == 10:
                data[kpp_key] = None
            elif len(kpp_digits) == 11:
                data[kpp_key] = None
            elif len(kpp_digits) != 9:
                data[kpp_key] = None
    return data


def _clean_address_step(data: dict) -> dict:
    """Очистка адресных полей от банковских реквизитов и телефонов."""
    addr_keys = [k for k in data if k.endswith("_адрес")]
    for addr_key in addr_keys:
        addr_val = data.get(addr_key)
        if isinstance(addr_val, str):
            data[addr_key] = _clean_address(addr_val)
    return data


def _fix_doc_number_step(data: dict) -> dict:
    """Исправление номеров документов: пробелы, О→0."""
    for key in ("номер_документа", "номер", "номер_дт"):
        val = data.get(key)
        if isinstance(val, str):
            # Убрать пробелы между цифрами
            fixed = re.sub(r'(\d)\s+(\d)', r'\1\2', val)
            # Кириллическая О перед латинскими буквами → 0
            fixed = re.sub(r'^[ОоO](?=[A-Za-z]{1,3}[/\\])', '0', fixed)
            # цифра+О+цифра → цифра+0+цифра
            fixed = re.sub(r'(?<=\d)[ОоO](?=\d)', '0', fixed)
            data[key] = fixed
    return data


def _postprocess_act_step(data: dict) -> dict:
    """Очистка продавца для актов (в актах нет продавца)."""
    doc_type = str(data.get("тип_документа", "")).lower().strip()
    if doc_type != "акт":
        return data

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


def _check_nds_vs_summa_step(data: dict) -> dict:
    """НДС не может быть больше суммы с НДС — предупреждение."""
    nds = data.get("ндс")
    summa = data.get("сумма_с_ндс")
    if isinstance(nds, (int, float)) and isinstance(summa, (int, float)):
        if nds > summa and summa > 0:
            if "проблемы" not in data:
                data["проблемы"] = []
            data["проблемы"].append(
                f"НДС ({nds}) больше суммы с НДС ({summa}) — возможная ошибка"
            )
    return data


def _validate_requisites_step(data: dict) -> dict:
    """Валидация реквизитов (для шаблонного/universal режима)."""
    warnings = validate_requisites(data)
    if warnings:
        if "проблемы" not in data:
            data["проблемы"] = []
        data["проблемы"].extend(warnings)
    return data


def _validate_flexible_step(data: dict) -> dict:
    """Валидация реквизитов (для гибкого режима)."""
    warnings = validate_flexible_requisites(data)
    if warnings:
        data["_предупреждения"] = warnings
    return data


# --- Реестр ---

_STEPS = {
    "split_inn_kpp": _split_inn_kpp_step,
    "validate_inn": _validate_inn_step,
    "validate_kpp": _validate_kpp_step,
    "clean_address": _clean_address_step,
    "fix_doc_number": _fix_doc_number_step,
    "postprocess_act": _postprocess_act_step,
    "check_nds_vs_summa": _check_nds_vs_summa_step,
    "validate_requisites": _validate_requisites_step,
    "validate_flexible": _validate_flexible_step,
}


def run_postprocessing(data: dict, steps: list[str]) -> dict:
    """Выполнить именованные шаги пост-обработки последовательно."""
    for step_name in steps:
        func = _STEPS.get(step_name)
        if func:
            data = func(data)
    return data
