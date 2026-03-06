"""Валидация ИНН, КПП, БИК, расчётных счетов."""

import re


def validate_inn(inn: str) -> bool:
    if not inn or not inn.isdigit():
        return False
    if len(inn) == 10:
        coeffs = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        check = sum(int(inn[i]) * coeffs[i] for i in range(9)) % 11 % 10
        return check == int(inn[9])
    elif len(inn) == 12:
        coeffs1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        coeffs2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        check1 = sum(int(inn[i]) * coeffs1[i] for i in range(10)) % 11 % 10
        check2 = sum(int(inn[i]) * coeffs2[i] for i in range(11)) % 11 % 10
        return check1 == int(inn[10]) and check2 == int(inn[11])
    return False


def find_fields(data: dict, field_name: str, prefix: str = "") -> list:
    results = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if key == field_name and isinstance(value, str):
            results.append((path, value))
        elif isinstance(value, dict):
            results.extend(find_fields(value, field_name, path))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    results.extend(find_fields(item, field_name, f"{path}[{i}]"))
    return results


def validate_requisites(data: dict) -> list[str]:
    """Валидация для шаблонного режима (плоская структура)."""
    warnings = []

    for key in ("покупатель_инн", "продавец_инн"):
        inn_value = data.get(key)
        if inn_value and isinstance(inn_value, str) and inn_value.isdigit():
            if len(inn_value) not in (10, 12):
                warnings.append(f"{key}: ИНН должен быть 10 или 12 цифр, получено {len(inn_value)}")
            elif not validate_inn(inn_value):
                warnings.append(f"{key}: ИНН не прошёл проверку контрольной суммы")

    for key in ("покупатель_кпп", "продавец_кпп"):
        kpp_value = data.get(key)
        if kpp_value and isinstance(kpp_value, str) and kpp_value.isdigit() and len(kpp_value) != 9:
            warnings.append(f"{key}: КПП должен быть 9 цифр, получено {len(kpp_value)}")

    return warnings


def _extract_inn_from_combined(value: str) -> tuple[str | None, str | None]:
    """Разделить ИНН/КПП из объединённой строки вроде '7723022111/772201001'."""
    if "/" in value:
        parts = value.split("/")
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else None
    # Может быть слитно: 10+9=19 цифр или 12+9=21 цифра
    digits = re.sub(r"\D", "", value)
    if len(digits) == 19:  # ИНН юрлица (10) + КПП (9)
        return digits[:10], digits[10:]
    if len(digits) == 21:  # ИНН ИП (12) + КПП (9)
        return digits[:12], digits[12:]
    return value, None


def validate_flexible_requisites(data: dict) -> list[str]:
    """Валидация для гибкого режима (произвольные ключи).

    Ищет поля, содержащие 'инн' или 'кпп' в названии ключа,
    и проверяет формат значений.
    """
    warnings = []

    for key, value in data.items():
        if not isinstance(value, str) or not value or value == "null" or value == "--":
            continue

        key_lower = key.lower()

        # Поля, содержащие "инн" в ключе
        if "инн" in key_lower:
            # Может быть объединённое поле ИНН/КПП
            if "/" in value:
                inn_part, kpp_part = _extract_inn_from_combined(value)
                if inn_part:
                    digits = re.sub(r"\D", "", inn_part)
                    if digits and len(digits) not in (10, 12):
                        warnings.append(f"{key}: ИНН должен быть 10 или 12 цифр, получено {len(digits)} ({inn_part})")
                    elif digits and not validate_inn(digits):
                        warnings.append(f"{key}: ИНН {inn_part} не прошёл проверку контрольной суммы")
                if kpp_part:
                    kpp_digits = re.sub(r"\D", "", kpp_part)
                    if kpp_digits and len(kpp_digits) != 9:
                        warnings.append(f"{key}: КПП должен быть 9 цифр, получено {len(kpp_digits)} ({kpp_part})")
            else:
                digits = re.sub(r"\D", "", value)
                if digits:
                    if len(digits) == 19:
                        inn_part, kpp_part = digits[:10], digits[10:]
                        if not validate_inn(inn_part):
                            warnings.append(f"{key}: ИНН {inn_part} (из слитной записи) не прошёл проверку")
                        warnings.append(f"{key}: ИНН и КПП слиты вместе ({value}), ожидалось через '/'")
                    elif len(digits) == 21:
                        inn_part, kpp_part = digits[:12], digits[12:]
                        if not validate_inn(inn_part):
                            warnings.append(f"{key}: ИНН {inn_part} (из слитной записи) не прошёл проверку")
                        warnings.append(f"{key}: ИНН и КПП слиты вместе ({value}), ожидалось через '/'")
                    elif len(digits) not in (10, 12):
                        warnings.append(f"{key}: ИНН должен быть 10 или 12 цифр, получено {len(digits)}")
                    elif not validate_inn(digits):
                        warnings.append(f"{key}: ИНН {value} не прошёл проверку контрольной суммы")

        elif "кпп" in key_lower:
            digits = re.sub(r"\D", "", value)
            if digits and len(digits) != 9:
                warnings.append(f"{key}: КПП должен быть 9 цифр, получено {len(digits)} ({value})")

    return warnings
