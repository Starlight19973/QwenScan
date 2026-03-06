"""Управление процессом vLLM: запуск, остановка, статус. Поддержка нескольких моделей."""

import asyncio
import os
import signal
import subprocess
from pathlib import Path

import httpx

from app.config import VLLM_BASE_URL, BASE_DIR, MODELS, ACTIVE_MODEL_ID

_active_model_id: str = ACTIVE_MODEL_ID
_vllm_proc: subprocess.Popen | None = None


def _get_script() -> Path:
    """Вернуть скрипт запуска для текущей выбранной модели."""
    model = MODELS.get(_active_model_id, {})
    return BASE_DIR / model.get("script", "start_vllm.sh")


def _find_vllm_pid() -> int | None:
    """Найти PID процесса vLLM/llama-server через /proc или ps."""
    global _vllm_proc
    if _vllm_proc and _vllm_proc.poll() is None:
        return _vllm_proc.pid
    # llama-server
    try:
        result = subprocess.run(
            ["pgrep", "-f", "llama-server.*--port 8001"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        if pids:
            return int(pids[0])
    except Exception:
        pass
    # vllm fallback
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server.*--port 8001"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        if pids:
            return int(pids[0])
    except Exception:
        pass
    return None


async def get_status() -> dict:
    """Проверить статус vLLM: running/stopped + info об активной модели."""
    pid = _find_vllm_pid()
    model = MODELS.get(_active_model_id, {})
    base = {
        "active_model_id": _active_model_id,
        "model_name": model.get("name", _active_model_id),
        "vram_gb": model.get("vram_gb"),
    }
    if not pid:
        return {"status": "stopped", "pid": None, "model": None, **base}

    # Проверить что реально отвечает
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/models")
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"status": "running", "pid": pid, "model": models[0] if models else None, **base}
    except Exception:
        return {"status": "starting", "pid": pid, "model": None, **base}


async def select_model(model_id: str) -> dict:
    """Выбрать активную модель без её загрузки."""
    global _active_model_id
    if model_id not in MODELS:
        raise ValueError(f"Неизвестная модель: {model_id}")
    _active_model_id = model_id
    return await get_status()


async def load_model() -> dict:
    """Запустить vLLM с выбранной моделью."""
    global _vllm_proc

    status = await get_status()
    if status["status"] in ("running", "starting"):
        return {"ok": False, "message": "Модель уже загружена", **status}

    script = _get_script()
    if not script.exists():
        return {"ok": False, "message": f"Скрипт не найден: {script}"}

    log = Path("/tmp/vllm_qwenscan.log")
    log_f = log.open("w")
    _vllm_proc = subprocess.Popen(
        ["bash", str(script)],
        stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Подождать пока стартанёт (до 90 сек)
    for _ in range(90):
        await asyncio.sleep(1)
        if _vllm_proc.poll() is not None:
            return {"ok": False, "message": "vLLM процесс завершился с ошибкой"}
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{VLLM_BASE_URL}/models")
                if resp.status_code == 200:
                    return {"ok": True, "message": "Модель загружена", "pid": _vllm_proc.pid}
        except Exception:
            continue

    return {"ok": True, "message": "vLLM запущен, модель загружается...", "pid": _vllm_proc.pid}


async def unload_model() -> dict:
    """Остановить vLLM процесс, освободить GPU."""
    global _vllm_proc

    pid = _find_vllm_pid()
    if not pid:
        return {"ok": False, "message": "Модель не загружена"}

    try:
        os.kill(pid, signal.SIGTERM)

        subprocess.run(
            ["pkill", "-f", "llama-server.*--port 8001"],
            timeout=5,
        )
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server.*--port 8001"],
            timeout=5,
        )
        subprocess.run(
            ["pkill", "-f", "from multiprocessing.*--model.*Qwen"],
            timeout=5,
        )

        for _ in range(15):
            await asyncio.sleep(1)
            if not _find_vllm_pid():
                _vllm_proc = None
                return {"ok": True, "message": "Модель выгружена, GPU освобождена"}

        os.kill(pid, signal.SIGKILL)
        subprocess.run(["pkill", "-9", "-f", "vllm.*--port 8001"], timeout=5)
        await asyncio.sleep(2)
        _vllm_proc = None
        return {"ok": True, "message": "Модель принудительно выгружена"}

    except Exception as e:
        return {"ok": False, "message": f"Ошибка: {e}"}
