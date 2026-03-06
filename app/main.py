"""QwenScan — FastAPI-приложение для извлечения данных из PDF.

Поддерживает пакетную загрузку, персистентные пакеты и возобновление после сбоя.
"""

import asyncio
import base64
import json
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from pathlib import Path
from app.config import UPLOADS_DIR, RESULTS_DIR, get_concurrency, set_concurrency
from app.batch_processor import merge_file_pages, start_processing, stop_processing, is_processing_active
from app.excel_export import generate_excel
from app import storage
from app import chat as chat_module
from datetime import datetime
from app.model_manager import get_status, load_model, unload_model, select_model
from app.vlm_client import call_vlm_chat_stream, call_vlm_chat
from app.template_registry import (
    list_templates,
    get_template,
    list_templates_with_system_prompts,
    update_template_system_prompt,
    create_template_with_system_prompt,
    delete_template_with_prompts,
)
from app.sync_api import sync_router
from app.ocr_overlay import process_pdf_to_searchable, build_searchable_pdf, extract_tags
from app.ocr_hint_registry import load_ocr_hints, update_ocr_hint
from urllib.parse import quote as url_quote


def _safe_content_disposition(filename: str) -> str:
    """Build Content-Disposition header safe for non-ASCII filenames (RFC 5987)."""
    try:
        filename.encode("ascii")
        return f'attachment; filename="{filename}"'
    except UnicodeEncodeError:
        encoded = url_quote(filename, safe="")
        return f"attachment; filename*=UTF-8''{encoded}"


app = FastAPI(title="QwenScan — Извлечение данных из документов",
    description="Пакетная обработка PDF-документов с помощью VLM (Qwen3-VL). Извлечение структурированных данных по шаблонам или свободным промптам.", docs_url=None, redoc_url=None, root_path="/qwenscan")

def _format_duration(delta_seconds: int) -> str:
    if delta_seconds < 60:
        return f"{delta_seconds} сек"
    minutes, seconds = divmod(delta_seconds, 60)
    if minutes < 60:
        return f"{minutes} мин {seconds} сек"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин {seconds} сек"


def _avg_page_ms(total_ms: int | None, samples: int | None) -> int | None:
    if total_ms is None or samples is None:
        return None
    if samples <= 0:
        return None
    try:
        return max(0, int(total_ms / samples))
    except Exception:
        return None


def _format_page_duration_ms(value_ms: int | None) -> str | None:
    if value_ms is None:
        return None
    if value_ms < 1000:
        return f"{value_ms} мс/стр"
    seconds = value_ms / 1000.0
    return f"{seconds:.2f} сек/стр"


def _calc_processing_time(batch: dict) -> str | None:
    """Calculate human-readable processing time from started_at/finished_at."""
    started = batch.get("started_at")
    if not started:
        return None
    try:
        t0 = datetime.fromisoformat(started)
        if batch.get("finished_at"):
            t1 = datetime.fromisoformat(batch["finished_at"])
        elif batch.get("status") == "processing":
            t1 = datetime.now()
        else:
            return None
        delta = max(0, int((t1 - t0).total_seconds()))
        return _format_duration(delta)
    except Exception:
        return None


def _calc_file_processing_time(file_info: dict) -> str | None:
    started = file_info.get("started_at")
    if not started:
        return None
    try:
        t0 = datetime.fromisoformat(started)
        if file_info.get("finished_at"):
            t1 = datetime.fromisoformat(file_info["finished_at"])
        elif file_info.get("status") in ("converting", "queued", "processing", "interrupted"):
            t1 = datetime.now()
        else:
            return None
        delta = max(0, int((t1 - t0).total_seconds()))
        return _format_duration(delta)
    except Exception:
        return None


def _calc_batch_avg_page_time(batch: dict) -> tuple[int | None, str | None]:
    """Average throughput time per page (wall-clock / pages_done)."""
    pages_done = batch.get("pages_done") or 0
    started = batch.get("started_at")
    if pages_done <= 0 or not started:
        return None, None
    try:
        t0 = datetime.fromisoformat(started)
        if batch.get("finished_at"):
            t1 = datetime.fromisoformat(batch["finished_at"])
        elif batch.get("status") == "processing":
            t1 = datetime.now()
        else:
            return None, None
        elapsed_ms = max(0, int((t1 - t0).total_seconds() * 1000))
        avg_ms = int(elapsed_ms / pages_done) if pages_done > 0 else None
        return avg_ms, _format_page_duration_ms(avg_ms)
    except Exception:
        return None, None


def _calc_file_avg_page_time(file_info: dict) -> tuple[int | None, str | None]:
    avg_ms = _avg_page_ms(file_info.get("page_time_ms_total"), file_info.get("page_time_samples"))
    return avg_ms, _format_page_duration_ms(avg_ms)



def _calc_batch_avg_doc_time(batch: dict) -> tuple[int | None, str | None]:
    """Average wall-clock time per document (elapsed / files_done)."""
    files = batch.get("files", [])
    files_done = sum(1 for fi in files if fi.get("status") == "done")
    started = batch.get("started_at")
    if files_done <= 0 or not started:
        return None, None
    try:
        t0 = datetime.fromisoformat(started)
        if batch.get("finished_at"):
            t1 = datetime.fromisoformat(batch["finished_at"])
        elif batch.get("status") == "processing":
            t1 = datetime.now()
        else:
            return None, None
        elapsed_ms = max(0, int((t1 - t0).total_seconds() * 1000))
        avg_ms = int(elapsed_ms / files_done)
        if avg_ms < 1000:
            text = f"{avg_ms} мс/док"
        else:
            text = f"{avg_ms / 1000:.1f} сек/док"
        return avg_ms, text
    except Exception:
        return None, None


def _calc_batch_avg_vlm_doc_time(batch: dict, batch_id: str = "") -> tuple[int | None, str | None]:
    """Average actual VLM processing time per document (from duration_ms in page_results)."""
    if not batch_id:
        return None, None
    total_ms = batch.get("page_time_ms_total") or 0
    samples = batch.get("page_time_samples") or 0
    if samples <= 0 or total_ms <= 0:
        return None, None
    avg_ms = int(total_ms / samples)
    if avg_ms < 1000:
        text = f"{avg_ms} мс/док"
    else:
        text = f"{avg_ms / 1000:.1f} сек/док"
    return avg_ms, text


def _serialize_progress_file(file_info: dict, batch_id: str = "") -> dict:
    avg_ms, avg_text = _calc_file_avg_page_time(file_info)
    actual_ms = None
    actual_text = None
    if file_info["status"] == "done" and batch_id:
        actual_ms = storage.get_file_duration_ms(batch_id, file_info["filename"])
        if actual_ms is not None:
            secs = actual_ms / 1000
            actual_text = f"{secs:.1f} сек" if secs >= 1 else f"{actual_ms} мс"
    return {
        "filename": file_info["filename"],
        "status": file_info["status"],
        "pages": file_info["pages"],
        "pages_done": file_info["pages_done"],
        "error": file_info.get("error"),
        "processing_time": _calc_file_processing_time(file_info),
        "actual_duration_ms": actual_ms,
        "actual_duration": actual_text,
        "avg_page_time_ms": avg_ms,
        "avg_page_time": avg_text,
        "detected_template_id": file_info.get("detected_template_id") or "",
    }


app.include_router(sync_router)

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return get_swagger_ui_html(openapi_url="openapi.json", title="QwenScan API")

@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    return get_redoc_html(openapi_url="openapi.json", title="QwenScan API")



@app.on_event("startup")
async def startup():
    storage.init_db()
    storage.mark_interrupted_batches()


# ── Templates ──────────────────────────────────────────────────────

@app.get("/api/templates", summary="Список шаблонов", tags=["Шаблоны извлечения"])
async def api_templates():
    """Список шаблонов извлечения.

    Возвращает все доступные шаблоны с полями (fields) для извлечения данных.

    Тест: `curl https://31.173.93.121/qwenscan/api/templates`
    """
    templates = list_templates()
    return JSONResponse([
        {"id": t["id"], "name": t["name"], "description": t["description"], "fields": t.get("fields", []), "page_range": t.get("page_range", "")}
        for t in templates
    ])


@app.get("/api/templates/{template_id}", summary="Детали шаблона", tags=["Шаблоны извлечения"])
async def api_template_detail(template_id: str):
    """Детали шаблона по ID.

    Возвращает имя, описание и список полей шаблона.

    Тест: `curl https://31.173.93.121/qwenscan/api/templates/upd`
    """
    tmpl = get_template(template_id)
    if not tmpl:
        raise HTTPException(404, f"Template '{template_id}' not found")
    return JSONResponse({
        "id": tmpl["id"],
        "name": tmpl["name"],
        "description": tmpl["description"],
        "fields": tmpl.get("fields", []),
        "page_range": tmpl.get("page_range", ""),
    })


# ── System prompts management (UI editor) ───────────────────────────

@app.get("/api/prompt-templates", summary="Шаблоны с промптами", tags=["Редактор шаблонов"])
async def api_prompt_templates():
    """Список шаблонов с системными промптами.

    Возвращает шаблоны вместе с системными промптами для VLM.
    Используется редактором шаблонов в веб-интерфейсе.

    Тест: `curl https://31.173.93.121/qwenscan/api/prompt-templates`
    """
    templates = list_templates_with_system_prompts()
    return JSONResponse(
        [
            {
                "id": t["id"],
                "name": t["name"],
                "description": t.get("description", ""),
                "system_prompt": t.get("system_prompt", ""),
                "fields": t.get("fields", []),
                "page_range": t.get("page_range", ""),
            }
            for t in templates
        ]
    )


@app.get("/api/prompt-templates/{template_id}", summary="Детали шаблона с промптом", tags=["Редактор шаблонов"])
async def api_prompt_template_detail(template_id: str):
    """Детали шаблона с промптом.

    Полная информация: имя, описание, системный промпт, поля.

    Тест: `curl https://31.173.93.121/qwenscan/api/prompt-templates/upd`
    """
    tmpl = get_template(template_id)
    if not tmpl:
        raise HTTPException(404, f"Template '{template_id}' not found")
    from app.template_registry import _DEFAULT_FIELDS
    return JSONResponse(
        {
            "id": tmpl["id"],
            "name": tmpl["name"],
            "description": tmpl.get("description", ""),
            "system_prompt": tmpl.get("system_prompt", ""),
            "fields": tmpl.get("fields") or _DEFAULT_FIELDS,
            "page_range": tmpl.get("page_range", ""),
        }
    )


@app.post("/api/prompt-templates", summary="Создать шаблон", tags=["Редактор шаблонов"])
async def api_prompt_template_create(request: Request):
    """Создать новый шаблон.

    Принимает JSON: name, description, system_prompt, fields.

    Тест: `curl -X POST .../api/prompt-templates -H 'Content-Type: application/json' -d '{"name":"test"}'`
    """
    data = await request.json()
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()
    system_prompt = (data.get("system_prompt") or "").strip()
    fields = data.get("fields")
    page_range = (data.get("page_range") or "").strip()

    try:
        created = create_template_with_system_prompt(
            name=name,
            description=description,
            system_prompt=system_prompt,
            fields=fields,
            page_range=page_range,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to create template: {e}")

    return JSONResponse(
        {
            "ok": True,
            "template": {
                "id": created["id"],
                "name": created["name"],
                "description": created.get("description", ""),
                "system_prompt": created.get("system_prompt", ""),
            },
        },
        status_code=201,
    )


@app.put("/api/prompt-templates/{template_id}", summary="Обновить шаблон", tags=["Редактор шаблонов"])
async def api_prompt_template_update(template_id: str, request: Request):
    """Обновить шаблон.

    Изменяет имя, описание, системный промпт или поля существующего шаблона.

    Тест: `curl -X PUT .../api/prompt-templates/ID -d '{"system_prompt":"..."}'`
    """
    data = await request.json()
    name = data.get("name")
    description = data.get("description")
    system_prompt = (data.get("system_prompt") or "").strip()
    fields = data.get("fields")
    page_range = data.get("page_range")

    try:
        updated = update_template_system_prompt(
            template_id=template_id,
            system_prompt=system_prompt,
            name=name,
            description=description,
            fields=fields,
            page_range=page_range,
        )
    except KeyError:
        raise HTTPException(404, f"Template '{template_id}' not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to update template: {e}")

    return JSONResponse(
        {
            "ok": True,
            "template": {
                "id": updated["id"],
                "name": updated["name"],
                "description": updated.get("description", ""),
                "system_prompt": updated.get("system_prompt", ""),
            },
        }
    )


@app.delete("/api/prompt-templates/{template_id}", summary="Удалить шаблон", tags=["Редактор шаблонов"])
async def api_prompt_template_delete(template_id: str):
    """Удалить шаблон.

    Удаляет шаблон и связанные промпты. Встроенные шаблоны удалить нельзя.

    Тест: `curl -X DELETE .../api/prompt-templates/my-template`
    """
    try:
        delete_template_with_prompts(template_id)
    except KeyError:
        raise HTTPException(404, f"Template '{template_id}' not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to delete template: {e}")
    return JSONResponse({"ok": True})



# ── OCR Hints: list / update ──────────────────────────────────────

@app.get("/api/ocr-hints", summary="Список OCR-режимов", tags=["OCR"])
async def api_ocr_hints():
    """Список OCR-режимов с хинтами для редактирования."""
    return JSONResponse(load_ocr_hints())


@app.put("/api/ocr-hints/{mode}", summary="Обновить OCR-режим", tags=["OCR"])
async def api_ocr_hint_update(mode: str, request: Request):
    """Обновить название, описание и хинт OCR-режима."""
    data = await request.json()
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    hint = data.get("hint", "")
    if not name:
        raise HTTPException(400, "name is required")
    try:
        updated = update_ocr_hint(mode, name, description, hint)
    except KeyError:
        raise HTTPException(404, f"OCR mode '{mode}' not found")
    return JSONResponse(updated)

# ── Batch: create / upload chunks / start ──────────────────────────

@app.post("/api/batch/create", summary="Создать пакет", tags=["Управление пакетами"])
async def create_batch(request: Request):
    """Создать пакет обработки.

    Принимает JSON: template_id и/или custom_prompt. Возвращает batch_id.

    Тест: `curl -X POST .../api/batch/create -H 'Content-Type: application/json' -d '{"template_id":"upd"}'`
    """
    data = await request.json()
    template_id = data.get("template_id", "")
    custom_prompt = data.get("custom_prompt", "")
    allowed_template_ids = data.get("allowed_template_ids", [])
    page_range = data.get("page_range", "")

    if template_id and not get_template(template_id):
        raise HTTPException(400, f"Template '{template_id}' not found")

    # Serialize allowed_template_ids as JSON string for DB storage
    allowed_templates_str = ""
    if isinstance(allowed_template_ids, list) and allowed_template_ids:
        import json as _json
        allowed_templates_str = _json.dumps(allowed_template_ids)

    batch_id = storage.next_batch_id()
    base_url = str(request.base_url).rstrip("/")
    storage.create_batch(
        batch_id, custom_prompt=custom_prompt, template_id=template_id,
        base_url=base_url, allowed_templates=allowed_templates_str,
        page_range=(page_range or "").strip(),
    )
    return JSONResponse({"batch_id": batch_id})


@app.post(
    "/api/batch/{batch_id}/upload",
    summary="Загрузка файлов в пакет", tags=["Управление пакетами"],
    response_description="Количество загруженных файлов",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "PDF файлы (можно несколько)"
                            }
                        }
                    }
                }
            }
        }
    },
)
async def upload_chunk(batch_id: str, request: Request):
    """Загрузить файлы в пакет.

    Принимает PDF-файлы через multipart/form-data (поле files). Можно загружать несколько файлов.

    Тест: `curl -X POST .../api/batch/BATCH_ID/upload -F files=@doc1.pdf -F files=@doc2.pdf`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    uploaded = 0
    try:
        async with request.form(max_files=10000, max_fields=10000, max_part_size=1024*1024*1024*10) as form:
            files = form.getlist("files")
            if not files:
                raise HTTPException(400, "No files in request")

            for f in files:
                if hasattr(f, "filename") and f.filename and f.filename.lower().endswith(".pdf"):
                    safe_name = f.filename.replace("/", "_").replace("\\", "_")
                    path = UPLOADS_DIR / f"{batch_id}_{safe_name}"
                    file_content = await f.read()
                    path.write_bytes(file_content)
                    storage.add_file(batch_id, safe_name)
                    uploaded += 1
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

    if uploaded == 0:
        raise HTTPException(400, "No PDF files in request")

    batch = storage.get_batch(batch_id)
    total_files = len(batch["files"])

    # Start processing immediately after first chunk — processor will pick up
    # new files as they arrive (batch_processor loops until upload_complete)
    if not is_processing_active(batch_id):
        start_processing(batch_id)

    return JSONResponse({"ok": True, "uploaded": uploaded, "total": total_files})


@app.post("/api/batch/{batch_id}/start", summary="Запустить обработку", tags=["Управление пакетами"])
async def start_batch(batch_id: str):
    """Запустить обработку пакета.

    Начинает распознавание всех файлов. Файлы должны быть загружены заранее.

    Тест: `curl -X POST .../api/batch/BATCH_ID/start`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    if not batch["files"]:
        raise HTTPException(400, "No files in batch")
    start_processing(batch_id)
    return JSONResponse({"ok": True})


@app.post("/api/batch/{batch_id}/upload-complete")
async def upload_complete(batch_id: str):
    """Signal that all files have been uploaded for this batch."""
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    storage.mark_upload_complete(batch_id)
    if not is_processing_active(batch_id) and batch["files"]:
        start_processing(batch_id)
    return JSONResponse({"ok": True})


@app.post("/api/batch/{batch_id}/resume", summary="Возобновить обработку", tags=["Управление пакетами"])
async def resume_batch(batch_id: str):
    """Возобновить прерванную обработку.

    Продолжает обработку пакета со статусом interrupted/error.

    Тест: `curl -X POST .../api/batch/BATCH_ID/resume`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    if batch["status"] not in ("interrupted", "error"):
        raise HTTPException(400, f"Cannot resume batch with status '{batch['status']}'")
    start_processing(batch_id)
    return JSONResponse({"ok": True})


@app.post("/api/batch/{batch_id}/stop", summary="Остановить обработку", tags=["Управление пакетами"])
async def stop_batch(batch_id: str):
    """Остановить обработку.

    Останавливает текущую обработку. Обработанные файлы сохраняются. Можно возобновить через /resume.

    Тест: `curl -X POST .../api/batch/BATCH_ID/stop`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    if batch["status"] != "processing":
        raise HTTPException(400, "Batch is not processing")
    stop_processing(batch_id)
    return JSONResponse({"ok": True})


# ── Batch: progress / results / excel / delete ────────────────────

@app.get("/api/batches", summary="Список пакетов", tags=["Управление пакетами"])
async def list_all_batches():
    """Список всех пакетов.

    Возвращает все пакеты со статусами и прогрессом.

    Тест: `curl https://31.173.93.121/qwenscan/api/batches`
    """
    batches = storage.list_batches()
    for b in batches:
        b["processing_time"] = _calc_processing_time(b)
    return JSONResponse(batches)


@app.get("/api/batch/{batch_id}", summary="Статус пакета", tags=["Управление пакетами"])
async def get_batch(batch_id: str):
    """Получить статус и информацию о пакете."""
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    return JSONResponse(batch)


@app.delete("/api/batch/{batch_id}", summary="Удалить пакет", tags=["Управление пакетами"])
async def delete_batch(batch_id: str):
    """Удалить пакет.

    Удаляет пакет, загруженные файлы и результаты. Нельзя удалить в процессе обработки.

    Тест: `curl -X DELETE .../api/batch/BATCH_ID`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    if batch["status"] == "processing":
        raise HTTPException(400, "Cannot delete batch while processing")
    filenames, excel = storage.delete_batch(batch_id)
    for fn in filenames:
        p = UPLOADS_DIR / f"{batch_id}_{fn}"
        p.unlink(missing_ok=True)
    if excel:
        p = RESULTS_DIR / excel
        p.unlink(missing_ok=True)
    return JSONResponse({"ok": True})


@app.get("/api/batch/{batch_id}/progress", summary="Прогресс обработки (SSE)", tags=["Результаты и экспорт"])
async def batch_progress(batch_id: str):
    """Прогресс обработки (SSE-стрим).

    Server-Sent Events с обновлениями в реальном времени: статус, страницы, время, файлы.

    Тест: `curl -N .../api/batch/BATCH_ID/progress`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    async def event_generator():
        last_sent_id = 0
        sent_file_results: set[str] = set()
        try:
            while True:
                b = storage.get_batch(batch_id)
                if not b:
                    break

                # Emit new page results incrementally
                new_rows = storage.get_new_page_results(batch_id, last_sent_id)
                # Calculate real throughput: wall-clock / pages_done
                throughput_ms = None
                pages_done = b.get("pages_done") or 0
                batch_started = b.get("started_at")
                if pages_done > 0 and batch_started:
                    try:
                        t0 = datetime.fromisoformat(batch_started)
                        elapsed_ms = max(0, int((datetime.now() - t0).total_seconds() * 1000))
                        throughput_ms = int(elapsed_ms / pages_done)
                    except Exception:
                        pass
                for row in new_rows:
                    try:
                        result = json.loads(row["result_json"]) if row["result_json"] else {}
                    except (json.JSONDecodeError, TypeError):
                        result = {}
                    yield {
                        "event": "page_result",
                        "data": json.dumps({
                            "id": row["id"],
                            "filename": row["filename"],
                            "page": row["page"],
                            "duration_ms": throughput_ms,
                            "result": result,
                        }, ensure_ascii=False),
                    }
                    last_sent_id = row["id"]

                # Emit file_result when a file finishes (merged across all pages)
                b_files = b.get("files", [])
                newly_done = [
                    fi["filename"] for fi in b_files
                    if fi["status"] == "done" and fi["filename"] not in sent_file_results
                ]
                if newly_done:
                    try:
                        file_page_rows = storage.get_page_results(batch_id, filenames=newly_done)
                        merged_files = merge_file_pages(file_page_rows)
                        for file_merged in merged_files:
                            fn = file_merged.get("filename", "")
                            yield {
                                "event": "file_result",
                                "data": json.dumps(
                                    {"filename": fn, "result": file_merged},
                                    ensure_ascii=False,
                                ),
                            }
                    except Exception as e:
                        print(f"[SSE] file_result emission error: {e}", flush=True)
                    # Mark ALL newly done files as sent (even if merge skipped them)
                    sent_file_results.update(newly_done)

                # Override: if DB says 'interrupted' but asyncio task is alive, treat as 'processing'
                effective_status = b["status"]
                if effective_status == "interrupted" and is_processing_active(batch_id):
                    effective_status = "processing"
                    print(f"[SSE] batch={batch_id} DB says interrupted but task active — overriding to processing", flush=True)

                batch_avg_ms, batch_avg_text = _calc_batch_avg_page_time(b)
                vlm_avg_ms, vlm_avg_text = _calc_batch_avg_vlm_doc_time(b, batch_id)
                done_files = [fi["filename"] for fi in b["files"] if fi["status"] == "done"]
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "status": effective_status,
                        "total_pages": b["total_pages"],
                        "pages_done": b["pages_done"],
                        "processing_time": _calc_processing_time(b),
                        "avg_page_time_ms": batch_avg_ms,
                        "avg_page_time": batch_avg_text,
                        "avg_vlm_doc_time": vlm_avg_text,
                        "auto_mode": not (b.get("template_id") or "").strip() and not (b.get("custom_prompt") or "").strip(),
                        "done_files_count": len(done_files),
                        "files_done": len(done_files),
                        "total_files": len(b["files"]),
                        "files": [_serialize_progress_file(fi, batch_id) for fi in b["files"]],
                    }, ensure_ascii=False),
                }
                if effective_status in ("done", "error", "interrupted"):
                    break
                await asyncio.sleep(1)
        except Exception as e:
            yield {
                "event": "stream_error",
                "data": json.dumps({"detail": f"SSE stream error: {e}"}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@app.get("/api/batch/{batch_id}/results", summary="Результаты обработки", tags=["Результаты и экспорт"])
async def batch_results(batch_id: str):
    """Результаты обработки пакета.

    Возвращает извлечённые данные из всех файлов (объединённые по файлам).

    Тест: `curl .../api/batch/BATCH_ID/results`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    all_results = storage.get_page_results(batch_id)
    merged = merge_file_pages(all_results)
    return JSONResponse({
        "batch_id": batch_id,
        "status": batch["status"],
        "results": merged,
    })


@app.get("/api/batch/{batch_id}/results/processed", summary="Частичные результаты", tags=["Результаты и экспорт"])
async def batch_processed_results(batch_id: str):
    """Частичные результаты (только обработанные файлы).

    Возвращает данные только для файлов со статусом done — до завершения всего пакета.

    Тест: `curl .../api/batch/BATCH_ID/results/processed`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    done_files = [fi["filename"] for fi in batch["files"] if fi["status"] == "done"]
    if not done_files:
        return JSONResponse({
            "batch_id": batch_id,
            "status": batch["status"],
            "processed_files": 0,
            "total_files": len(batch["files"]),
            "results": [],
        })

    all_results = storage.get_page_results(batch_id, filenames=done_files)
    merged = merge_file_pages(all_results)
    return JSONResponse({
        "batch_id": batch_id,
        "status": batch["status"],
        "processed_files": len(done_files),
        "total_files": len(batch["files"]),
        "results": merged,
    })


@app.post("/api/batch/{batch_id}/generate-excel", summary="Сформировать Excel", tags=["Результаты и экспорт"])
async def generate_batch_excel(batch_id: str):
    """Сформировать Excel с результатами.

    Генерирует .xlsx из результатов. Колонки по полям шаблона. Доступно после status=done.

    Тест: `curl -X POST .../api/batch/BATCH_ID/generate-excel`
    """
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    all_results = storage.get_page_results(batch_id)
    merged = merge_file_pages(all_results)
    if not merged:
        raise HTTPException(400, "No results to export")

    custom_prompt = batch.get("custom_prompt", "")
    template_id = batch.get("template_id", "")
    base_url = batch.get("base_url", "")

    # Auto-detection: resolve actual detected template(s) for correct Excel columns
    effective_template_id = template_id
    auto_mode = not template_id and not custom_prompt
    force_dynamic = False

    if auto_mode:
        detected_ids = set()
        for fi in batch.get("files", []):
            dtid = (fi.get("detected_template_id") or "").strip()
            if dtid:
                detected_ids.add(dtid)
        if len(detected_ids) == 1:
            effective_template_id = detected_ids.pop()
        elif len(detected_ids) > 1:
            force_dynamic = True

    if force_dynamic:
        excel_bytes = generate_excel(
            merged,
            batch_id=batch_id,
            base_url=base_url,
            template_id="",
            is_template_mode=False,
        )
    else:
        excel_bytes = generate_excel(
            merged,
            batch_id=batch_id,
            base_url=base_url,
            template_id=effective_template_id,
            is_template_mode=not custom_prompt and not effective_template_id,
        )
    excel_filename = f"batch_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path = RESULTS_DIR / excel_filename
    excel_path.write_bytes(excel_bytes)
    storage.update_batch(batch_id, excel_filename=excel_filename)

    return JSONResponse({"ok": True, "filename": excel_filename})


@app.get("/api/batch/{batch_id}/excel", summary="Скачать Excel", tags=["Результаты и экспорт"])
async def download_batch_excel(batch_id: str):
    """Скачать Excel-файл.

    Возвращает .xlsx. Сначала вызовите /generate-excel.

    Тест: `curl -O .../api/batch/BATCH_ID/excel`
    """
    batch = storage.get_batch(batch_id)
    if not batch or not batch.get("excel_filename"):
        raise HTTPException(404, "Excel not ready")
    path = RESULTS_DIR / batch["excel_filename"]
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(
        path,
        filename=batch["excel_filename"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── File serving ───────────────────────────────────────────────────

@app.get("/api/files/{batch_id}/{filename:path}", summary="Просмотр PDF", tags=["Файлы"])
async def serve_pdf(batch_id: str, filename: str):
    """Просмотр загруженного PDF.

    Возвращает оригинальный файл из пакета для предпросмотра.
    """
    path = UPLOADS_DIR / f"{batch_id}_{filename}"
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename, media_type="application/pdf")


# ── Chat with VLM ─────────────────────────────────────────────────

@app.get("/api/chat/documents", summary="Документы для чата", tags=["Чат"])
async def chat_documents():
    """Список документов из завершённых батчей для чата."""
    batches = storage.list_batches()
    docs = []
    for b_summary in batches:
        if b_summary.get("status") != "done":
            continue
        bid = b_summary["id"]
        b = storage.get_batch(bid)
        if not b:
            continue
        for fi in b.get("files", []):
            if fi.get("status") != "done":
                continue
            pdf_path = UPLOADS_DIR / f"{bid}_{fi['filename']}"
            if pdf_path.exists():
                docs.append({
                    "batch_id": bid,
                    "filename": fi["filename"],
                    "pages": fi.get("pages", 0),
                })
    return JSONResponse(docs)


@app.post("/api/chat/upload", summary="Загрузить PDF в чат", tags=["Чат"])
async def chat_upload(request: Request):
    """Загрузить PDF напрямую в чат (без создания батча)."""
    async with request.form() as form:
        f = form.get("file")
        if not f or not hasattr(f, "filename"):
            raise HTTPException(400, "No file uploaded")
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are accepted")

        safe_name = f.filename.replace("/", "_").replace("\\", "_")
        path = UPLOADS_DIR / f"chat_{safe_name}"
        content = await f.read()
        path.write_bytes(content)

    pages = chat_module.get_page_count(path)
    return JSONResponse({"filename": safe_name, "pages": pages})


@app.get("/api/page-image/{batch_id}/{filename:path}/{page}", summary="PNG страницы", tags=["Чат"])
async def page_image(batch_id: str, filename: str, page: int):
    """Отрендерить страницу PDF как PNG."""
    pdf_path = chat_module.resolve_pdf_path(batch_id, filename)
    if not pdf_path:
        raise HTTPException(404, "PDF not found")
    try:
        img_bytes = chat_module.get_or_render_page(pdf_path, page)
    except IndexError:
        raise HTTPException(400, "Page out of range")
    return Response(content=img_bytes, media_type="image/png")


@app.post("/api/chat/start", summary="Создать сессию чата", tags=["Чат"])
async def chat_start(request: Request):
    """Создать сессию чата. Без filename — свободный режим."""
    data = await request.json()
    batch_id = data.get("batch_id")
    filename = data.get("filename", "")
    page = data.get("page", 0)

    if filename:
        pdf_path = chat_module.resolve_pdf_path(batch_id, filename)
        if not pdf_path:
            raise HTTPException(404, "PDF not found")
        chat_id = chat_module.create_session(batch_id, filename, page, pdf_path)
    else:
        chat_id = chat_module.create_session()

    return JSONResponse({"chat_id": chat_id})


@app.post("/api/chat/{chat_id}/message", summary="Отправить сообщение (SSE)", tags=["Чат"])
async def chat_message(chat_id: str, request: Request):
    """Отправить сообщение в чат. Поддерживает JSON и multipart (с изображениями)."""
    from app.config import CHAT_MAX_IMAGES_PER_MESSAGE
    from app.pdf_utils import pdf_to_images

    session = chat_module.get_session(chat_id)
    if not session:
        raise HTTPException(404, "Chat session not found")

    content_type = request.headers.get("content-type", "")
    message = ""
    image_b64_list: list[str] = []

    if "multipart/form-data" in content_type:
        async with request.form() as form:
            message = (form.get("message") or "").strip()
            for key in form:
                if not key.startswith("image"):
                    continue
                upload = form[key]
                if not hasattr(upload, "read"):
                    continue
                raw = await upload.read()
                if upload.filename and upload.filename.lower().endswith(".pdf"):
                    pdf_imgs = pdf_to_images(raw)
                    for img_bytes in pdf_imgs[:CHAT_MAX_IMAGES_PER_MESSAGE - len(image_b64_list)]:
                        image_b64_list.append(base64.b64encode(img_bytes).decode())
                else:
                    image_b64_list.append(base64.b64encode(raw).decode())
            image_b64_list = image_b64_list[:CHAT_MAX_IMAGES_PER_MESSAGE]
    else:
        data = await request.json()
        message = (data.get("message") or "").strip()

    if not message and not image_b64_list:
        raise HTTPException(400, "Empty message")

    chat_module.add_message(
        chat_id, "user", message,
        images=image_b64_list if image_b64_list else None,
    )

    page_image_b64: str | None = None
    if session.get("mode") == "document" and session.get("pdf_path"):
        try:
            page_bytes = chat_module.get_or_render_page(session["pdf_path"], session["page"])
            page_image_b64 = base64.b64encode(page_bytes).decode()
        except Exception:
            pass

    vlm_messages = chat_module.build_vlm_messages(session, page_image_b64)

    async def stream_response():
        full_text = ""
        try:
            async for chunk in call_vlm_chat_stream(vlm_messages):
                full_text += chunk
                yield {
                    "event": "delta",
                    "data": json.dumps({"text": chunk}, ensure_ascii=False),
                }
            chat_module.add_message(chat_id, "assistant", full_text)
            yield {
                "event": "done",
                "data": json.dumps({"text": full_text}, ensure_ascii=False),
            }
        except Exception as e:
            if full_text:
                chat_module.add_message(chat_id, "assistant", full_text)
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(stream_response())


@app.get("/api/chat/{chat_id}/history", summary="История чата", tags=["Чат"])
async def chat_history(chat_id: str):
    """Вернуть историю сообщений сессии (без base64 данных изображений)."""
    session = chat_module.get_session(chat_id)
    if not session:
        raise HTTPException(404, "Chat session not found")

    messages_out = []
    for msg in session["messages"]:
        m = {"role": msg["role"], "content": msg["content"], "ts": msg["ts"]}
        if msg.get("images"):
            m["image_count"] = len(msg["images"])
        messages_out.append(m)

    return JSONResponse({
        "chat_id": chat_id,
        "filename": session.get("filename", ""),
        "page": session.get("page", 0),
        "mode": session.get("mode", "document"),
        "messages": messages_out,
    })


# ── Model management ──────────────────────────────────────────────

@app.get("/api/model/status", summary="Статус модели", tags=["Модель VLM"])
async def model_status():
    """Статус модели VLM.

    Загружена ли модель, GPU, использование видеопамяти.

    Тест: `curl https://31.173.93.121/qwenscan/api/model/status`
    """
    return JSONResponse(await get_status())


@app.post("/api/model/load", summary="Загрузить модель", tags=["Модель VLM"])
async def model_load():
    """Загрузить модель VLM в GPU.

    Загружает Qwen3-VL в видеопамять (~20 ГБ). Первый запуск ~30 сек.

    Тест: `curl -X POST https://31.173.93.121/qwenscan/api/model/load`
    """
    result = await load_model()
    return JSONResponse(result)


@app.post("/api/model/unload", summary="Выгрузить модель", tags=["Модель VLM"])
async def model_unload():
    """Выгрузить модель VLM из GPU.

    Освобождает ~20 ГБ видеопамяти. Для переключения на Surya OCR.

    Тест: `curl -X POST https://31.173.93.121/qwenscan/api/model/unload`
    """
    result = await unload_model()
    return JSONResponse(result)


# ── GPU stats ─────────────────────────────────────────────────────



@app.get("/api/models", summary="Список доступных моделей", tags=["Модель VLM"])
async def api_models():
    from app.config import MODELS
    return JSONResponse([
        {"id": m["id"], "name": m["name"], "description": m["description"], "vram_gb": m["vram_gb"]}
        for m in MODELS.values()
    ])


@app.post("/api/model/select", summary="Выбрать модель", tags=["Модель VLM"])
async def api_model_select(request: Request):
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id:
        raise HTTPException(400, "model_id required")
    try:
        result = await select_model(model_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return JSONResponse(result)


@app.get("/api/gpu/stats", summary="Статистика GPU", tags=["Модель VLM"])
async def gpu_stats():
    """Текущая загрузка GPU (nvidia-smi)."""
    import subprocess
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0:
            return JSONResponse({"error": "nvidia-smi failed"}, status_code=500)
        parts = [p.strip() for p in proc.stdout.strip().split(",")]
        if len(parts) < 4:
            return JSONResponse({"error": "unexpected nvidia-smi output"}, status_code=500)
        return JSONResponse({
            "gpu_util": int(parts[0]),
            "vram_used_mb": int(parts[1]),
            "vram_total_mb": int(parts[2]),
            "temp_c": int(parts[3]),
        })
    except FileNotFoundError:
        return JSONResponse({"error": "nvidia-smi not found"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# ── Settings ──────────────────────────────────────────────────────

@app.get("/api/settings/concurrency", summary="Текущая параллельность", tags=["Настройки"])
async def api_get_concurrency():
    return JSONResponse({"concurrency": get_concurrency()})


@app.put("/api/settings/concurrency", summary="Изменить параллельность", tags=["Настройки"])
async def api_set_concurrency(request: Request):
    data = await request.json()
    value = data.get("concurrency")
    if not isinstance(value, int) or value < 1 or value > 8:
        raise HTTPException(400, "concurrency must be 1-8")
    new_val = set_concurrency(value)
    return JSONResponse({"concurrency": new_val})


# ── OCR / Searchable PDF ──────────────────────────────────────────

@app.post("/api/searchable-pdf", summary="Создать searchable PDF", tags=["OCR"])
async def create_searchable_pdf_endpoint(request: Request):
    """Загрузить PDF → получить searchable PDF с невидимым текстовым слоем.

    Отправить: multipart/form-data, поле file (PDF).
    Вернёт: PDF с невидимым текстовым слоем.
    """
    async with request.form() as form:
        f = form.get("file")
        if not f or not hasattr(f, "filename"):
            raise HTTPException(400, "No file uploaded")
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are accepted")
        pdf_bytes = await f.read()

    try:
        result_bytes = await process_pdf_to_searchable(pdf_bytes)
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")

    safe_name = (f.filename or "document.pdf").replace("/", "_").replace("\\", "_")
    out_name = f"searchable_{safe_name}"
    return Response(
        content=result_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@app.post("/api/batch/{batch_id}/generate-searchable-pdf",
          summary="Создать searchable PDF для пакета", tags=["OCR"])
async def generate_batch_searchable_pdf(batch_id: str):
    """Построить searchable PDF для всех файлов пакета из OCR-данных в БД."""
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    results = []
    for file_info in batch.get("files", []):
        filename = file_info["filename"]
        if file_info.get("status") != "done":
            continue

        pdf_path = UPLOADS_DIR / f"{batch_id}_{filename}"
        if not pdf_path.exists():
            continue

        rows = storage.get_page_results(batch_id, filenames=[filename])
        if not rows:
            continue

        rows_sorted = sorted(rows, key=lambda x: x.get("page", 0))
        ocr_results = [r.get("lines", []) for r in rows_sorted]

        pdf_bytes = pdf_path.read_bytes()
        result_bytes = build_searchable_pdf(pdf_bytes, ocr_results)
        result_filename = f"searchable_{batch_id}_{filename}"
        (RESULTS_DIR / result_filename).write_bytes(result_bytes)

        results.append({"filename": filename, "searchable_filename": result_filename})

    return JSONResponse({"batch_id": batch_id, "files": results})


@app.get("/api/batch/{batch_id}/searchable-pdf/{filename:path}",
         summary="Скачать searchable PDF", tags=["OCR"])
async def download_searchable_pdf(batch_id: str, filename: str):
    """Скачать searchable PDF для конкретного файла."""
    safe = filename.replace("/", "_").replace("\\", "_")
    result_filename = f"searchable_{batch_id}_{safe}"
    result_path = RESULTS_DIR / result_filename
    if not result_path.exists():
        raise HTTPException(404, "Searchable PDF not found")
    return FileResponse(
        result_path,
        filename=f"searchable_{safe}",
        media_type="application/pdf",
    )




@app.get("/api/batch/{batch_id}/ocr-text/{filename:path}",
         summary="OCR текст по страницам", tags=["OCR"])
async def get_ocr_text(batch_id: str, filename: str):
    """Получить распознанный OCR-текст для файла (по страницам)."""
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    rows = storage.get_page_results(batch_id, filenames=[filename])
    pages = []
    for r in sorted(rows, key=lambda x: x.get("page", 0)):
        pg = r.get("page", 0)
        ocr_text = r.get("ocr_text", "")
        lines = r.get("lines", [])
        pages.append({"page": pg, "text": ocr_text, "lines": lines})
    return JSONResponse({"batch_id": batch_id, "filename": filename, "pages": pages})


@app.get("/api/batch/{batch_id}/page-image/{filename:path}/{page}",
         summary="PNG страницы из батча", tags=["OCR"])
async def batch_page_image(batch_id: str, filename: str, page: int):
    """Рендерить страницу PDF из батча как PNG."""
    pdf_path = UPLOADS_DIR / f"{batch_id}_{filename}"
    if not pdf_path.exists():
        raise HTTPException(404, "PDF not found")
    import fitz
    try:
        doc = fitz.open(pdf_path)
        if page < 1 or page > len(doc):
            doc.close()
            raise HTTPException(400, "Page out of range")
        pg = doc[page - 1]
        mat = fitz.Matrix(150 / 72.0, 150 / 72.0)
        pix = pg.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Render failed: {e}")
    return Response(content=img_bytes, media_type="image/png")

@app.post("/api/batch/create-ocr", summary="Создать OCR-пакет", tags=["OCR"])
async def create_ocr_batch(request: Request):
    """Создать пакет в OCR-режиме. Возвращает batch_id."""
    batch_id = storage.next_batch_id()
    base_url = str(request.base_url).rstrip("/")
    # Parse ocr_type from JSON body or default to 'universal'
    ocr_type = "universal"
    try:
        body = await request.json()
        ocr_type = body.get("ocr_type", "universal")
    except Exception:
        pass
    storage.create_batch(batch_id, base_url=base_url, mode="ocr")
    storage.update_batch(batch_id, ocr_type=ocr_type)
    return JSONResponse({"batch_id": batch_id})


# ── Tags CRUD ──────────────────────────────────────────────────────

@app.get("/api/batch/{batch_id}/tags/{filename:path}",
         summary="Получить теги файла", tags=["OCR"])
async def get_file_tags(batch_id: str, filename: str):
    """Получить извлечённые теги для файла."""
    tags = storage.get_tags(batch_id, filename)
    if tags is None:
        return JSONResponse({"tags": {"names": [], "dates": [], "documents": []}})
    return JSONResponse({"tags": tags})


@app.put("/api/batch/{batch_id}/tags/{filename:path}",
         summary="Обновить тег + автозамена", tags=["OCR"])
async def update_file_tags(batch_id: str, filename: str, request: Request):
    """Обновить тег и применить автозамену в OCR тексте.

    Body: {"old": "old_value", "new": "new_value", "category": "names|dates|documents"}
    """
    body = await request.json()
    old_val = body.get("old", "")
    new_val = body.get("new", "")
    category = body.get("category", "")

    if not old_val or not new_val or not category:
        raise HTTPException(400, "old, new, category are required")

    # 1. Update tags
    tags = storage.get_tags(batch_id, filename)
    if tags is None:
        tags = {"names": [], "dates": [], "documents": []}

    if category in tags and isinstance(tags[category], list):
        tags[category] = [new_val if v == old_val else v for v in tags[category]]
    storage.update_tags(batch_id, filename, tags)

    # 2. Find-replace in OCR text for all pages
    rows = storage.get_page_results(batch_id, filenames=[filename])
    for r in rows:
        pg = r.get("page", 0)
        ocr_text = r.get("ocr_text", "")
        if old_val in ocr_text:
            new_text = ocr_text.replace(old_val, new_val)
            storage.update_page_ocr_text(batch_id, filename, pg, new_text)

    return JSONResponse({"ok": True, "tags": tags})


@app.post("/api/batch/{batch_id}/regenerate-pdf/{filename:path}",
          summary="Перегенерировать searchable PDF", tags=["OCR"])
async def regenerate_searchable_pdf(batch_id: str, filename: str):
    """Перегенерировать searchable PDF с обновлённым OCR текстом."""
    from app.config import RESULTS_DIR

    pdf_path = UPLOADS_DIR / f"{batch_id}_{filename}"
    if not pdf_path.exists():
        raise HTTPException(404, "Original PDF not found")

    # Get updated OCR results from DB
    rows = storage.get_page_results(batch_id, filenames=[filename])
    rows_sorted = sorted(rows, key=lambda x: x.get("page", 0))

    ocr_results = []
    for r in rows_sorted:
        lines = r.get("lines", [])
        # Update line text from the updated ocr_text
        ocr_text = r.get("ocr_text", "")
        ocr_results.append(lines)

    pdf_bytes = pdf_path.read_bytes()
    result_bytes = build_searchable_pdf(pdf_bytes, ocr_results)

    result_filename = f"searchable_{batch_id}_{filename}"
    result_path = RESULTS_DIR / result_filename
    result_path.write_bytes(result_bytes)

    return JSONResponse({"ok": True, "filename": result_filename})


# ── Export ─────────────────────────────────────────────────────────

@app.get("/api/batch/{batch_id}/export/{filename:path}",
         summary="Экспорт OCR результатов", tags=["OCR"])
async def export_ocr_results(batch_id: str, filename: str, format: str = "txt"):
    """Экспорт OCR результатов в разных форматах: txt, json, tags-txt, tags-json."""
    batch = storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    safe = filename.replace("/", "_").replace("\\", "_")

    if format == "txt":
        rows = storage.get_page_results(batch_id, filenames=[filename])
        rows_sorted = sorted(rows, key=lambda x: x.get("page", 0))
        parts = []
        for r in rows_sorted:
            pg = r.get("page", 0)
            text = r.get("ocr_text", "")
            parts.append(f"--- Страница {pg} ---\n{text}")
        content = "\n\n".join(parts)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": _safe_content_disposition(safe + ".txt")},
        )

    elif format == "json":
        rows = storage.get_page_results(batch_id, filenames=[filename])
        rows_sorted = sorted(rows, key=lambda x: x.get("page", 0))
        pages = []
        for r in rows_sorted:
            pages.append({
                "page": r.get("page", 0),
                "lines": r.get("lines", []),
            })
        return Response(
            content=json.dumps({"pages": pages}, ensure_ascii=False, indent=2).encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": _safe_content_disposition(safe + ".json")},
        )

    elif format == "tags-txt":
        tags = storage.get_tags(batch_id, filename)
        if not tags:
            tags = {"names": [], "dates": [], "documents": []}
        lines = []
        lines.append("Названия:")
        for n in tags.get("names", []):
            lines.append(f"  - {n}")
        lines.append("\nДаты:")
        for d in tags.get("dates", []):
            lines.append(f"  - {d}")
        lines.append("\nДокументы:")
        for d in tags.get("documents", []):
            lines.append(f"  - {d}")
        content = "\n".join(lines)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": _safe_content_disposition(safe + "_tags.txt")},
        )

    elif format == "tags-json":
        tags = storage.get_tags(batch_id, filename)
        if not tags:
            tags = {"names": [], "dates": [], "documents": []}
        return Response(
            content=json.dumps(tags, ensure_ascii=False, indent=2).encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": _safe_content_disposition(safe + "_tags.json")},
        )

    else:
        raise HTTPException(400, f"Unknown format: {format}. Use: txt, json, tags-txt, tags-json")


# ── Web UI ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return HTMLResponse(content=HTML_PAGE, headers={"Cache-Control": "no-cache"})


HTML_PAGE = (Path(__file__).resolve().parent / "templates" / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
