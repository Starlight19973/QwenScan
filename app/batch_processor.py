"""Batch processing with persistent storage and resume support.

Processing goes in groups of PDF_BATCH_SIZE files:
convert group to images -> process via VLM -> free memory -> next group.
Each page result is saved to SQLite immediately, so no data is lost on crash.
"""

import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from collections import OrderedDict

from app.config import UPLOADS_DIR, MAX_CONCURRENT_REQUESTS, MAX_DOCUMENT_PAGES, get_concurrency
from app.pdf_utils import pdf_to_images, get_document_dpi
from app.pipeline import process_page, process_document
from app.template_registry import resolve_prompts, list_templates, get_template_field_keys
from app.vlm_client import call_vlm
from app import storage
from app.ocr_overlay import process_pdf_to_searchable, extract_tags

PDF_BATCH_SIZE = 16
_SUM_FIELDS = frozenset({"ндс", "сумма_с_ндс"})
_AUTO_DETECT_MAX_TOKENS = 160

# Строгий список атрибутов задаётся отдельно от промпта (templates.json fields)
# и используется для фильтрации вывода модели: никаких \"придуманных\" ключей.
_INTERNAL_KEYS = frozenset({
    "filename", "page", "parse_error", "raw_text", "error",
    "проблемы", "уверенность", "extracted", "_предупреждения",
})
_SCHEMA_CACHE: dict[str, list[str]] = {}


def clear_schema_cache():
    """Clear cached schema keys. Called when templates are updated."""
    _SCHEMA_CACHE.clear()


def _safe_strip(value):
    if isinstance(value, str):
        v = value.strip()
        if v in ("", "--", "null"):
            return None
        return v
    return value


def _extract_schema_keys_from_user_prompt(user_prompt: str) -> list[str]:
    """Попытаться извлечь список ключей из user_prompt, если он содержит JSON-объект.

    Поддерживает сценарий custom_prompt вида:
      Извлеки: {"поле1":"string","поле2":"number"}
    """
    text = (user_prompt or "").strip()
    if not text:
        return []
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    snippet = text[start:end + 1].strip()
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        # Попытка исправить trailing commas
        fixed = re.sub(r",\\s*([}\\]])", r"\\1", snippet)
        try:
            obj = json.loads(fixed)
        except Exception:
            return []
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    keys: list[str] = []
    for k in obj.keys():
        if isinstance(k, str) and k.strip():
            keys.append(k.strip())
    return keys


def _get_allowed_keys_for_template(template_id: str) -> list[str]:
    tid = (template_id or "").strip()
    if not tid:
        return []
    cached = _SCHEMA_CACHE.get(tid)
    if cached is not None:
        return cached
    keys = get_template_field_keys(tid)
    _SCHEMA_CACHE[tid] = keys
    return keys


def _enforce_output_schema(
    result: dict,
    *,
    allowed_keys: list[str],
    batch_id: str,
    filename: str,
    page_num: int,
    template_id: str,
) -> dict:
    if not isinstance(result, dict):
        return {"parse_error": True, "raw_text": str(result)}

    src = dict(result)
    cleaned: dict = {}

    # Preserve service keys
    for k, v in src.items():
        if (isinstance(k, str) and k.startswith("_")) or k in _INTERNAL_KEYS:
            cleaned[k] = v

    allowed_set = set(allowed_keys or [])
    for k in allowed_keys:
        cleaned[k] = _safe_strip(src.get(k))

    dropped = []
    for k in src.keys():
        if k in allowed_set:
            continue
        if (isinstance(k, str) and k.startswith("_")) or k in _INTERNAL_KEYS:
            continue
        dropped.append(k)
    if dropped:
        print(
            f"[SCHEMA_DROP] batch={batch_id} file={filename} page={page_num} template={template_id} dropped={dropped[:40]}",
            flush=True,
        )

    return cleaned

# Track active processing tasks
_active_tasks: dict[str, asyncio.Task] = {}


_FILENAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"упд|upd", re.IGNORECASE), "upd"),
    (re.compile(r"счёт.*факт|счет.*факт|schet.*fakt", re.IGNORECASE), "schet_faktura"),
    (re.compile(r"торг|torg", re.IGNORECASE), "torg12"),
    (re.compile(r"транспорт|transport", re.IGNORECASE), "transportnaya"),
    (re.compile(r"деклар", re.IGNORECASE), "deklaratsiya"),
    (re.compile(r"доверен", re.IGNORECASE), "doverennost"),
    (re.compile(r"счёт.*догов|счет.*догов|schet.*dog", re.IGNORECASE), "schet_dogovor"),
    (re.compile(r"счёт.*опл|счет.*опл|schet.*na", re.IGNORECASE), "schet_na_oplatu"),
    (re.compile(r"акт|akt", re.IGNORECASE), "akt"),
]


def _classify_by_filename(filename: str, allowed_ids: set[str]) -> str | None:
    """Try to detect template_id from filename patterns. Returns None if no match."""
    name = (filename or "").lower()
    for pattern, tid in _FILENAME_PATTERNS:
        if pattern.search(name) and tid in allowed_ids:
            return tid
    return None


def parse_page_range(spec: str, total_pages: int) -> set[int] | None:
    """Parse page range spec like '1,3-5,last' into a set of 1-based page numbers.

    Returns None if all pages should be processed.
    """
    text = (spec or "").strip().lower()
    if not text or text in ("все", "all"):
        return None

    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "last":
            pages.add(total_pages)
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                start = int(bounds[0].strip())
                end_str = bounds[1].strip()
                end = total_pages if end_str == "last" else int(end_str)
                for p in range(max(1, start), min(total_pages, end) + 1):
                    pages.add(p)
            except (ValueError, IndexError):
                continue
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.add(p)
            except ValueError:
                continue

    return pages if pages else None


def create_batch(filenames, custom_prompt="", base_url="", template_id=""):
    batch_id = str(uuid.uuid4())[:8]
    storage.create_batch(batch_id, custom_prompt=custom_prompt, template_id=template_id, base_url=base_url)
    for fn in filenames:
        storage.add_file(batch_id, fn)
    return batch_id


def merge_file_pages(results: list[dict]) -> list[dict]:
    """Merge pages of the same file into one row."""
    by_file: OrderedDict[str, list[dict]] = OrderedDict()
    for r in results:
        fn = r.get("filename", "unknown")
        by_file.setdefault(fn, []).append(r)

    merged: list[dict] = []

    for fn, pages in by_file.items():
        real_pages = []
        for p in pages:
            meaningful_keys = {
                k for k, v in p.items()
                if k not in ("filename", "page", "parse_error", "raw_text", "error",
                             "_страница_без_данных", "extracted")
                and v is not None and v != "" and v != "--" and v != "null"
            }
            if meaningful_keys:
                real_pages.append(p)

        if not real_pages:
            continue

        if len(real_pages) == 1:
            merged.append(real_pages[0])
            continue

        base = dict(real_pages[0])

        max_item_num = 0
        for key in base:
            if key.startswith("товары_") and "_" in key:
                parts = key.rsplit("_", 1)
                if parts[-1].isdigit():
                    max_item_num = max(max_item_num, int(parts[-1]))

        for extra in real_pages[1:]:
            extra_items: dict[int, dict[str, str]] = {}
            for key, val in extra.items():
                if key in ("filename", "page", "parse_error", "raw_text",
                           "error", "_страница_без_данных"):
                    continue

                if key.startswith("товары_") and "_" in key:
                    parts = key.rsplit("_", 1)
                    if parts[-1].isdigit():
                        item_num = int(parts[-1])
                        field_prefix = parts[0]
                        extra_items.setdefault(item_num, {})[field_prefix] = val
                        continue

                if key in _SUM_FIELDS:
                    if val is not None and val != "" and val != "--" and val != "null":
                        base[key] = val
                    continue

                if val is not None and val != "" and val != "--" and val != "null":
                    if key not in base or base[key] is None or base[key] == "" or base[key] == "--":
                        base[key] = val

            for old_num in sorted(extra_items):
                max_item_num += 1
                for field_prefix, val in extra_items[old_num].items():
                    base[f"{field_prefix}_{max_item_num}"] = val

        merged.append(base)

    return merged


def is_processing_active(batch_id) -> bool:
    """Check if a processing task is currently active for this batch."""
    task = _active_tasks.get(batch_id)
    return task is not None and not task.done()


def start_processing(batch_id):
    """Start or resume processing as an asyncio task."""
    if is_processing_active(batch_id):
        return _active_tasks[batch_id]
    task = asyncio.create_task(process_batch(batch_id))
    _active_tasks[batch_id] = task
    return task


def stop_processing(batch_id):
    """Cancel a running batch processing task."""
    task = _active_tasks.get(batch_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


def _norm_type(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "_", (value or "").lower()).strip("_")


def _build_auto_detect_prompt(templates: list[dict]) -> str:
    lines = []
    for t in templates:
        tid = str(t.get("id", "")).strip()
        if not tid:
            continue
        name = str(t.get("name", "")).strip()
        desc = str(t.get("description", "")).strip()
        line = f"- {tid}: {name}"
        if desc:
            line += f" — {desc}"
        lines.append(line)
    return "Список template_id:\n" + "\n".join(lines)


def _normalize_detected_template_id(raw_value: str, templates: list[dict]) -> str:
    allowed_ids_ordered = [str(t.get("id", "")).strip() for t in templates if str(t.get("id", "")).strip()]
    allowed_ids = set(allowed_ids_ordered)
    value = (raw_value or "").strip()
    if value in allowed_ids:
        return value

    by_norm_id = {_norm_type(t["id"]): t["id"] for t in templates if t.get("id")}
    by_norm_name = {_norm_type(t.get("name", "")): t["id"] for t in templates if t.get("id")}

    aliases = {
        "upd": "upd",
        "универсальный_передаточный_документ": "upd",
        "счет_фактура": "schet_faktura",
        "счёт_фактура": "schet_faktura",
        "schet_faktura": "schet_faktura",
        "акт": "akt",
        "akt": "akt",
        "товарная_накладная": "torg12",
        "торг_12": "torg12",
        "torg12": "torg12",
        "счет_на_оплату": "schet_na_oplatu",
        "счёт_на_оплату": "schet_na_oplatu",
        "schet_na_oplatu": "schet_na_oplatu",
        "счет_договор": "schet_dogovor",
        "счёт_договор": "schet_dogovor",
        "schet_dogovor": "schet_dogovor",
        "договор": "dogovor",
        "dogovor": "dogovor",
        "транспортная_накладная": "transportnaya",
        "transportnaya": "transportnaya",
        "декларация": "deklaratsiya",
        "deklaratsiya": "deklaratsiya",
        "платежное_поручение": "platezhnoe",
        "платёжное_поручение": "platezhnoe",
        "platezhnoe": "platezhnoe",
        "доверенность": "doverennost",
        "doverennost": "doverennost",
        "универсальный": "universal",
        "universal": "universal",
        "паспорт": "pasport",
        "pasport": "pasport",
        "паспорт_рф": "pasport",
        "passport": "pasport",
    }

    normalized = _norm_type(value)
    candidate = by_norm_id.get(normalized) or by_norm_name.get(normalized) or aliases.get(normalized)
    if candidate in allowed_ids:
        return candidate

    for key, tid in by_norm_id.items():
        if key and (key in normalized or normalized in key):
            return tid
    for key, tid in by_norm_name.items():
        if key and (key in normalized or normalized in key):
            return tid

    if "universal" in allowed_ids:
        return "universal"
    return allowed_ids_ordered[0] if allowed_ids_ordered else ""


async def _detect_template_id_for_file(images: list[bytes], templates: list[dict]) -> str:
    allowed_ids = [str(t.get("id", "")).strip() for t in templates if t.get("id")]
    if not allowed_ids:
        return ""

    system_prompt = (
        "Ты классификатор документов.\n"
        "Верни СТРОГО JSON вида {\"template_id\":\"<id>\"}.\n"
        "template_id обязан быть одним из списка.\n"
        "Если не уверен — выбери universal.\n"
        "Никакого текста вне JSON.\n/no_think"
    )
    user_prompt = _build_auto_detect_prompt(templates) + "\n\nОпредели template_id для этого документа."

    fallback = "universal" if "universal" in allowed_ids else allowed_ids[0]

    # Single page detection (first page is sufficient for classification)
    if not images:
        return fallback

    try:
        data = await call_vlm(
            image_bytes=images[0],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=_AUTO_DETECT_MAX_TOKENS,
        )
        raw = (
            data.get("template_id")
            or data.get("id")
            or data.get("document_type")
            or data.get("type")
            or data.get("тип_документа")
            or data.get("raw_text")
            or ""
        )
        detected = _normalize_detected_template_id(str(raw), templates)
        return detected if detected else fallback
    except Exception:
        return fallback



_IDLE_CHECK_INTERVAL = 3      # seconds between checks for new files
_IDLE_TIMEOUT = 300           # 5 minutes — auto-finish if no new files and upload not marked complete



async def _process_batch_ocr(batch_id: str, batch: dict):
    """Process a batch in OCR mode: create searchable PDFs.

    Key: saves each page result to DB immediately after OCR so that
    the SSE progress stream can deliver real-time preview updates.
    """
    from app.config import RESULTS_DIR, OCR_DPI
    from app.ocr_overlay import ocr_page
    import fitz as _fitz

    ocr_type = (batch.get("ocr_type") or "universal").strip()
    print(f"[BATCH_OCR] START batch={batch_id} ocr_type={ocr_type}", flush=True)

    try:
        files = batch.get("files", [])
        total_files = len(files)

        for idx, file_info in enumerate(files):
            filename = file_info["filename"]
            if file_info.get("status") == "done":
                continue

            pdf_path = UPLOADS_DIR / f"{batch_id}_{filename}"
            if not pdf_path.exists():
                storage.update_file(batch_id, filename, status="error", error="File not found")
                continue

            storage.mark_file_started(batch_id, filename)
            storage.update_file(batch_id, filename, status="processing", error=None)

            try:
                pdf_bytes = pdf_path.read_bytes()

                doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages = len(doc)

                if file_info.get("pages", 0) == 0:
                    storage.set_file_pages(batch_id, filename, num_pages)
                    storage.add_total_pages(batch_id, num_pages)

                t0 = time.perf_counter()
                mat = _fitz.Matrix(OCR_DPI / 72.0, OCR_DPI / 72.0)
                ocr_results: list[list[dict]] = []
                page_texts: list[str] = []

                # Process pages one-by-one, saving to DB immediately
                for page_idx in range(num_pages):
                    page = doc[page_idx]
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("png")

                    pt0 = time.perf_counter()
                    lines = await ocr_page(img_bytes, mode=ocr_type)
                    page_duration_ms = int((time.perf_counter() - pt0) * 1000)

                    ocr_results.append(lines)
                    page_text = "\n".join(l.get("text", "") for l in lines)
                    page_texts.append(page_text)

                    # Save to DB immediately — SSE will pick this up
                    storage.save_page_result(
                        batch_id, filename, page_idx + 1,
                        {"ocr_text": page_text, "lines": lines},
                        page_duration_ms=page_duration_ms,
                    )
                    storage.increment_pages_done(batch_id, filename, page_duration_ms=page_duration_ms)
                    print(f"[OCR] page {page_idx + 1}/{num_pages}: {len(lines)} lines, {page_duration_ms}ms", flush=True)

                doc.close()
                duration_ms = int((time.perf_counter() - t0) * 1000)

                # Searchable PDF is NOT built automatically anymore;
                # user triggers it via the overlay button (generate-searchable-pdf endpoint)

                # Extract tags from combined OCR text
                try:
                    full_text = "\n\n".join(page_texts)
                    if full_text.strip():
                        tags = await extract_tags(full_text)
                        storage.save_tags(batch_id, filename, tags)
                        print(f"[BATCH_OCR] tags extracted for {filename}: {len(tags.get('names', []))} names, {len(tags.get('dates', []))} dates, {len(tags.get('documents', []))} docs", flush=True)
                except Exception as te:
                    print(f"[BATCH_OCR] tags extraction failed for {filename}: {te}", flush=True)

                storage.update_file(batch_id, filename, status="done")
                storage.mark_file_finished(batch_id, filename)
                print(f"[BATCH_OCR] file={filename} pages={num_pages} time={duration_ms}ms", flush=True)

            except Exception as e:
                storage.update_file(batch_id, filename, status="error", error=str(e))
                storage.mark_file_finished(batch_id, filename)
                print(f"[BATCH_OCR] ERROR file={filename}: {e}", flush=True)
                import traceback
                traceback.print_exc()

        storage.update_batch(batch_id, status="done", finished_at=datetime.now().isoformat())
        print(f"[BATCH_OCR] DONE batch={batch_id}", flush=True)

    except asyncio.CancelledError:
        storage.mark_processing_files_interrupted(batch_id)
        storage.update_batch(batch_id, status="interrupted", finished_at=datetime.now().isoformat())
    except Exception as e:
        storage.update_batch(batch_id, status="error", error=str(e), finished_at=datetime.now().isoformat())
        import traceback
        traceback.print_exc()
    finally:
        _active_tasks.pop(batch_id, None)


async def process_batch(batch_id: str):
    """Process a batch with persistence. Continuously picks up new files as they are uploaded."""
    print(f"[BATCH] process_batch START batch={batch_id}", flush=True)
    batch = storage.get_batch(batch_id)
    if not batch:
        print(f"[BATCH] process_batch batch={batch_id} not found", flush=True)
        return

    # Sync counters from DB (important for resume)
    storage.sync_counters(batch_id)
    batch = storage.get_batch(batch_id)
    started_at = batch.get("started_at") if batch else None
    storage.update_batch(
        batch_id,
        status="processing",
        error=None,
        started_at=started_at or datetime.now().isoformat(),
        finished_at=None,
    )
    semaphore = asyncio.Semaphore(get_concurrency())

    custom_prompt = (batch.get("custom_prompt", "") or "").strip()
    template_id = (batch.get("template_id", "") or "").strip()
    page_range_spec = (batch.get("page_range", "") or "").strip()
    # Check for OCR mode
    batch_mode = (batch.get("mode") or "extract").strip()
    if batch_mode == "ocr":
        await _process_batch_ocr(batch_id, batch)
        return

    auto_mode = not template_id and not custom_prompt
    templates_for_auto = list_templates() if auto_mode else []

    # Filter templates_for_auto by allowed_templates if specified
    allowed_templates_raw = (batch.get("allowed_templates", "") or "").strip()
    if auto_mode and allowed_templates_raw:
        try:
            allowed_ids_list = json.loads(allowed_templates_raw)
            if isinstance(allowed_ids_list, list) and allowed_ids_list:
                allowed_set = set(allowed_ids_list)
                templates_for_auto = [t for t in templates_for_auto if t.get("id") in allowed_set]
                print(f"[BATCH] batch={batch_id} auto-mode restricted to {len(templates_for_auto)} templates: {sorted(allowed_set)}", flush=True)
        except (json.JSONDecodeError, TypeError):
            pass

    if auto_mode:
        system_prompt, user_prompt, max_tokens, postprocessing_steps = "", "", 0, []
    else:
        system_prompt, user_prompt, max_tokens, postprocessing_steps = resolve_prompts(
            template_id=template_id, custom_prompt=custom_prompt,
        )

    try:
        idle_since = None  # timestamp when we last had no work

        while True:
            # Get unfinished files from DB
            unfinished = storage.get_unfinished_files(batch_id)

            if unfinished:
                idle_since = None  # reset idle timer

                for i in range(0, len(unfinished), PDF_BATCH_SIZE):
                    group = unfinished[i:i + PDF_BATCH_SIZE]
                    await _process_file_group(
                        group, batch_id, semaphore,
                        system_prompt, user_prompt, max_tokens,
                        postprocessing_steps, template_id,
                        auto_mode=auto_mode,
                        templates_for_auto=templates_for_auto,
                        strict_schema=True,
                        custom_prompt=custom_prompt,
                        page_range_spec=page_range_spec,
                    )
                continue  # check for more files immediately

            # No unfinished files — check if we should finish or wait
            if storage.is_upload_complete(batch_id):
                break

            # Upload not complete yet — wait for more files
            if idle_since is None:
                idle_since = time.time()

            if time.time() - idle_since > _IDLE_TIMEOUT:
                print(f"[BATCH] batch={batch_id} idle timeout ({_IDLE_TIMEOUT}s), finishing", flush=True)
                break

            await asyncio.sleep(_IDLE_CHECK_INTERVAL)

        # Final status
        batch_after = storage.get_batch(batch_id)
        print(f"[BATCH] process_batch DONE batch={batch_id} pages={batch_after['pages_done']}/{batch_after['total_pages']}", flush=True)
        if batch_after["pages_done"] == 0 and batch_after["total_pages"] > 0:
            storage.update_batch(batch_id, status="error", error="No pages processed — check model status", finished_at=datetime.now().isoformat())
        else:
            storage.update_batch(batch_id, status="done", finished_at=datetime.now().isoformat())

    except asyncio.CancelledError:
        print(f"[BATCH] process_batch CANCELLED batch={batch_id}", flush=True)
        storage.mark_processing_files_interrupted(batch_id)
        storage.update_batch(batch_id, status="interrupted", finished_at=datetime.now().isoformat())
    except Exception as e:
        print(f"[BATCH] process_batch ERROR batch={batch_id}: {e}", flush=True)
        storage.update_batch(batch_id, status="error", error=str(e), finished_at=datetime.now().isoformat())
        import traceback
        traceback.print_exc()
    finally:
        _active_tasks.pop(batch_id, None)

async def _process_file_group(
    file_infos, batch_id, semaphore,
    system_prompt, user_prompt, max_tokens,
    postprocessing_steps, template_id,
    auto_mode=False,
    templates_for_auto=None,
    strict_schema: bool = True,
    custom_prompt: str = "",
    page_range_spec: str = "",
):
    """Process a group of PDF files: one VLM call per document (multi-image).

    All pages of each document are sent in a single request so the model
    sees full context. Adaptive DPI is used to fit all pages into the
    model's context window.
    """
    import fitz as _fitz

    all_tasks = []  # list of (filename, images, sp, up, mt, steps, tid, file_info_ref)
    converted_files = set()

    for file_info in file_infos:
        filename = file_info["filename"]
        pdf_path = UPLOADS_DIR / f"{batch_id}_{filename}"

        if not pdf_path.exists():
            storage.update_file(batch_id, filename, status="error", error="File not found")
            storage.mark_file_finished(batch_id, filename)
            continue

        storage.update_file(batch_id, filename, status="converting", error=None)

        # Count pages for adaptive DPI
        try:
            doc = _fitz.open(pdf_path)
            raw_num_pages = len(doc)
            doc.close()
        except Exception as e:
            storage.update_file(batch_id, filename, status="error", error=f"PDF open failed: {e}")
            storage.mark_file_finished(batch_id, filename)
            continue

        num_pages = min(raw_num_pages, MAX_DOCUMENT_PAGES)
        dpi = get_document_dpi(num_pages)

        try:
            images = pdf_to_images(pdf_path, dpi=dpi)
        except Exception as e:
            storage.update_file(batch_id, filename, status="error", error=f"PDF conversion failed: {e}")
            storage.mark_file_finished(batch_id, filename)
            continue

        if not images:
            storage.update_file(batch_id, filename, status="error", error="PDF has no pages")
            storage.mark_file_finished(batch_id, filename)
            continue

        # Truncate to MAX_DOCUMENT_PAGES
        if len(images) > MAX_DOCUMENT_PAGES:
            images = images[:MAX_DOCUMENT_PAGES]

        # Apply page range filter if specified
        selected_pages = parse_page_range(page_range_spec, len(images)) if page_range_spec else None
        if selected_pages is not None:
            images = [images[i - 1] for i in sorted(selected_pages) if 1 <= i <= len(images)]
            print(f"[PAGE_RANGE] batch={batch_id} file={filename} selected {len(images)}/{raw_num_pages} pages", flush=True)

        effective_count = len(images)
        if effective_count == 0:
            storage.update_file(batch_id, filename, status="error", error="No pages after filtering")
            storage.mark_file_finished(batch_id, filename)
            continue

        file_template_id = template_id
        file_system_prompt = system_prompt
        file_user_prompt = user_prompt
        file_max_tokens = max_tokens
        file_postprocessing_steps = postprocessing_steps

        if auto_mode:
            fn_template = _classify_by_filename(filename, {t["id"] for t in (templates_for_auto or [])})
            if fn_template:
                file_template_id = fn_template
                print(f"[FILENAME_CLASSIFY] batch={batch_id} file={filename} template={file_template_id}", flush=True)
            else:
                file_template_id = await _detect_template_id_for_file(images, templates_for_auto or [])
                if not file_template_id:
                    file_template_id = "universal"
                print(f"[AUTO_TEMPLATE] batch={batch_id} file={filename} template={file_template_id}", flush=True)
            file_system_prompt, file_user_prompt, file_max_tokens, file_postprocessing_steps = resolve_prompts(
                template_id=file_template_id,
                custom_prompt="",
            )

        # Check if already done (resume case)
        done_pages = storage.get_processed_pages(batch_id, filename)
        if 0 in done_pages:
            # Document already processed as whole (page=0)
            storage.update_file(batch_id, filename, status="done", error=None)
            storage.mark_file_finished(batch_id, filename)
            converted_files.add(filename)
            print(f"[RESUME] batch={batch_id} file={filename} already done (page=0)", flush=True)
            continue

        if file_info["pages"] == 0:
            storage.set_file_pages(batch_id, filename, effective_count)
            storage.add_total_pages(batch_id, effective_count)

        storage.update_file(batch_id, filename, status="queued", detected_template_id=file_template_id)
        converted_files.add(filename)

        all_tasks.append((
            filename, images, file_system_prompt, file_user_prompt,
            file_max_tokens, file_postprocessing_steps, file_template_id, file_info,
            effective_count, dpi,
        ))

    if not all_tasks:
        return

    # Phase 2: Process all documents concurrently (one VLM call per document)
    async def process_one_doc(fn, imgs, sp, up, mt, steps, tid, fi, num_pg, used_dpi):
        async with semaphore:
            storage.mark_file_started(batch_id, fn)
            storage.update_file(batch_id, fn, status="processing")
            started_at = time.perf_counter()
            result = await process_document(
                imgs,
                system_prompt=sp,
                user_prompt=up,
                max_tokens=mt,
                postprocessing_steps=steps,
                template_id=tid,
            )
            duration_ms = int((time.perf_counter() - started_at) * 1000)

            # Schema enforcement
            try:
                if strict_schema:
                    schema_tid = (tid or "").strip()
                    allowed = []
                    if schema_tid:
                        allowed = _get_allowed_keys_for_template(schema_tid)
                    else:
                        custom_keys = _extract_schema_keys_from_user_prompt(up)
                        if custom_keys:
                            allowed = custom_keys
                            schema_tid = "custom"
                        else:
                            schema_tid = "universal"
                            allowed = _get_allowed_keys_for_template(schema_tid)
                    if allowed:
                        result = _enforce_output_schema(
                            result,
                            allowed_keys=allowed,
                            batch_id=batch_id,
                            filename=fn,
                            page_num=0,
                            template_id=schema_tid,
                        )
            except Exception as e:
                print(f"[SCHEMA_ERROR] batch={batch_id} file={fn} err={e}", flush=True)

            result["filename"] = fn
            result["page"] = 0  # 0 = whole document

            # Save to storage
            storage.mark_document_done(batch_id, fn, num_pg, result, duration_ms=duration_ms)
            print(f"[DOC_DONE] batch={batch_id} file={fn} pages={num_pg} dpi={used_dpi} time={duration_ms}ms", flush=True)
            return result

    results = await asyncio.gather(
        *[
            process_one_doc(fn, imgs, sp, up, mt, steps, tid, fi, num_pg, used_dpi)
            for fn, imgs, sp, up, mt, steps, tid, fi, num_pg, used_dpi in all_tasks
        ],
        return_exceptions=True,
    )

    # Phase 3: Handle errors
    for task_info, r in zip(all_tasks, results):
        fn = task_info[0]
        if isinstance(r, Exception):
            storage.update_file(batch_id, fn, status="error", error=str(r))
            storage.mark_file_finished(batch_id, fn)
            import traceback
            traceback.print_exception(type(r), r, r.__traceback__)

