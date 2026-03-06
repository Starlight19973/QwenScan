# QwenScan

Система распознавания и извлечения данных из сканированных PDF-документов (бухгалтерские: УПД, счета-фактуры, акты, накладные, договоры и др.) с помощью VLM Qwen3-VL-8B.

## Архитектура

```
Браузер (index.html)
    │ HTTP / SSE
    ▼
FastAPI (main.py, порт 8081, uvicorn)
    ├── batch_processor.py  — пакетная обработка PDF
    ├── pipeline.py         — VLM-пайплайн: PDF→PNG→base64→VLM→JSON→постобработка
    ├── storage.py          — SQLite (qwenscan.db), персистентные пакеты
    ├── template_registry.py — шаблоны документов (templates.json + prompts/*.md)
    ├── vlm_client.py       — HTTP-клиент к vLLM (OpenAI-compatible)
    ├── model_manager.py    — запуск/остановка vLLM процесса
    ├── chat.py             — чат с VLM по загруженным документам
    ├── sync_api.py         — синхронный API (/api/attributes)
    ├── ocr_overlay.py      — OCR + searchable PDF (вкладка Поиск)
    ├── excel_export.py     — генерация Excel
    ├── postprocessors.py   — постобработка: ИНН/КПП, адреса, номера
    └── validators.py       — валидация ИНН/КПП по контрольным суммам
    │
    ▼
vLLM (порт 8001, Qwen3-VL-8B-Instruct-AWQ-8bit, RTX 3090)
```

Два раздельных процесса: FastAPI (веб) и vLLM (inference). Модель загружается/выгружается через API.

## Структура проекта

```
~/QwenScan/
├── app/
│   ├── main.py              # FastAPI routes (~1460 строк)
│   ├── batch_processor.py   # Пакетная обработка (~844 строк)
│   ├── storage.py           # SQLite ORM (~640 строк)
│   ├── template_registry.py # Шаблоны документов (~449 строк)
│   ├── pipeline.py          # VLM pipeline (~216 строк)
│   ├── vlm_client.py        # HTTP → vLLM (~242 строк)
│   ├── chat.py              # Чат с VLM (~205 строк)
│   ├── sync_api.py          # Синхронный API (~278 строк)
│   ├── ocr_overlay.py       # OCR overlay (~219 строк)
│   ├── excel_export.py      # Excel (~300 строк)
│   ├── postprocessors.py    # Постобработка (~223 строк)
│   ├── prompts.py           # Системные промпты (~214 строк)
│   ├── validators.py        # Валидация ИНН/КПП (~123 строк)
│   ├── model_manager.py     # Управление vLLM (~124 строк)
│   ├── pdf_utils.py         # PDF → PNG (~57 строк)
│   ├── config.py            # Конфигурация (~54 строк)
│   ├── templates/
│   │   └── index.html       # Веб-интерфейс SPA (~2924 строк)
│   └── templates.json       # Реестр шаблонов документов
├── prompts/                  # Промпты шаблонов (по папкам: {id}/system_prompt.md, user_prompt.md)
├── models/                   # Файлы модели Qwen3-VL-8B (~5 GB)
├── uploads/                  # Загруженные PDF (временные)
├── results/                  # Сгенерированные Excel
├── docs/                     # API_QWEN.md, README_QWEN.md
├── qwenscan.db               # SQLite база (пакеты, файлы, результаты)
├── start_vllm.sh             # Запуск vLLM
├── start_app.sh              # Запуск FastAPI
├── requirements.txt
└── venv/
```

## Ключевые конфигурации (config.py)

| Параметр | Значение | Описание |
|---|---|---|
| VLLM_BASE_URL | `http://localhost:8001/v1` | vLLM API |
| MODEL_NAME | `document-parser` | served-model-name |
| MAX_CONCURRENT_REQUESTS | 2 (по умолчанию, до 8 через API) | Параллельные VLM-запросы |
| PDF_DPI | 260 | Качество рендеринга PDF |
| EXTRACTION_MAX_TOKENS | 2048 | Макс. токенов ответа |
| REQUEST_TIMEOUT | 600 сек | Таймаут VLM |
| TEMPERATURE | 0.0 | Детерминированная генерация |
| MAX_MODEL_LEN | 32768 | Контекст vLLM |
| MAX_DOCUMENT_PAGES | 20 | Макс. страниц в одном PDF |

## Шаблоны документов

Шаблоны хранятся в `app/templates.json` (реестр) + `prompts/{id}/system_prompt.md` и `user_prompt.md` (промпты на диске).

Каждый шаблон содержит:
- `id` — slug (латиница)
- `name` — отображаемое имя
- `description` — описание
- `fields` — список полей для извлечения `[{key, label, type}]`
- `page_range` — диапазон страниц по умолчанию (формат: `1,3-5,last`)
- `max_tokens`, `postprocessing` — параметры обработки

Встроенные шаблоны: universal, schet_faktura, upd, akt, torg12, schet_na_oplatu, schet_dogovor, dogovor, transportnaya, deklaratsiya, platezhnoe, doverennost.

Пользовательские шаблоны создаются через UI вкладку «Промпты» (CRUD: POST/PUT/DELETE `/api/prompt-templates`).

## API — основные эндпоинты

### Шаблоны
- `GET /api/templates` — список шаблонов (id, name, description, fields, page_range)
- `GET /api/templates/{id}` — детали шаблона
- `GET /api/prompt-templates` — шаблоны + system_prompt (для редактора)
- `POST /api/prompt-templates` — создать шаблон (name, description, system_prompt, fields, page_range)
- `PUT /api/prompt-templates/{id}` — обновить шаблон
- `DELETE /api/prompt-templates/{id}` — удалить шаблон

### Пакетная обработка (chunked, основной)
- `POST /api/batch/create` — создать пакет (template_id, custom_prompt, page_range, allowed_template_ids)
- `POST /api/batch/{id}/upload` — загрузить файлы (multipart, порциями по 50)
- `POST /api/batch/{id}/start` — запустить обработку
- `POST /api/batch/{id}/stop` — остановить
- `POST /api/batch/{id}/resume` — возобновить
- `GET /api/batch/{id}/progress` — SSE-прогресс
- `GET /api/batch/{id}/results` — результаты JSON
- `GET /api/batch/{id}/results/processed` — промежуточные результаты (готовые файлы)
- `POST /api/batch/{id}/generate-excel` — сгенерировать Excel
- `GET /api/batch/{id}/excel` — скачать Excel
- `DELETE /api/batch/{id}` — удалить пакет
- `GET /api/batches` — список пакетов (до 50)

### Загрузка (legacy)
- `POST /api/upload` — загрузка + запуск одним запросом

### Синхронный API
- `POST /api/attributes` — извлечение из одного PDF (multipart: file, attributes JSON, template)

### Модель
- `GET /api/model/status` — статус (running/stopped/starting)
- `POST /api/model/load` — загрузить в GPU
- `POST /api/model/unload` — выгрузить

### Чат
- `POST /api/chat` — чат с VLM по документу (SSE)

### OCR
- `POST /api/ocr/upload` — загрузить PDF для OCR
- `GET /api/ocr/{id}/status` — статус OCR
- `GET /api/ocr/{id}/download` — скачать searchable PDF

### Утилиты
- `GET /api/concurrency` — текущий параллелизм
- `POST /api/concurrency` — установить (1-8)
- `GET /api/gpu` — GPU метрики (VRAM, utilization, temp)

## База данных (SQLite)

Файл: `qwenscan.db`. Три основные таблицы:
- **batches** — пакеты (id, status, template_id, custom_prompt, page_range, created_at, started_at, finished_at, ...)
- **batch_files** — файлы в пакетах (batch_id, filename, status, pages, pages_done, ...)
- **page_results** — результаты по страницам (batch_id, filename, page_num, result JSON, duration_ms, ...)

Управляется через `storage.py`. Все операции crash-safe (результаты сохраняются постранично).

## Веб-интерфейс (index.html)

SPA, 4 вкладки: Сканирование | Промпты | Чат | Поиск (OCR).

Вкладка **Сканирование**:
- Drag-drop зона для PDF (файлы + папки рекурсивно)
- При >50 файлах — свёрнутый список (сводка + кнопка очистки)
- Прогресс сканирования папок («Сканирование... найдено N файлов»)
- Выбор шаблона (select) + пользовательский промпт (textarea)
- Поле page_range (страницы: `1,3-5,last`)
- Chunked-загрузка: `UPLOAD_CHUNK_SIZE=50` файлов за запрос
- SSE прогресс-бар
- Таблица результатов + экспорт Excel

Вкладка **Промпты**:
- Список шаблонов (select)
- Редактор: название, описание, системный промпт (textarea), страницы (page_range)
- Создание / обновление / удаление шаблонов

## Запуск

```bash
# vLLM (inference, GPU)
cd ~/QwenScan && nohup bash start_vllm.sh > vllm.log 2>&1 &

# FastAPI (веб)
cd ~/QwenScan && nohup venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8081 >> /tmp/qwenscan.log 2>&1 &
```

URL: `https://31.173.93.121/qwenscan/` (через nginx reverse proxy, root_path=/qwenscan)

## Важные детали

- `root_path="/qwenscan"` — все API-пути проксируются nginx через `/qwenscan/`
- JS-функция `apiUrl(path)` в index.html строит URL относительно root_path
- Шаблоны загружаются при старте из `templates.json` + файлов промптов. Кеш сбрасывается через `invalidate_cache()`
- Continuation-оптимизация: для стр. 2+ известных шаблонов (upd, akt, torg12...) используется облегчённый промпт (только суммы)
- Постобработка: `split_inn_kpp`, `validate_inn`, `validate_kpp`, `clean_address`, `fix_doc_number` — только для шаблонного режима
- Chunked-загрузка: create → upload (×N) → start. Upload chunk size = 50 файлов
- `page_range` хранится per-batch (в БД) И per-template (в templates.json). При выборе шаблона page_range автозаполняется в UI
- Стороны документа: `контрагент1` = продавец/поставщик, `контрагент2` = покупатель/заказчик
