from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Database
DB_PATH = BASE_DIR / "qwenscan.db"

# vLLM
VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_CHAT_URL = f"{VLLM_BASE_URL}/chat/completions"
MODEL_NAME = "document-parser"

# Processing
MAX_CONCURRENT_REQUESTS = 2  # default, overridden at runtime via API
_runtime_concurrency: int = MAX_CONCURRENT_REQUESTS


def get_concurrency() -> int:
    return _runtime_concurrency


def set_concurrency(value: int) -> int:
    global _runtime_concurrency
    _runtime_concurrency = max(1, min(value, 8))
    return _runtime_concurrency
PDF_DPI = 260
MAX_DOCUMENT_PAGES = 20

# vLLM context
MAX_MODEL_LEN = 32768

# Generation
EXTRACTION_MAX_TOKENS = 6144
REQUEST_TIMEOUT = 600.0
TEMPERATURE = 0.0

# Chat
CHAT_MAX_TOKENS = 4096
CHAT_TEMPERATURE = 0.3
CHAT_MAX_HISTORY = 20
CHAT_MAX_IMAGES_PER_MESSAGE = 5

# OCR / Searchable PDF
OCR_DPI = 200
OCR_MAX_TOKENS = 4096
OCR_REQUEST_TIMEOUT = 300.0

# Tags extraction
TAGS_MAX_TOKENS = 4096

# Models registry
MODELS = {
    "8b": {
        "id": "8b",
        "name": "Qwen3-VL-8B",
        "description": "Быстрая модель для стандартных документов",
        "script": "start_vllm.sh",
        "vram_gb": 10,
    },
    "27b": {
        "id": "27b",
        "name": "Qwen3.5-27B (качество)",
        "description": "Лучшее качество: рукопись, выцветшие, сложные документы",
        "script": "start_vllm_35b.sh",
        "vram_gb": 20,
    },
}
ACTIVE_MODEL_ID = "8b"  # default
