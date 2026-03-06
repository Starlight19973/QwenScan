"""HTTP-клиент для vLLM (OpenAI-совместимый API)."""

import base64
import json
import re
from collections.abc import AsyncGenerator

import httpx

from app.config import (
    VLLM_CHAT_URL, MODEL_NAME, REQUEST_TIMEOUT,
    EXTRACTION_MAX_TOKENS, TEMPERATURE,
    CHAT_MAX_TOKENS, CHAT_TEMPERATURE,
)

# Persistent HTTP client — reuses connections to vLLM
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=300,
            ),
        )
    return _client


async def call_vlm(
    image_bytes: bytes,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = EXTRACTION_MAX_TOKENS,
) -> dict:
    """Отправить изображение + промпт в vLLM, вернуть распарсенный JSON."""

    img_b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    client = _get_client()
    response = await client.post(VLLM_CHAT_URL, json=payload)
    result = response.json()

    if "error" in result:
        return {"error": result["error"], "parse_error": True}
    if "choices" not in result:
        return {"error": f"Unexpected response: {str(result)[:200]}", "parse_error": True}

    msg = result["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content", "")
    return parse_vlm_json(content)




async def call_vlm_document(
    images: list[bytes],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = EXTRACTION_MAX_TOKENS,
) -> dict:
    """Отправить ВСЕ страницы документа в одном запросе к vLLM."""

    content_parts = []
    for img_bytes in images:
        img_b64 = base64.b64encode(img_bytes).decode()
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
    content_parts.append({"type": "text", "text": user_prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    client = _get_client()
    response = await client.post(VLLM_CHAT_URL, json=payload)
    result = response.json()

    if "error" in result:
        return {"error": result["error"], "parse_error": True}
    if "choices" not in result:
        return {"error": f"Unexpected response: {str(result)[:200]}", "parse_error": True}

    msg = result["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content", "")
    return parse_vlm_json(content)

def parse_vlm_json(text: str) -> dict:
    """Извлечение JSON из ответа VLM с обработкой частых ошибок."""
    text = text.strip()

    # Убрать <think>...</think> теги Qwen3
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Убрать markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Попытка исправить trailing commas
        text = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_text": text, "parse_error": True}


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from streaming text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


async def call_vlm_chat_stream(
    messages: list[dict],
    max_tokens: int = CHAT_MAX_TOKENS,
    temperature: float = CHAT_TEMPERATURE,
) -> AsyncGenerator[str, None]:
    """Stream chat completion from vLLM. Yields text chunks."""

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    client = _get_client()
    # Buffer for incomplete think tags
    think_buffer = ""
    in_think = False

    async with client.stream("POST", VLLM_CHAT_URL, json=payload) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            text = delta.get("content", "")
            if not text:
                continue

            # Strip <think>...</think> tags
            think_buffer += text
            # Process complete think blocks
            think_buffer = re.sub(
                r"<think>.*?</think>", "", think_buffer, flags=re.DOTALL
            )
            # Check for incomplete opening tag
            if "<think>" in think_buffer and "</think>" not in think_buffer:
                in_think = True
                # Output everything before <think>
                idx = think_buffer.index("<think>")
                if idx > 0:
                    yield think_buffer[:idx]
                think_buffer = think_buffer[idx:]
                continue
            if in_think:
                if "</think>" in think_buffer:
                    # Think block complete — strip it
                    think_buffer = re.sub(
                        r"<think>.*?</think>", "", think_buffer, flags=re.DOTALL
                    )
                    in_think = False
                else:
                    continue

            if think_buffer:
                yield think_buffer
                think_buffer = ""


async def call_vlm_chat(
    messages: list[dict],
    max_tokens: int = CHAT_MAX_TOKENS,
    temperature: float = CHAT_TEMPERATURE,
) -> str:
    """Non-streaming chat completion (fallback)."""

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    client = _get_client()
    response = await client.post(VLLM_CHAT_URL, json=payload)
    result = response.json()

    if "error" in result:
        return f"[Ошибка VLM: {result['error']}]"
    if "choices" not in result:
        return f"[Неожиданный ответ: {str(result)[:200]}]"

    msg = result["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content", "")
    # Strip think tags
    content = _strip_think_tags(content).strip()
    return content
