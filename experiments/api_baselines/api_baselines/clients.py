from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class GenerationResult:
    text: str
    raw: dict[str, Any]
    latency_sec: float
    usage: dict[str, Any]


class APIClientError(RuntimeError):
    pass


def image_to_base64(path: str | Path) -> tuple[str, str]:
    p = Path(path)
    data = p.read_bytes()
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return mime, base64.b64encode(data).decode("ascii")


def require_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise APIClientError(f"Missing required environment variable: {key}")
    return value


def post_json(name: str, url: str, **kwargs: Any) -> dict[str, Any]:
    """Keep endpoint addresses, credentials, and server error bodies out of logs."""
    try:
        response = requests.post(url, **kwargs)
    except requests.RequestException as exc:
        raise APIClientError(f"{name}: {type(exc).__name__}") from None
    if response.status_code >= 400:
        raise APIClientError(f"{name}: HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise APIClientError(f"{name}: invalid JSON response") from None
    if not isinstance(data, dict):
        raise APIClientError(f"{name}: expected a JSON object")
    return data


def parse_openai_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return ""


class OpenAICompatibleChatClient:
    """Client for OpenAI Chat Completions compatible multimodal APIs."""

    def __init__(self, config: dict[str, Any]):
        self.name = config["name"]
        self.model = config["model"]
        self.api_key = require_env(config["api_key_env"])
        self.base_url = require_env(config.get("base_url_env", "COMPAT_BASE_URL"))
        self.timeout_sec = int(config.get("timeout_sec", 120))
        self.extra_headers = dict(config.get("extra_headers", {}))
        self.extra_payload = dict(config.get("extra_payload", {}))

    def generate(self, prompt: str, image_paths: list[str], generation: dict[str, Any]) -> GenerationResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths[: int(generation.get("max_images", 8))]:
            if not Path(image_path).exists():
                continue
            mime, b64 = image_to_base64(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": generation.get("temperature", 0),
            "max_tokens": generation.get("max_tokens", 256),
        }
        payload.update(self.config_extra_payload())

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        started = time.time()
        data = post_json(self.name, url, headers=headers, json=payload, timeout=self.timeout_sec)
        latency = time.time() - started
        return GenerationResult(
            text=parse_openai_chat_text(data),
            raw=data,
            latency_sec=latency,
            usage=data.get("usage") or {},
        )

    def config_extra_payload(self) -> dict[str, Any]:
        return dict(self.extra_payload)


class AnthropicClient:
    def __init__(self, config: dict[str, Any]):
        self.name = config["name"]
        self.model = config["model"]
        self.api_key = require_env(config["api_key_env"])
        self.base_url = require_env(config.get("base_url_env", "ANTHROPIC_BASE_URL"))
        self.timeout_sec = int(config.get("timeout_sec", 120))

    def generate(self, prompt: str, image_paths: list[str], generation: dict[str, Any]) -> GenerationResult:
        content: list[dict[str, Any]] = []
        for image_path in image_paths[: int(generation.get("max_images", 8))]:
            if not Path(image_path).exists():
                continue
            mime, b64 = image_to_base64(image_path)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "max_tokens": generation.get("max_tokens", 256),
            "temperature": generation.get("temperature", 0),
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        started = time.time()
        data = post_json(
            self.name,
            self.base_url.rstrip("/") + "/messages",
            headers=headers,
            json=payload,
            timeout=self.timeout_sec,
        )
        latency = time.time() - started
        parts = []
        for item in data.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return GenerationResult(
            text="\n".join(p for p in parts if p).strip(),
            raw=data,
            latency_sec=latency,
            usage=data.get("usage") or {},
        )


class GeminiClient:
    def __init__(self, config: dict[str, Any]):
        self.name = config["name"]
        self.model = config["model"]
        self.api_key = require_env(config["api_key_env"])
        self.base_url = require_env(config.get("base_url_env", "GEMINI_BASE_URL"))
        self.timeout_sec = int(config.get("timeout_sec", 120))

    def generate(self, prompt: str, image_paths: list[str], generation: dict[str, Any]) -> GenerationResult:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for image_path in image_paths[: int(generation.get("max_images", 8))]:
            if not Path(image_path).exists():
                continue
            mime, b64 = image_to_base64(image_path)
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": generation.get("temperature", 0),
                "maxOutputTokens": generation.get("max_tokens", 256),
            },
        }
        url = f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"

        started = time.time()
        data = post_json(
            self.name, url, headers={"x-goog-api-key": self.api_key},
            json=payload, timeout=self.timeout_sec,
        )
        latency = time.time() - started
        text_parts = []
        for cand in data.get("candidates", []):
            for part in (cand.get("content") or {}).get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])
            if text_parts:
                break
        return GenerationResult(
            text="\n".join(text_parts).strip(),
            raw=data,
            latency_sec=latency,
            usage=data.get("usageMetadata") or {},
        )


class DashScopeMultimodalClient:
    """Client for DashScope native multimodal-generation API."""

    def __init__(self, config: dict[str, Any]):
        self.name = config["name"]
        self.model = config["model"]
        self.api_key = require_env(config["api_key_env"])
        self.endpoint = require_env(config.get("endpoint_env", "DASHSCOPE_ENDPOINT"))
        self.timeout_sec = int(config.get("timeout_sec", 180))

    def generate(self, prompt: str, image_paths: list[str], generation: dict[str, Any]) -> GenerationResult:
        content: list[dict[str, str]] = []
        for image_path in image_paths[: int(generation.get("max_images", 8))]:
            if not Path(image_path).exists():
                continue
            mime, b64 = image_to_base64(image_path)
            content.append({"image": f"data:{mime};base64,{b64}"})
        content.append({"text": prompt})

        payload = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {
                "temperature": generation.get("temperature", 0),
                "max_tokens": generation.get("max_tokens", 256),
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        started = time.time()
        data = post_json(self.name, self.endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
        latency = time.time() - started
        text_parts = []
        choices = (data.get("output") or {}).get("choices") or []
        for choice in choices:
            content_items = ((choice.get("message") or {}).get("content")) or []
            if isinstance(content_items, str):
                text_parts.append(content_items)
            else:
                for item in content_items:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
            if text_parts:
                break
        usage = data.get("usage") or {}
        return GenerationResult(
            text="\n".join(text_parts).strip(),
            raw=data,
            latency_sec=latency,
            usage=usage,
        )


def build_client(config: dict[str, Any]):
    provider = config["provider"]
    if provider == "openai_compatible_chat":
        return OpenAICompatibleChatClient(config)
    if provider == "anthropic":
        return AnthropicClient(config)
    if provider == "gemini":
        return GeminiClient(config)
    if provider == "dashscope_multimodal":
        return DashScopeMultimodalClient(config)
    raise APIClientError(f"Unsupported provider: {provider}")
