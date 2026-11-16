from __future__ import annotations

import copy
import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class ModelCallError(RuntimeError):
    def __init__(self, message: str, call_records: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.call_records = call_records or []


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_base: str
    api_key: str
    model: str
    max_tokens: int = 800
    temperature: float = 0.3
    timeout_seconds: int = 90
    extra_body: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.api_base.strip() or not self.api_key.strip() or not self.model.strip():
            raise ValueError("api_base、api_key 与 model 均不能为空")
        if self.max_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("max_tokens 与 timeout_seconds 必须为正数")


class OpenAICompatibleClient:
    """仅实现谈判演示需要的 chat/completions 最小适配器。"""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        config.validate()
        self._config = config

    def complete(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        request_body: dict[str, Any] = {
            "model": self._config.model,
            "messages": copy.deepcopy(messages),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        request_body.update(copy.deepcopy(self._config.extra_body))
        encoded = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        endpoint = self._config.api_base.rstrip("/") + "/chat/completions"
        failed_record = {"provider": "openai-compatible", "endpoint": endpoint,
                         "request": copy.deepcopy(request_body)}
        request = urllib.request.Request(
            endpoint,
            data=encoded,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            failed_record["error"] = {"http_status": exc.code, "detail": detail}
            raise ModelCallError(f"模型 API HTTP {exc.code}: {detail}", [failed_record]) from exc
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError) as exc:
            failed_record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, http.client.IncompleteRead):
                failed_record["error"]["partial_response"] = exc.partial.decode("utf-8", errors="replace")
            raise ModelCallError(f"模型 API 调用失败: {exc}", [failed_record]) from exc
        try:
            payload = json.loads(raw_response)
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            failed_record["error"] = {"message": "响应不符合 chat/completions 契约", "raw_response": raw_response}
            raise ModelCallError("模型 API 响应不符合 chat/completions 契约", [failed_record]) from exc
        if not isinstance(content, str) or not content.strip():
            failed_record["error"] = {"message": "模型未返回最终内容", "raw_response": payload}
            raise ModelCallError(
                f"模型未返回最终内容，finish_reason={choice.get('finish_reason')}", [failed_record]
            )
        record = {
            "provider": "openai-compatible",
            "endpoint": endpoint,
            "request": request_body,
            "response": {
                "id": payload.get("id"),
                "model": payload.get("model", self._config.model),
                "created": payload.get("created"),
                "finish_reason": choice.get("finish_reason"),
                "content": content,
                "usage": payload.get("usage"),
            },
        }
        return content, record
