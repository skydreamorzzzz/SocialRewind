from __future__ import annotations

from typing import Any, Protocol


QUALITY_LEVELS = ("poor", "acceptable", "good", "excellent")


class Judge(Protocol):
    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]: ...


class JudgeOutputError(ValueError):
    pass


class MockJudge:
    """离线确定性 Judge；测试与演示不会联网。"""

    def __init__(self, quality: str = "good") -> None:
        if quality not in QUALITY_LEVELS:
            raise ValueError(f"未知质量档位: {quality}")
        self.quality = quality
        self.calls = 0

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        evidence_ids = [item["id"] for item in request["evidence"]]
        return {
            "effective_hours": request["claimed_hours"],
            "quality": self.quality,
            "reason": "模拟 Judge：证据完整，按固定档位判定。",
            "evidence_ids": evidence_ids,
            "model_metadata": {
                "provider": "mock",
                "model": "deterministic-v1",
                "call_id": f"mock-{request['submission_id']}",
            },
        }


def validate_verdict(request: dict[str, Any], verdict: object) -> dict[str, Any]:
    if not isinstance(verdict, dict):
        raise JudgeOutputError("Judge 输出必须是对象")
    expected = {"effective_hours", "quality", "reason", "evidence_ids", "model_metadata"}
    if set(verdict) != expected:
        raise JudgeOutputError("Judge 输出字段不符合受限契约")

    hours = verdict["effective_hours"]
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        raise JudgeOutputError("effective_hours 必须是数字")
    if hours < 0 or hours > request["claimed_hours"]:
        raise JudgeOutputError("effective_hours 超出已提交工作量")
    if verdict["quality"] not in QUALITY_LEVELS:
        raise JudgeOutputError("quality 档位无效")
    if not isinstance(verdict["reason"], str) or not verdict["reason"].strip():
        raise JudgeOutputError("reason 不能为空")

    evidence_ids = verdict["evidence_ids"]
    allowed_ids = {item["id"] for item in request["evidence"]}
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or any(not isinstance(item, str) for item in evidence_ids)
        or not set(evidence_ids).issubset(allowed_ids)
    ):
        raise JudgeOutputError("evidence_ids 必须引用本次输入证据")
    metadata = verdict["model_metadata"]
    if not isinstance(metadata, dict) or not all(
        isinstance(metadata.get(key), str) and metadata[key].strip()
        for key in ("provider", "model", "call_id")
    ):
        raise JudgeOutputError("model_metadata 缺少 provider/model/call_id")
    return {
        "effective_hours": hours,
        "quality": verdict["quality"],
        "reason": verdict["reason"],
        "evidence_ids": list(evidence_ids),
        "model_metadata": dict(metadata),
    }
