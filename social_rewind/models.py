from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionNote:
    """Agent 主动提供的短说明；不是隐藏推理链。"""

    intent: str
    known_facts: tuple[str, ...]
    uncertainty: str

    def validate(self) -> None:
        if not self.intent.strip():
            raise ValueError("decision.intent 不能为空")
        if not self.known_facts or any(not fact.strip() for fact in self.known_facts):
            raise ValueError("decision.known_facts 至少包含一条非空事实")
        if not self.uncertainty.strip():
            raise ValueError("decision.uncertainty 不能为空")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["known_facts"] = list(self.known_facts)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DecisionNote:
        note = cls(
            intent=str(value["intent"]),
            known_facts=tuple(str(item) for item in value["known_facts"]),
            uncertainty=str(value["uncertainty"]),
        )
        note.validate()
        return note


@dataclass(frozen=True)
class Action:
    """统一行动信封；kind 决定 payload 的最小契约。"""

    kind: str
    actor: str
    payload: dict[str, Any]
    decision: DecisionNote

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.actor.strip():
            raise ValueError("action.kind 与 action.actor 不能为空")
        self.decision.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "actor": self.actor,
            "payload": self.payload,
            "decision": self.decision.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Action:
        return cls(
            kind=str(value["kind"]),
            actor=str(value["actor"]),
            payload=dict(value["payload"]),
            decision=DecisionNote.from_dict(dict(value["decision"])),
        )


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    fictional: bool
    public_facts: tuple[str, ...]
    roles: dict[str, dict[str, Any]]
    private_facts: dict[str, tuple[str, ...]]
    project_budget: int
    tasks: dict[str, dict[str, Any]]
    max_daily_interactions: int


@dataclass(frozen=True)
class TimelineMemory:
    """刻意放在 World 外部的跨时间线经验。"""

    owner: str
    lesson: str
