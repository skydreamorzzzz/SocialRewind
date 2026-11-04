from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .llm import OpenAICompatibleClient
from .models import Action, DecisionNote
from .trade import AUDITOR, ROLES, TradeMemory, TradeWorld


class AgentOutputError(RuntimeError):
    def __init__(self, message: str, call_records: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.call_records = call_records


class IllegalAgentAction(RuntimeError):
    def __init__(
        self,
        message: str,
        call_records: list[dict[str, Any]],
        receipts: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.call_records = call_records
        self.receipts = receipts


@dataclass(frozen=True)
class AgentTurn:
    kind: str
    speech: str
    intent: str
    known_facts: tuple[str, ...]
    uncertainty: str
    price: int | None = None
    offer_id: str | None = None
    evidence_event_ids: tuple[int, ...] = ()

    def to_action(self, role: str) -> Action:
        if self.kind == "offer":
            payload: dict[str, Any] = {"text": self.speech, "price": self.price}
        elif self.kind in ("accept", "reject"):
            payload = {"text": self.speech, "offer_id": self.offer_id}
        elif self.kind in ("message", "exit"):
            payload = {"text": self.speech}
        else:
            raise ValueError(f"未知 Agent 行动: {self.kind}")
        return Action(
            kind=self.kind,
            actor=role,
            payload=payload,
            decision=DecisionNote(
                intent=self.intent,
                known_facts=self.known_facts,
                uncertainty=self.uncertainty,
                evidence_event_ids=self.evidence_event_ids,
            ),
        )


@dataclass
class AgentReply:
    turn: AgentTurn
    call_records: list[dict[str, Any]] = field(default_factory=list)


class NegotiationAgent(Protocol):
    def start(self, role: str, context: dict[str, Any], memory: TradeMemory | None) -> None: ...

    def respond(self, observation: dict[str, Any]) -> AgentReply: ...


def render_rebirth_prompt(memory: TradeMemory, counterpart_role: str) -> str:
    event_lines: list[str] = []
    for event in memory.visible_events:
        public_input = json.dumps(event["input"], ensure_ascii=False, sort_keys=True)
        line = (
            f"- 事件 {event['event_id']}：{event['actor']} 执行 {event['kind']}，"
            f"公开内容={public_input}，结果={event['result_summary']}"
        )
        if "decision" in event:
            line += f"；你当时的显式决策说明={event['decision']}"
        event_lines.append(line)
    history = "\n".join(event_lines) if event_lines else "- 你没有选择保留具体事件片段。"
    return f"""【轮回开始：第二世】

上一世，你曾作为 {memory.role} 参与同一场交易。
你经历了完整的谈判，做出过报价与选择，也亲眼见证了对方的公开回应和最终结果。
然而，时间倒流了。商品、规则、身份和初始条件都恢复了原状，唯独你的记忆没有消失。

更重要的是，这是双方共同知道的事实：{counterpart_role} 也重生了，也记得其上一世亲眼见过的谈判，并可能改变策略。但你并不知道对方未公开的私人估值或私人决策。

你获准保留的上一世经历如下：
{history}

上一世结果：{json.dumps(memory.outcome, ensure_ascii=False, sort_keys=True)}
你自己的私人事实：{memory.own_private_fact}

这一世，你可以延续旧决定、吸取教训、试探对方或采用不同策略；不要求必须改变，也不要求必须成交。不要把猜测说成已验证事实。
"""


class ScriptedAgent:
    def __init__(self, turns: list[AgentTurn]) -> None:
        self._initial_turns = list(turns)
        self._turns: list[AgentTurn] = []
        self.role: str | None = None
        self.memory: TradeMemory | None = None

    def start(self, role: str, context: dict[str, Any], memory: TradeMemory | None) -> None:
        self.role = role
        self.memory = memory
        self._turns = list(self._initial_turns)

    def respond(self, observation: dict[str, Any]) -> AgentReply:
        if not self._turns:
            raise AgentOutputError("脚本 Agent 没有剩余行动", [])
        return AgentReply(turn=self._turns.pop(0))


class LLMNegotiationAgent:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        max_format_repairs: int = 1,
    ) -> None:
        if max_format_repairs < 0 or max_format_repairs > 2:
            raise ValueError("max_format_repairs 必须在 0 到 2 之间")
        self._client = client
        self._max_format_repairs = max_format_repairs
        self._role: str | None = None
        self._messages: list[dict[str, str]] = []

    def start(self, role: str, context: dict[str, Any], memory: TradeMemory | None) -> None:
        if role not in ROLES:
            raise ValueError("未知谈判角色")
        if memory is not None and memory.role != role:
            raise PermissionError("不能向角色注入其他人的轮回记忆")
        counterpart = "seller" if role == "buyer" else "buyer"
        private_rule = (
            f"你的私人最高愿付价是 {context['private']['maximum_price']} 元，正式报价或接受不得超过它。"
            if role == "buyer"
            else f"你的私人最低接受价是 {context['private']['minimum_price']} 元，正式报价或接受不得低于它。"
        )
        rebirth = render_rebirth_prompt(memory, counterpart) if memory is not None else (
            "【第一世】你没有前世记忆。这是一场新的谈判。"
        )
        schema = """每次只输出一个 JSON 对象，禁止 Markdown：
{
  "decision": {
    "intent": "简短第一人称意图",
    "known_facts": ["一条或多条事实或自述"],
    "uncertainty": "简短不确定性",
    "evidence_event_ids": [可选的当前观察事件编号]
  },
  "speech": "向对方公开说的话",
  "action": {"kind": "message|offer|accept|reject|exit", "price": 仅 offer 使用的整数, "offer_id": 仅 accept/reject 使用}
}
自然语言中的暗示不会成交；只有 accept 当前对方有效 offer_id 才会成交。不得虚构不存在的 offer_id。"""
        system = f"""你是虚构二手交易中的 {role}，对方是 {counterpart}。请自主、诚实地进行价格谈判，可试探、拒绝、退出或成交，但不要暴露隐藏推理链。

公开商品：{json.dumps(context['product'], ensure_ascii=False, sort_keys=True)}
公开挂牌价：{context['listed_price']} 元；最多 {context['max_rounds']} 轮。
{private_rule}

{rebirth}

{schema}
请有效率地谈判，不要为了拖延而重复同一句话。"""
        self._role = role
        self._messages = [{"role": "system", "content": system}]

    def respond(self, observation: dict[str, Any]) -> AgentReply:
        if self._role is None:
            raise RuntimeError("Agent 尚未 start")
        active_offer = observation.get("active_offer")
        if active_offer is None:
            action_guidance = "当前没有有效报价；accept/reject 均不可用。"
        elif active_offer["actor"] == self._role:
            action_guidance = (
                f"当前有效报价 {active_offer['id']} 是你自己的，不能 accept/reject。"
                "若想采用对方先前说过但已失效的价格，必须由你重新执行 offer。"
            )
        else:
            action_guidance = (
                f"你只能 accept/reject 当前对方有效报价 {active_offer['id']}；"
                "其他旧 offer_id 已失效。"
            )
        user_message = (
            "轮到你行动。自然语言中的价格不等于正式报价，也不会复活旧报价。"
            f"{action_guidance}\n以下是你当前有权观察的信息：\n"
            + json.dumps(observation, ensure_ascii=False, sort_keys=True)
        )
        self._messages.append({"role": "user", "content": user_message})
        call_records: list[dict[str, Any]] = []
        last_error = ""
        for attempt in range(self._max_format_repairs + 1):
            raw, record = self._client.complete(self._messages)
            record["role"] = self._role
            record["format_attempt"] = attempt
            call_records.append(record)
            self._messages.append({"role": "assistant", "content": raw})
            try:
                turn = self._parse_turn(raw)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                record["validation_error"] = last_error
                if attempt >= self._max_format_repairs:
                    raise AgentOutputError(
                        f"模型输出在有限修复后仍无效: {last_error}", call_records
                    ) from exc
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"上一条输出不符合 JSON 行动契约：{last_error}。"
                            "只重新输出修正后的 JSON 对象，不要解释。"
                        ),
                    }
                )
                continue
            return AgentReply(turn=turn, call_records=call_records)
        raise AgentOutputError(f"模型输出无效: {last_error}", call_records)

    @staticmethod
    def _parse_turn(raw: str) -> AgentTurn:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"decision", "speech", "action"}:
            raise ValueError("顶层字段必须严格为 decision/speech/action")
        decision = value["decision"]
        action = value["action"]
        speech = value["speech"]
        if not isinstance(decision, dict) or not isinstance(action, dict):
            raise ValueError("decision 与 action 必须是对象")
        if not isinstance(speech, str) or not speech.strip():
            raise ValueError("speech 必须是非空字符串")
        required_decision = {"intent", "known_facts", "uncertainty"}
        if not required_decision.issubset(decision) or not set(decision).issubset(
            required_decision | {"evidence_event_ids"}
        ):
            raise ValueError("decision 字段无效")
        known_facts = decision["known_facts"]
        evidence_ids = decision.get("evidence_event_ids", [])
        if not isinstance(known_facts, list) or not all(
            isinstance(item, str) and item.strip() for item in known_facts
        ):
            raise ValueError("known_facts 必须是非空字符串数组")
        if not known_facts:
            raise ValueError("known_facts 至少包含一条自述")
        if not isinstance(evidence_ids, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in evidence_ids
        ):
            raise ValueError("evidence_event_ids 必须是整数数组")
        if not isinstance(decision["intent"], str) or not decision["intent"].strip():
            raise ValueError("decision.intent 必须是非空字符串")
        if not isinstance(decision["uncertainty"], str) or not decision["uncertainty"].strip():
            raise ValueError("decision.uncertainty 必须是非空字符串")
        kind = action.get("kind")
        if kind in ("message", "exit"):
            if set(action) != {"kind"}:
                raise ValueError(f"{kind} 不能包含额外行动字段")
            price = None
            offer_id = None
        elif kind == "offer":
            if set(action) != {"kind", "price"}:
                raise ValueError("offer 必须且只能包含整数 price")
            price = action["price"]
            if isinstance(price, bool) or not isinstance(price, int):
                raise ValueError("offer.price 必须是整数")
            offer_id = None
        elif kind in ("accept", "reject"):
            if set(action) != {"kind", "offer_id"} or not isinstance(
                action["offer_id"], str
            ):
                raise ValueError(f"{kind} 必须且只能包含字符串 offer_id")
            price = None
            offer_id = action["offer_id"]
        else:
            raise ValueError("action.kind 无效")
        return AgentTurn(
            kind=kind,
            speech=speech,
            intent=decision["intent"],
            known_facts=tuple(known_facts),
            uncertainty=decision["uncertainty"],
            price=price,
            offer_id=offer_id,
            evidence_event_ids=tuple(evidence_ids),
        )


@dataclass(frozen=True)
class NegotiationRun:
    timeline_id: int
    metrics: dict[str, Any]
    model_calls: tuple[dict[str, Any], ...]
    receipts: tuple[dict[str, Any], ...]


class NegotiationRunner:
    def run(
        self,
        world: TradeWorld,
        buyer: NegotiationAgent,
        seller: NegotiationAgent,
        *,
        memories: dict[str, TradeMemory] | None = None,
        on_turn: Callable[[str, AgentTurn, dict[str, Any]], None] | None = None,
    ) -> NegotiationRun:
        memory_map = memories or {}
        if set(memory_map) - set(ROLES):
            raise ValueError("记忆映射包含未知角色")
        agents: dict[str, NegotiationAgent] = {"buyer": buyer, "seller": seller}
        for role, agent in agents.items():
            memory = memory_map.get(role)
            if memory is not None and memory.role != role:
                raise PermissionError("记忆所有者与注入角色不一致")
            agent.start(role, world.scenario_context(role), memory)

        calls: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        timeline_id = world.audit_timeline_id(AUDITOR)
        while world.audit_state(AUDITOR)["status"] == "negotiating":
            role = world.audit_state(AUDITOR)["next_actor"]
            reply = agents[role].respond(world.observe(role))
            for record in reply.call_records:
                record["timeline_id"] = timeline_id
                record["call_sequence"] = len(calls) + 1
                calls.append(copy.deepcopy(record))
            receipt = world.act(reply.turn.to_action(role))
            receipts.append(copy.deepcopy(receipt))
            if not receipt["accepted"]:
                raise IllegalAgentAction(
                    f"{role} 行动被世界拒绝: {receipt['result']['refusal']}",
                    calls,
                    receipts,
                )
            if on_turn is not None:
                on_turn(role, reply.turn, receipt)
        return NegotiationRun(
            timeline_id=timeline_id,
            metrics=world.metrics(AUDITOR),
            model_calls=tuple(calls),
            receipts=tuple(receipts),
        )
