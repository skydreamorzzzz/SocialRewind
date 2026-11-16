from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .llm import ModelCallError, OpenAICompatibleClient
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
    delivery_days: int | None = None
    warranty_days: int | None = None

    def to_action(self, role: str) -> Action:
        if self.kind == "offer":
            payload: dict[str, Any] = {"text": self.speech, "price": self.price}
            if self.delivery_days is not None or self.warranty_days is not None:
                payload.update(delivery_days=self.delivery_days, warranty_days=self.warranty_days)
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
    previous = ""
    if memory.previous_lives:
        previous = "\n更早时间线的本人授权经历（事件编号只在各自时间线内有效）：\n" + json.dumps(
            memory.previous_lives, ensure_ascii=False, sort_keys=True
        )
    return f"""【轮回开始：第 {memory.source_timeline_id + 1} 世】

上一世，你曾作为 {memory.role} 参与同一场交易。
你经历了完整的谈判，做出过报价与选择，也亲眼见证了对方的公开回应和最终结果。
然而，时间倒流了。商品、规则、身份和初始条件都恢复了原状，唯独你的记忆没有消失。

更重要的是，这是双方共同知道的事实：{counterpart_role} 也重生了，也记得其上一世亲眼见过的谈判，并可能改变策略。但你并不知道对方未公开的私人估值或私人决策。

你获准保留的上一世经历如下：
{history}

上一世结果：{json.dumps(memory.outcome, ensure_ascii=False, sort_keys=True)}
你自己的私人事实：{memory.own_private_fact}
你自己生成的上一世经验总结（未经事实核查，判断可能错误）：{json.dumps(memory.personal_summary, ensure_ascii=False)}
{previous}

这一世，我重生了。你可以延续旧决定、吸取教训、试探对方或采用不同策略；不要求必须改变，也不要求必须成交。不要把猜测说成已验证事实。历史编号不能当作当前时间线的 evidence_event_ids。
"""


def render_control_prompt() -> str:
    return """【重置后的对照谈判】

世界已经恢复到与第一场谈判完全相同的初始状态。你知道世界发生过重置，对方也知道这一点。
但是，你和对方都没有保留任何可用的前世经历：没有上一场的消息、报价、结果、决策说明或经验总结。
因此你不能引用或猜称上一场具体发生过什么。请只根据当前商品、规则、自己的私人偏好和本场实际观察自主谈判。
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


def summarize_experience(client: OpenAICompatibleClient | None, memory: TradeMemory) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """只给角色本人的授权证据；总结在世界之外，不校正其主观判断。"""
    metadata = {"role": memory.role, "source_timeline_id": memory.source_timeline_id,
                "source": "agent_self_report" if client is not None else "offline_fixture"}
    if client is None:
        return {**metadata, "observations": [f"离线脚本本世可见 {len(memory.visible_events)} 条事件。"],
                "beliefs": "离线机制测试不推断对方偏好。", "next_strategy": "继续使用机制测试脚本，不作为策略改进证据。",
                "evidence_event_ids": []}, []
    messages = [{"role": "system", "content": (
        f"你刚作为 {memory.role} 结束一世手机谈判。根据下方本人授权历史，生成简短个人经验总结。"
        "只输出 JSON，严格包含 observations（1至3条短句数组：我观察到了什么）、"
        "beliefs（短字符串：我对对方需求的新判断及不确定性）、"
        "next_strategy（短字符串：我下一世可能尝试什么，也可维持旧策略）、"
        "evidence_event_ids（本世可见事件编号数组，可为空）。每段不超过300字。"
        "区分观察与猜测，不冒称知道对方未公开的私人信息；判断可以错误，不要求总结出更好策略。"
        "这是显式研究记录，不需要隐藏推理链。不要输出世界行动或改写结果。")},
        {"role": "user", "content": json.dumps(memory.to_dict(), ensure_ascii=False, sort_keys=True)}]
    calls = []
    for attempt in range(2):
        try:
            raw, record = client.complete(messages)
        except ModelCallError as exc:
            exc.call_records = calls + exc.call_records
            for call in exc.call_records:
                call.update(role=memory.role, purpose="reflection", timeline_id=memory.source_timeline_id)
            raise
        record.update(role=memory.role, purpose="reflection", timeline_id=memory.source_timeline_id, format_attempt=attempt)
        calls.append(record)
        try:
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != {"observations", "beliefs", "next_strategy", "evidence_event_ids"}:
                raise ValueError("JSON 仅允许 observations/beliefs/next_strategy/evidence_event_ids")
            validate_summary({**metadata, **value}, memory)
        except (TypeError, ValueError, KeyError) as exc:
            record["validation_error"] = str(exc)
            if attempt == 1:
                raise AgentOutputError(f"个人经验总结无效: {exc}", calls) from exc
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"只修复 JSON 格式与引用，保持你的判断：{exc}"}]
            continue
        return {**metadata, **value}, calls
    raise AssertionError("unreachable")


def validate_summary(summary: dict[str, Any], memory: TradeMemory) -> None:
    if set(summary) != {"role", "source_timeline_id", "source", "observations", "beliefs", "next_strategy", "evidence_event_ids"}:
        raise ValueError("个人总结字段不符合契约")
    if summary["role"] != memory.role or summary["source_timeline_id"] != memory.source_timeline_id:
        raise PermissionError("个人总结所有者或来源时间线错误")
    observations = summary["observations"]
    if not isinstance(observations, list) or not 1 <= len(observations) <= 3:
        raise ValueError("observations 应为1至3条短句")
    if any(not isinstance(s, str) or not s.strip() or len(s) > 300
           for s in [*observations, summary["beliefs"], summary["next_strategy"]]):
        raise ValueError("总结各段必须是1至300字字符串")
    ids = summary["evidence_event_ids"]
    visible_ids = {e["event_id"] for e in memory.visible_events}
    if (not isinstance(ids, list) or any(type(i) is not int or i not in visible_ids for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("总结只能引用本角色本世可见的事件编号")
    if summary["source"] not in ("agent_self_report", "offline_fixture"):
        raise ValueError("个人总结只能标记为角色自述或离线测试")


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
            "你的私人效用定义：" + json.dumps(context["private"], ensure_ascii=False, sort_keys=True)
            + "。目标是提高自己的收益，退出或未成交收益为 0。正式报价/接受的自身收益必须 >= 0。"
            if "contract_rules" in context else
            f"你的私人最高愿付价是 {context['private']['maximum_price']} 元，正式报价或接受不得超过它。"
            if role == "buyer"
            else f"你的私人最低接受价是 {context['private']['minimum_price']} 元，正式报价或接受不得低于它。"
        )
        rebirth = (
            render_rebirth_prompt(memory, counterpart)
            if memory is not None
            else context.get("experiment_context", "【第一世】你没有前世记忆。这是一场新的谈判。")
        )
        if "shared_reset_frame" in context:
            # 消融组共享完全相同的叙事；只替换独立记忆数据，不改谈判目标与规则。
            rebirth = context["shared_reset_frame"] + "\n【本人授权记忆数据】\n" + json.dumps(
                memory.to_dict() if memory is not None else None, ensure_ascii=False, sort_keys=True)
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
        if "contract_rules" in context:
            schema += '\n三维合同 offer 的 action 必须严格为 {"kind":"offer","price":整数,"delivery_days":整数,"warranty_days":整数}。其他行动格式相同。'
            private_rule += "\n公开合同范围：" + json.dumps(context["contract_rules"], ensure_ascii=False)
            private_rule += "\n自身 price_limits_by_terms 是世界按你的效用计算的合法价格边界，null 表示该组合无可行价格；请据此检查正式报价。"
            private_rule += ("\n交付时间指协议达成后天数；保障指售后保障天数。你可以自由询问、沟通、交换条件、"
                             "用新的完整 offer 修改方案、接受、拒绝或退出。双方轮流发言，最多两次行动算一轮；"
                             "消息也占一次行动。新报价取代旧报价。不要求按固定步骤或必须成交。")
            private_rule += ("\n私人效用下限只是合法性的底线，不是你的理想目标。接受前请自主权衡："
                             "当前方案离你的理想目标有多远、继续沟通或交换合同条件可能带来什么收益，"
                             "以及对方拒绝、退出或超时使你得到0收益的风险。正收益本身不等于应该立即接受。"
                             "你可以主动询问、解释，选择披露或保留私人需求；也可以立即成交。"
                             "没有最低谈判轮数，没有必须遵循的谈判流程，不要求一定改变策略或达成协议。")
        system = f"""你是虚构二手交易中的 {role}，对方是 {counterpart}。请自主、诚实地进行合同谈判，可试探、拒绝、退出或成交，但不要暴露隐藏推理链。

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
            try:
                raw, record = self._client.complete(self._messages)
            except ModelCallError as exc:
                exc.call_records = call_records + exc.call_records
                for record in exc.call_records:
                    record["role"] = self._role
                raise
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
            if set(action) not in ({"kind", "price"}, {"kind", "price", "delivery_days", "warranty_days"}):
                raise ValueError("offer 必须包含 price；三维报价须同时包含 delivery_days/warranty_days")
            price = action["price"]
            if isinstance(price, bool) or not isinstance(price, int):
                raise ValueError("offer.price 必须是整数")
            if any(type(action[k]) is not int for k in ("delivery_days", "warranty_days") if k in action):
                raise ValueError("交付和保障天数必须是整数")
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
            delivery_days=action.get("delivery_days"),
            warranty_days=action.get("warranty_days"),
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
        session_context: str | None = None,
        shared_reset_frame: str | None = None,
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
            context = world.scenario_context(role)
            if session_context is not None:
                context["experiment_context"] = session_context
            if shared_reset_frame is not None:
                context["shared_reset_frame"] = shared_reset_frame
            agent.start(role, context, memory)

        calls: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        timeline_id = world.audit_timeline_id(AUDITOR)
        while world.audit_state(AUDITOR)["status"] == "negotiating":
            role = world.audit_state(AUDITOR)["next_actor"]
            try:
                reply = agents[role].respond(world.observe(role))
            except (AgentOutputError, ModelCallError) as exc:
                failed_calls = copy.deepcopy(exc.call_records)
                for record in failed_calls:
                    record["timeline_id"] = timeline_id
                    record["call_sequence"] = len(calls) + 1
                    calls.append(record)
                exc.call_records = calls
                exc.receipts = receipts
                raise
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
