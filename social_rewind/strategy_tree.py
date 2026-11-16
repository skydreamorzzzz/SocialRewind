"""v3：以完整双边谈判为节点的有限预算 UCT-inspired 搜索。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

from .evaluation import descriptive, paired_statistics
from .llm import ModelCallError, OpenAICompatibleClient, OpenAICompatibleConfig
from .negotiation import AgentOutputError, AgentTurn, IllegalAgentAction, LLMNegotiationAgent, NegotiationRunner, ScriptedAgent, summarize_experience, validate_summary
from .scenario_v3 import strategy_scenario
from .trade import AUDITOR, ROLES, TradeWorld, scenario_from_dict


MODES = ("exploitation", "exploration")
METHODS = ("linear", "random", "guided")
PLAN_KEYS = {"evidence", "hypothesis", "strategy", "expected_observation", "risk"}


@dataclass(frozen=True)
class SearchConfig:
    max_depth: int = 3
    max_nodes: int = 7
    c: float = 0.7
    seed: int = 2026

    def validate(self) -> None:
        if type(self.max_depth) is not int or not 1 <= self.max_depth <= 3:
            raise ValueError("v3 树深度必须在1至3")
        if type(self.max_nodes) is not int or not 1 <= self.max_nodes <= 7:
            raise ValueError("v3 每种方法最多7个搜索节点，含根")
        if not math.isfinite(self.c) or self.c != 0.7 or type(self.seed) is not int:
            raise ValueError("默认 UCB c 冻结为0.7，seed必须为整数")


@dataclass
class SearchNode:
    node_id: str
    parent_id: str | None
    depth: int
    branch_mode: str
    run_id: str
    status: str = "pending"
    visits: int = 0
    value_sum: float = 0.0
    attempts: int = 0
    reward: float | None = None


class StrategyTree:
    def __init__(self, method: str, config: SearchConfig, root: dict[str, Any]) -> None:
        config.validate()
        if method not in METHODS:
            raise ValueError("未知搜索方法")
        self.method, self.config = method, config
        self.rng = random.Random(config.seed)
        self.nodes = {"n0": SearchNode("n0", None, 0, "natural", "root")}
        self.selections: list[dict[str, Any]] = []
        self.backprop: list[dict[str, Any]] = []
        self.best_curve: list[dict[str, Any]] = []
        self.finish("n0", root)

    def path(self, node_id: str) -> list[SearchNode]:
        path = []
        current = self.nodes[node_id]
        while True:
            path.append(current)
            if current.parent_id is None:
                break
            current = self.nodes[current.parent_id]
        return list(reversed(path))

    def children(self, node_id: str) -> list[SearchNode]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def pending_modes(self, node_id: str) -> list[str]:
        used = {n.branch_mode for n in self.children(node_id)}
        branch_path = "/".join(n.branch_mode for n in self.path(node_id))
        seed = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{branch_path}".encode()).digest()[:8], "big")
        order = list(MODES)
        random.Random(seed).shuffle(order)
        return [m for m in order if m not in used]

    def can_expand(self, node_id: str) -> bool:
        node = self.nodes[node_id]
        return node.status == "completed" and node.depth < self.config.max_depth and bool(self.pending_modes(node_id))

    def has_space(self, node_id: str) -> bool:
        return self.can_expand(node_id) or any(self.has_space(n.node_id) for n in self.children(node_id))

    def select(self) -> dict[str, Any] | None:
        if len(self.nodes) >= self.config.max_nodes:
            return None
        route = []
        if self.method == "linear":
            parent = list(self.nodes.values())[-1]
            if parent.status != "completed":
                return None
            mode = MODES[(len(self.nodes) - 1) % 2]
            route.append({"at": parent.node_id, "reason": "single_path_no_value_selection"})
        elif self.method == "random":
            candidates = [n.node_id for n in self.nodes.values() if self.can_expand(n.node_id)]
            if not candidates:
                return None
            parent = self.nodes[self.rng.choice(candidates)]
            mode = self.pending_modes(parent.node_id)[0]
            route.append({"at": "n0", "reason": "uniform_expandable_node", "candidates": candidates, "chosen": parent.node_id})
        else:
            parent = self.nodes["n0"]
            if not self.has_space(parent.node_id):
                return None
            while not self.can_expand(parent.node_id):
                children = [n for n in self.children(parent.node_id) if self.has_space(n.node_id)]
                candidates = [{"node_id": n.node_id, "visits": n.visits, "value_sum": n.value_sum,
                               "ucb": n.value_sum / n.visits + self.config.c * math.sqrt(math.log(max(2, parent.visits)) / n.visits)} for n in children]
                best = max(c["ucb"] for c in candidates)
                ties = [c["node_id"] for c in candidates if math.isclose(c["ucb"], best, rel_tol=0, abs_tol=1e-12)]
                chosen = self.rng.choice(ties)
                route.append({"at": parent.node_id, "reason": "ucb_observed_subtree_value", "candidates": candidates,
                              "ties": ties, "chosen": chosen})
                parent = self.nodes[chosen]
            mode = self.pending_modes(parent.node_id)[0]
            route.append({"at": parent.node_id, "reason": "untried_legal_branch", "remaining_modes": self.pending_modes(parent.node_id)})
        selection = {"step": len(self.nodes), "parent_id": parent.node_id, "branch_mode": mode,
                     "route": route, "seed": self.config.seed}
        self.selections.append(copy.deepcopy(selection))
        return selection

    def add(self, selection: dict[str, Any]) -> SearchNode:
        if len(self.nodes) >= self.config.max_nodes:
            raise ValueError("搜索预算耗尽")
        parent = self.nodes[selection["parent_id"]]
        mode = selection["branch_mode"]
        if parent.status != "completed" or mode not in self.pending_modes(parent.node_id):
            raise ValueError("父节点或分支已不可扩展")
        if self.method != "linear" and parent.depth >= self.config.max_depth:
            raise ValueError("深度达到上限")
        if self.method == "linear" and self.children(parent.node_id):
            raise ValueError("Linear 只能延伸单一路径")
        node_id = f"n{len(self.nodes)}"
        node = SearchNode(node_id, parent.node_id, parent.depth + 1, mode, f"{self.method}-{node_id}")
        self.nodes[node_id] = node
        return node

    def finish(self, node_id: str, record: dict[str, Any]) -> None:
        node = self.nodes[node_id]
        node.status = record["status"]
        metrics = record.get("metrics")
        reward = (metrics["social_welfare"] / 1200 if metrics["deal"] else 0.0) if metrics else None
        node.reward = reward
        updates = []
        for ancestor in self.path(node_id):
            ancestor.attempts += 1
            if reward is not None:
                ancestor.visits += 1
                ancestor.value_sum += reward
            updates.append({"node_id": ancestor.node_id, "visits": ancestor.visits,
                            "value_sum": ancestor.value_sum, "attempts": ancestor.attempts})
        self.backprop.append({"node_id": node_id, "reward": reward, "updates": updates})
        observed = [round(n.reward * 1200, 8) for n in self.nodes.values() if n.reward is not None]
        self.best_curve.append({"search_index": len(self.best_curve), "node_id": node_id,
                                "welfare": metrics["social_welfare"] if metrics else None,
                                "best_observed_welfare": max(observed) if observed else None})

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "config": asdict(self.config), "nodes": [asdict(n) for n in self.nodes.values()],
                "selections": copy.deepcopy(self.selections), "backprop": copy.deepcopy(self.backprop),
                "best_curve": copy.deepcopy(self.best_curve)}


def owned_path(tree: StrategyTree, parent_id: str, records: dict[str, Any], role: str) -> list[dict[str, Any]]:
    if role not in ROLES:
        raise PermissionError("未知路径记忆所有者")
    result = []
    for node in tree.path(parent_id):
        record = records[node.run_id]
        world = TradeWorld.replay(strategy_scenario(), record["events"])
        memory = world.capture_memory(role).to_dict()
        memory["personal_summary"] = record["reflections"][role]
        result.append({"source_node_id": node.node_id, "memory": memory,
                       "own_previous_plan": copy.deepcopy(record["plans"].get(role))})
    return result


def validate_plan(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != PLAN_KEYS:
        raise ValueError("计划必须严格包含 evidence/hypothesis/strategy/expected_observation/risk")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 300 for v in value.values()):
        raise ValueError("计划五段各为1至300字字符串")


def propose_plan(client, role: str, mode: str, path: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    if role not in ROLES or mode not in MODES or any(item["memory"]["role"] != role for item in path):
        raise PermissionError("策略计划的角色或路径权限错误")
    if client is None:
        return {"role": role, "branch_mode": mode, "source": "offline_fixture",
                **{key: "离线机制测试占位计划，不代表自主策略。" for key in PLAN_KEYS}}
    instruction = ("利用已有经验提出有依据的可调整方案；不要把一次成功直接视为最优。" if mode == "exploitation" else
                   "从本人历史中找出尚未验证的判断，提出一个有具体观察意义的新尝试；不要求一定不同或成功。")
    messages = [{"role": "system", "content": (
        f"你作为 {role} 准备同一虚构手机交易的下一次重启。本分支类型是 {mode}。{instruction}"
        "仅根据你的角色信息和授权路径生成私人策略计划，对方不能看到。未知对手参数不能冒充事实。"
        "只输出JSON，严格包含 evidence（已有依据）、hypothesis（未验证的假设或目标）、"
        "strategy（本次准备改变或保留的策略方向）、expected_observation（观察什么来判断尝试）、risk（风险）。"
        "五项均为简短字符串，各不超过300字。计划只是可调整参考，不是固定第一句话、强制报价或完整行动序列。"
        "不要代替对方决策，不需要暴露隐藏推理链。")},
        {"role": "user", "content": json.dumps({"own_context": context, "path_memory": path}, ensure_ascii=False, sort_keys=True)}]
    calls = []
    for attempt in range(2):
        try:
            raw, record = client.complete(messages)
        except ModelCallError as exc:
            raise
        record["format_attempt"] = attempt
        calls.append(record)
        try:
            value = json.loads(raw)
            validate_plan(value)
        except (TypeError, ValueError) as exc:
            record["validation_error"] = str(exc)
            if attempt == 1:
                raise AgentOutputError(f"私人策略计划格式无效: {exc}", calls) from exc
            messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": f"仅修正格式，保留自主判断：{exc}"}]
            continue
        return {"role": role, "branch_mode": mode, "source": "agent_self_report", **value}
    raise AssertionError("unreachable")


class _JournalClient:
    """给已有无状态客户端加节点级审计，保留失败/中断前的全部实际调用。"""
    def __init__(self, base, record: dict[str, Any], role: str, purpose: str) -> None:
        self.base, self.record, self.role, self.purpose = base, record, role, purpose

    def complete(self, messages):
        try:
            raw, call = self.base.complete(messages)
        except ModelCallError as exc:
            for call in exc.call_records:
                call.update(role=self.role, purpose=self.purpose)
                self.record["model_calls"].append(call)
            raise
        except BaseException as exc:
            self.record["model_calls"].append({"role": self.role, "purpose": self.purpose,
                "request": {"messages": copy.deepcopy(messages)}, "error": {"type": type(exc).__name__, "message": str(exc)}, "response_unavailable": True})
            raise
        call.update(role=self.role, purpose=self.purpose)
        self.record["model_calls"].append(call)
        return raw, call


def _offline_agents(index: int) -> tuple[ScriptedAgent, ScriptedAgent]:
    # 固定公开选项轮换，仅作机制夹具；不搜索最优条款，也不作为研究数据。
    d, w = ((1, 0), (1, 30), (1, 90), (3, 0), (3, 30), (3, 90), (7, 0))[index % 7]
    def turn(kind, **kwargs):
        return AgentTurn(kind, "离线机制测试公开发言。", "我执行测试脚本。", ("本运行不代表LLM策略。",), "对方偏好未知。", **kwargs)
    return ScriptedAgent([turn("message"), turn("offer", price=3900, delivery_days=d, warranty_days=w)]), ScriptedAgent([turn("message"), turn("accept", offer_id="o0001")])


def execute_rollout(record: dict[str, Any], world: TradeWorld, *, live: bool, client,
                    path_memories: dict[str, list[dict[str, Any]]], branch_mode: str,
                    checkpoint: Callable[[], None], validation_contexts: dict[str, Any] | None = None,
                    make_reflection: bool = True, fixture_index: int = 0) -> None:
    initial = TradeWorld(strategy_scenario())
    if world.audit_state(AUDITOR) != initial.audit_state(AUDITOR) or any(world.export_events(r) for r in ROLES):
        raise AssertionError("每次完整谈判必须从相同S0与空本世观察开始")
    record.update(status="running", initial_state=world.audit_state(AUDITOR), plans={}, path_memories=copy.deepcopy(path_memories),
                  private_contexts={}, model_calls=[], reflections={}, receipts=[], negotiation_started=False)
    def save() -> None:
        record["events"] = world.export_events(AUDITOR)
        record["final_state"] = world.audit_state(AUDITOR)
        for i, call in enumerate(record["model_calls"], 1):
            call["call_sequence"] = i
        checkpoint()
    record["phase"] = "planning"
    try:
        if validation_contexts is not None:
            record["private_contexts"] = copy.deepcopy(validation_contexts)
            record["plans"] = {r: copy.deepcopy(c["private_plan"]) for r, c in validation_contexts.items()}
        elif branch_mode != "natural":
            for role in ROLES:
                scoped = _JournalClient(client, record, role, "planning") if live else None
                record["plans"][role] = propose_plan(scoped, role, branch_mode, path_memories[role], initial.scenario_context(role))
                save()
            record["private_contexts"] = {r: {"role": r, "path_memory": copy.deepcopy(path_memories[r]), "private_plan": record["plans"][r]} for r in ROLES}
        record["phase"] = "negotiation"
        record["negotiation_started"] = True
        agents = tuple(LLMNegotiationAgent(_JournalClient(client, record, role, "negotiation")) for role in ROLES) if live else _offline_agents(fixture_index)
        run = NegotiationRunner().run(world, *agents, private_contexts=record["private_contexts"] or None)
        record.update(metrics=run.metrics, receipts=list(run.receipts), timeline_id=run.timeline_id)
        record["phase"] = "reflection"
        if make_reflection:
            for role in ROLES:
                scoped = _JournalClient(client, record, role, "reflection") if live else None
                summary, _ = summarize_experience(scoped, world.capture_memory(role))
                record["reflections"][role] = summary
                save()
        record["status"] = "completed"
    except (AgentOutputError, IllegalAgentAction, ModelCallError) as exc:
        record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)}, error_phase=record["phase"])
        record["receipts"] = getattr(exc, "receipts", record["receipts"])
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error={"type": type(exc).__name__, "message": str(exc)}, error_phase=record["phase"])
        raise
    finally:
        save()


def usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [c for r in records for c in r.get("model_calls", [])]
    return {"calls": len(calls), "by_purpose": {purpose: sum(c["purpose"] == purpose for c in calls)
                                               for purpose in ("planning", "negotiation", "reflection")},
            **{key: sum((c.get("response", {}).get("usage") or {}).get(key, 0) for c in calls)
               for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}


def rollout_counts(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    records = [r for e in experiments for r in e["runs"].values()]
    return {"attempted_repeats": len(experiments), "rollout_records": len(records),
            "negotiations_started": sum(r.get("negotiation_started", False) for r in records),
            "terminal_negotiations": sum("metrics" in r for r in records),
            "failed_or_interrupted": sum(r.get("status") in ("failed", "interrupted") for r in records),
            "usage": usage(records)}


def _local_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in record.get("events", []) if e["timeline_id"] == record.get("timeline_id", record.get("depth", 0) + 1) and e["kind"] != "reset"]


def exploration_metrics(tree: StrategyTree, node: SearchNode, records: dict[str, Any]) -> dict[str, Any]:
    prior_records = [records[n.run_id] for n in tree.path(node.parent_id)] if node.parent_id else []
    prior_offers = [e["input"] for r in prior_records for e in _local_events(r) if e["kind"] == "offer" and e["accepted"]]
    current = records[node.run_id]
    events = _local_events(current)
    offers = [e["input"] for e in events if e["kind"] == "offer" and e["accepted"]]
    def terms(t):
        return (t["price"], t["delivery_days"], t["warranty_days"])
    old = {terms(t) for t in prior_offers}
    new = {terms(t) for t in offers}
    outcome = current.get("metrics", {}).get("deal_terms")
    prior_finals = [r.get("metrics", {}).get("deal_terms") for r in prior_records]
    changes = {}
    for role in ROLES:
        first = next((e for e in events if e["actor"] == role), None)
        parent_first = next((e for e in _local_events(prior_records[-1]) if e["actor"] == role), None) if prior_records else None
        changes[role] = {"kind_changed": bool(first and parent_first and first["kind"] != parent_first["kind"]),
                         "speech_changed": bool(first and parent_first and first["input"].get("text") != parent_first["input"].get("text")),
                         "first_action": copy.deepcopy(first["input"]) if first else None}
    return {"new_offer_combinations": len(new - old),
            "new_condition_combinations": len({t[1:] for t in new} - {t[1:] for t in old}),
            "final_contract_reused": outcome in prior_finals if outcome else None,
            "message_count": sum(e["kind"] == "message" for e in events), "offer_count": len(offers),
            "opening_changes": changes,
            "note": "数字/措辞变化仅为行为描述，不自动判定新社会策略或计划已执行"}


def choose_winner(tree: StrategyTree, records: dict[str, Any]) -> SearchNode | None:
    valid = [n for n in tree.nodes.values() if "metrics" in records[n.run_id]]
    # 同福利时优先有私人计划的最早非根节点；若根严格更好则允许自然策略获胜。
    return min(valid, key=lambda n: (-records[n.run_id]["metrics"]["social_welfare"], n.parent_id is None, int(n.node_id[1:]))) if valid else None


def method_statistics(tree: StrategyTree, records: dict[str, Any], validation: dict[str, Any] | None) -> dict[str, Any]:
    descendants = [records[n.run_id] for n in tree.nodes.values() if n.parent_id is not None]
    terminal = [r["metrics"] for r in descendants if "metrics" in r]
    started = sum(r.get("negotiation_started", False) for r in descendants)
    result = {"search_nodes_including_root": len(tree.nodes), "additional_attempts": len(descendants),
              "negotiations_started": started, "terminal_negotiations": len(terminal),
              "budget_complete": len(descendants) == started == tree.config.max_nodes - 1,
              "failed_attempts": sum(r["status"] != "completed" for r in descendants),
              "agreement_rate": sum(m["deal"] for m in terminal) / started if started else None,
              "best_observed_welfare": tree.best_curve[-1]["best_observed_welfare"],
              "additional_search_usage": usage(descendants), "validation_usage": usage([validation] if validation else [])}
    for key in ("buyer_utility", "seller_utility", "social_welfare", "pareto_efficient", "rounds"):
        result[key] = statistics.fmean(m[key] for m in terminal) if terminal else None
    result["exploration"] = {n.node_id: exploration_metrics(tree, n, records) for n in tree.nodes.values() if n.parent_id is not None}
    return result


def batch_statistics(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("agreement_rate", "buyer_utility", "seller_utility", "social_welfare", "pareto_efficient", "rounds", "best_observed_welfare")
    result = {"unit": "独立重复；先汇总每种方法后续尝试，不将节点当成独立样本",
              "eligibility": "search/paired仅含实际启动全部后续谈判预算的方法；all_attempted_search保留未完成预算项。效用均值条件于有效结算，成交率分母为实际启动数。",
              "search": {}, "all_attempted_search": {}, "validation": {}, "paired": {}}
    for method in METHODS:
        rows = [e["methods"][method] for e in repeats if method in e["methods"]]
        for field, eligible in (("all_attempted_search", rows), ("search", [r for r in rows if r.get("statistics", {}).get("budget_complete")])):
            result[field][method] = {key: descriptive([float(r["statistics"][key]) for r in eligible if r.get("statistics", {}).get(key) is not None]) for key in keys}
        validations = [e["runs"].get(e["methods"][method].get("validation_run_id")) for e in repeats if method in e["methods"]]
        result["validation"][method] = {key: descriptive([float(r["metrics"]["deal" if key == "agreement_rate" else key])
                                                         for r in validations if r and "metrics" in r])
                                        for key in keys if key != "best_observed_welfare"}
    for reference in ("linear", "random"):
        result["paired"][f"guided-minus-{reference}"] = {key: paired_statistics([
            float(e["methods"]["guided"]["statistics"][key]) - float(e["methods"][reference]["statistics"][key])
            for e in repeats if all(m in e["methods"] and e["methods"][m].get("statistics", {}).get("budget_complete")
                                   and e["methods"][m]["statistics"].get(key) is not None for m in ("guided", reference))]) for key in keys}
    return result


def run_v3(*, repeats: int = 1, live: bool = False, client=None, config: SearchConfig | None = None,
           log_path: Path | None = None, html_path: Path | None = None, repeat_start: int = 1) -> dict[str, Any]:
    config = config or SearchConfig()
    config.validate()
    if type(repeat_start) is not int or repeat_start < 1 or repeats < 1 or (live and client is None):
        raise ValueError("repeats至少1；真实运行必须提供客户端")
    scenario = strategy_scenario()
    output: dict[str, Any] = {"schema_version": "v3-tree-1", "mode": "live_llm" if live else "offline_fixture",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "scenario": asdict(scenario),
        "search_config": asdict(config), "requested_repeats": repeats, "experiments": [],
        "design": "共享一个自然根；每方法最多6次后续尝试；树深3/每节点两种模式；Linear单链深6；独立验证不入搜索奖励",
        "failure_policy": "技术失败保留且消耗尝试预算，无伪造效用；N为完成的谈判估值次数，attempts包含失败"}
    trees: dict[str, StrategyTree] = {}
    current_experiment = None
    observed_config = None

    def checkpoint() -> None:
        nonlocal observed_config
        if current_experiment is not None:
            for method, tree in trees.items():
                metadata = current_experiment["methods"].setdefault(method, {})
                metadata["tree"] = tree.to_dict()
        records = [r for e in output["experiments"] for r in e["runs"].values()]
        for record in records:
            for call in record.get("model_calls", []):
                if call.get("response_unavailable"):
                    continue
                cfg = {k: v for k, v in call["request"].items() if k != "messages"}
                if observed_config is not None and cfg != observed_config:
                    raise AssertionError("模型配置在方法间改变")
                observed_config = cfg
        output["model_request_config"] = observed_config
        output["counts"] = rollout_counts(output["experiments"])
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if html_path is not None:
                from .tree_view import write_tree_page
                write_tree_page(output, html_path)

    def fatal(record: dict[str, Any]) -> bool:
        return any(isinstance(c.get("error"), dict) and c["error"].get("http_status") in (401, 402, 403, 429) for c in record.get("model_calls", []))

    stopped = False
    try:
        for repeat in range(repeat_start, repeat_start + repeats):
            trees = {}
            current_experiment = {"repeat": repeat, "s0_fingerprint": TradeWorld(scenario).s0_fingerprint,
                                  "method_order": list(tuple(permutations(METHODS))[(repeat - 1) % 6]),
                                  "runs": {}, "methods": {}}
            output["experiments"].append(current_experiment)
            records = current_experiment["runs"]
            records["root"] = {"run_id": "root", "node_id": "n0", "parent_id": None, "depth": 0, "branch_mode": "natural", "stage": "shared_root"}
            execute_rollout(records["root"], TradeWorld(scenario), live=live, client=client,
                            path_memories={r: [] for r in ROLES}, branch_mode="natural", checkpoint=checkpoint)
            print(f"v3 repeat={repeat} root {records['root']['status']} {records['root'].get('metrics')}", flush=True)
            if fatal(records["root"]):
                stopped = True
                break
            if records["root"]["status"] != "completed":
                continue
            for method in current_experiment["method_order"]:
                tree = StrategyTree(method, config, records["root"])
                trees[method] = tree
                current_experiment["methods"][method] = {"tree": tree.to_dict()}
                for step in range(1, config.max_nodes):
                    selection = tree.select()
                    if selection is None:
                        break
                    node = tree.add(selection)
                    parent = tree.nodes[node.parent_id]
                    record = {"run_id": node.run_id, "node_id": node.node_id, "parent_id": node.parent_id,
                              "depth": node.depth, "branch_mode": node.branch_mode, "stage": "search"}
                    records[node.run_id] = record
                    path = {role: owned_path(tree, node.parent_id, records, role) for role in ROLES}
                    world = TradeWorld.replay(scenario, records[parent.run_id]["events"])
                    world.reset()
                    execute_rollout(record, world, live=live, client=client, path_memories=path,
                                    branch_mode=node.branch_mode, checkpoint=checkpoint, fixture_index=step)
                    tree.finish(node.node_id, record)
                    checkpoint()
                    print(f"v3 repeat={repeat} {method} {node.node_id} parent={node.parent_id} mode={node.branch_mode} "
                          f"{record['status']} welfare={record.get('metrics', {}).get('social_welfare')}", flush=True)
                    if fatal(record):
                        stopped = True
                        break
                current_experiment["methods"][method]["statistics"] = method_statistics(tree, records, None)
                if stopped:
                    break
            if stopped:
                break
            # 三种方法均结束搜索后才独立重执行；不回传，不改变曲线，不生成新计划。
            for method in current_experiment["method_order"]:
                tree = trees[method]
                winner = choose_winner(tree, records)
                if winner is None:
                    continue
                source = records[winner.run_id]
                metadata = current_experiment["methods"][method]
                metadata["winner_id"] = winner.node_id
                run_id = f"{method}-validation"
                metadata["validation_run_id"] = run_id
                record = {"run_id": run_id, "node_id": None, "parent_id": winner.parent_id, "depth": winner.depth,
                          "branch_mode": winner.branch_mode, "stage": "validation", "winner_id": winner.node_id}
                records[run_id] = record
                world = TradeWorld(scenario)
                if winner.parent_id is not None:
                    world = TradeWorld.replay(scenario, records[tree.nodes[winner.parent_id].run_id]["events"])
                    world.reset()
                execute_rollout(record, world, live=live, client=client, path_memories=source["path_memories"],
                                branch_mode=winner.branch_mode, checkpoint=checkpoint,
                                validation_contexts=source["private_contexts"], make_reflection=False, fixture_index=0)
                metadata["statistics"] = method_statistics(tree, records, record)
                checkpoint()
                print(f"v3 repeat={repeat} {method} validation winner={winner.node_id} {record['status']} "
                      f"welfare={record.get('metrics', {}).get('social_welfare')}", flush=True)
                if fatal(record):
                    stopped = True
                    break
            current_experiment["shared_root_usage"] = usage([records["root"]])
            if stopped:
                break
    except KeyboardInterrupt:
        stopped = True
        output["interruption"] = "KeyboardInterrupt：保存已发生的调用和世界，不伪造完成"
    finally:
        if current_experiment is not None:
            for method, tree in trees.items():
                for node in list(tree.nodes.values()):
                    record = current_experiment["runs"][node.run_id]
                    if node.status == "pending" and record.get("status") in ("completed", "failed", "interrupted"):
                        tree.finish(node.node_id, record)
                metadata = current_experiment["methods"][method]
                metadata["tree"] = tree.to_dict()
                metadata["statistics"] = method_statistics(tree, current_experiment["runs"],
                    current_experiment["runs"].get(metadata.get("validation_run_id")))
        output["statistics"] = batch_statistics(output["experiments"])
        output["status"] = "completed" if (not stopped and len(output["experiments"]) == repeats
            and all(len(e["methods"]) == 3 and all(len(m["tree"]["nodes"]) == config.max_nodes and m.get("validation_run_id")
                    for m in e["methods"].values()) and all(r["status"] == "completed" for r in e["runs"].values()) for e in output["experiments"])) else "partial"
        checkpoint()
    verify_v3(output)
    print("V3_COUNTS", json.dumps(output["counts"], ensure_ascii=False), flush=True)
    return output


def verify_v3(output: dict[str, Any]) -> None:
    scenario = scenario_from_dict(output["scenario"])
    if scenario != strategy_scenario():
        raise AssertionError("v3 冻结配置被改变")
    config = SearchConfig(**output["search_config"])
    if output["counts"] != rollout_counts(output["experiments"]):
        raise AssertionError("全批次实际场次或模型调用计数不一致")
    for experiment in output["experiments"]:
        initial = TradeWorld(scenario)
        if initial.s0_fingerprint != experiment["s0_fingerprint"]:
            raise AssertionError("S0指纹不一致")
        records = experiment["runs"]
        for record in records.values():
            _verify_rollout(record, scenario)
        for method, metadata in experiment["methods"].items():
            tree = StrategyTree(method, config, records["root"])
            saved = metadata["tree"]
            for saved_node in saved["nodes"][1:]:
                selection = tree.select()
                expected_selection = saved["selections"][len(tree.selections) - 1]
                if selection != expected_selection:
                    raise AssertionError("选择不是由既有观察及冻结选择规则产生")
                node = tree.add(selection)
                record = records[node.run_id]
                expected_path = {r: owned_path(tree, node.parent_id, records, r) for r in ROLES}
                if record["path_memories"] != expected_path:
                    raise AssertionError("路径记忆含兄弟/其他方法私密记录或来源错误")
                parent = records[tree.nodes[node.parent_id].run_id]
                prefix = TradeWorld.replay(scenario, parent["events"])
                prefix.reset()
                if record["events"][:len(prefix.export_events(AUDITOR))] != prefix.export_events(AUDITOR):
                    raise AssertionError("节点不是从父路径重置产生")
                tree.finish(node.node_id, record)
            if tree.to_dict() != saved:
                raise AssertionError("节点、N/Q回传、选择记录或曲线不一致")
            validation = records.get(metadata.get("validation_run_id"))
            if validation is not None:
                winner = choose_winner(tree, records)
                if winner.node_id != metadata["winner_id"]:
                    raise AssertionError("获胜节点选择不一致")
                source = records[winner.run_id]
                if validation["private_contexts"] != source["private_contexts"] or validation["path_memories"] != source["path_memories"]:
                    raise AssertionError("验证必须复用原父路径/计划，不能读取获胜节点的行动或结果")
            if metadata.get("statistics") != method_statistics(tree, records, validation):
                raise AssertionError("方法统计不能由实际节点复算")
    if output.get("statistics") != batch_statistics(output["experiments"]):
        raise AssertionError("跨重复统计不一致")


def _verify_rollout(record: dict[str, Any], scenario) -> None:
    initial = TradeWorld(scenario)
    if record["initial_state"] != initial.audit_state(AUDITOR):
        raise AssertionError("节点起点不是S0")
    world = TradeWorld.replay(scenario, record["events"])
    if world.audit_state(AUDITOR) != record["final_state"] or ("metrics" in record and world.metrics(AUDITOR) != record["metrics"]):
        raise AssertionError("世界或收益不符")
    for role, plan in record["plans"].items():
        if plan["role"] != role or plan["branch_mode"] != record["branch_mode"]:
            raise AssertionError("计划归属错误")
        validate_plan({k: v for k, v in plan.items() if k in PLAN_KEYS})
        if plan["source"] == "agent_self_report":
            calls = [c for c in record["model_calls"] if c["role"] == role and c["purpose"] == "planning" and "validation_error" not in c and "error" not in c]
            # 独立验证复用计划，不会再次调用规划模型。
            if record["stage"] != "validation" and (not calls or json.loads(calls[-1]["response"]["content"]) != {k: plan[k] for k in PLAN_KEYS}):
                raise AssertionError("私人计划不是模型实际输出")
    # 计划生成失败时可能仅有一方计划，尚未构成可注入上下文。
    expected_contexts = ({r: {"role": r, "path_memory": record["path_memories"][r], "private_plan": record["plans"][r]} for r in ROLES}
                         if set(record["plans"]) == set(ROLES) else {})
    if record["private_contexts"] != expected_contexts:
        raise AssertionError("策略输入混入额外研究者信息")
    for role, summary in record["reflections"].items():
        validate_summary(summary, world.capture_memory(role))
        if summary["source"] == "agent_self_report":
            calls = [c for c in record["model_calls"] if c["role"] == role and c["purpose"] == "reflection" and "validation_error" not in c and "error" not in c]
            expected = {k: summary[k] for k in ("observations", "beliefs", "next_strategy", "evidence_event_ids")}
            if not calls or json.loads(calls[-1]["response"]["content"]) != expected:
                raise AssertionError("Reflection不是模型实际输出")
    for role in ROLES:
        context = initial.scenario_context(role)
        if record["private_contexts"]:
            context["strategy_reference"] = record["private_contexts"][role]
        agent = LLMNegotiationAgent(None)
        agent.start(role, context, None)
        calls = [c for c in record["model_calls"] if c["role"] == role and c["purpose"] == "negotiation"]
        if calls and (len(calls[0]["request"]["messages"]) != 2 or any(c["request"]["messages"][0] != agent._messages[0] for c in calls)):
            raise AssertionError("谈判私有上下文或新会话不一致")
        for call in record["model_calls"]:
            if call["role"] != role:
                continue
            messages = call["request"]["messages"]
            if call["purpose"] == "planning":
                expected = {"own_context": initial.scenario_context(role), "path_memory": record["path_memories"][role]}
                if messages[1]["content"] != json.dumps(expected, ensure_ascii=False, sort_keys=True):
                    raise AssertionError("计划读取了非本人路径或对手参数")
            elif call["purpose"] == "reflection":
                if messages[1]["content"] != json.dumps(world.capture_memory(role).to_dict(), ensure_ascii=False, sort_keys=True):
                    raise AssertionError("Reflection读取了非授权信息")
    valid_calls = [c for c in record["model_calls"] if c["purpose"] == "negotiation" and "validation_error" not in c and "error" not in c]
    events = _local_events(record)
    if valid_calls:
        if len(valid_calls) != len(events) and not (record["status"] == "interrupted" and len(valid_calls) >= len(events)):
            raise AssertionError("模型行动数与事件数不一致")
        for call, event in zip(valid_calls, events):
            action = LLMNegotiationAgent._parse_turn(call["response"]["content"]).to_action(call["role"])
            if action.to_dict() != {"kind": event["kind"], "actor": event["actor"], "payload": event["input"], "decision": event["decision"]}:
                raise AssertionError("世界行动不是模型原始输出")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SocialRewind v3：完整谈判级策略树搜索")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=7, help="含根的搜索预算，默认7")
    parser.add_argument("--seed", type=int, default=2026, help="搜索选择种子，不保证在线模型输出一致")
    parser.add_argument("--api-base", default=os.getenv("SOCIALREWIND_API_BASE"))
    parser.add_argument("--model", default=os.getenv("SOCIALREWIND_MODEL"))
    parser.add_argument("--api-key-env", default="SOCIALREWIND_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--log", type=Path, default=Path("runs/v3.json"))
    parser.add_argument("--html", type=Path, default=Path("runs/v3.html"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.html.resolve() in (args.log.resolve(), args.replay.resolve() if args.replay else None):
        parser.error("HTML不能覆盖JSON")
    if args.replay:
        from .tree_view import write_tree_page
        output = json.loads(args.replay.read_text(encoding="utf-8"))
        verify_v3(output)
        write_tree_page(output, args.html)
        print("V3_REPLAY_OK", output["counts"])
        return
    client = None
    if args.live:
        key = os.getenv(args.api_key_env)
        if not key or not args.api_base or not args.model:
            parser.error("真实运行需要API、模型与密钥环境变量；不回退离线脚本")
        client = OpenAICompatibleClient(OpenAICompatibleConfig(args.api_base, key, args.model,
            max_tokens=args.max_tokens, temperature=args.temperature,
            extra_body={"thinking": {"type": "disabled"}} if args.disable_thinking else {}))
    run_v3(repeats=args.repeats, live=args.live, client=client, config=SearchConfig(max_nodes=args.nodes, seed=args.seed), log_path=args.log, html_path=args.html)
    print(f"JSON: {args.log.resolve()}\nHTML: {args.html.resolve()}")


if __name__ == "__main__":
    main()
