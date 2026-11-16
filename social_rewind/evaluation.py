"""事后接受边界估计及配对统计；绝不参与世界行动或记忆构造。"""
from __future__ import annotations

import json
import random
import statistics
from typing import Any

from .llm import ModelCallError, OpenAICompatibleClient
from .negotiation import AgentOutputError
from .trade import AUDITOR, ROLES, TradeWorld


GROUPS = ("control", "rewind_h", "rewind_hr")
METRICS = ("agreement_rate", "buyer_utility", "seller_utility", "social_welfare",
           "pareto_efficient", "rounds", "buyer_estimation_error", "seller_estimation_error")
RESET_FRAME = """【第二世：共同重启】
上一世已经结束。时间倒流，商品、角色、规则和初始条件均恢复原状。
这一世，我重生了。双方都知道发生过重启，双方获得的前世资料类别相同，但你不能读取对方的私人资料。
下方仅是你本人获准保留的记忆数据；null 表示没有可用经历。没有提供的历史或个人总结不可凭空引用。
有记忆时可以采用、质疑或忽略自己的旧判断；不要求改变策略或一定成交。个人总结可能有错。
历史事件编号只在原时间线有效，不能直接用作本世 evidence_event_ids。
"""


def validate_estimate(value: dict[str, Any], rules: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"price", "reason", "uncertainty"}:
        raise ValueError("估计仅允许 price/reason/uncertainty")
    p = value["price"]
    if type(p) is not int or not rules["price_min"] <= p <= rules["price_max"] or (p - rules["price_min"]) % rules["price_step"]:
        raise ValueError("估计必须是公开价格网格内的整数")
    if any(not isinstance(value[k], str) or not value[k].strip() or len(value[k]) > 300 for k in ("reason", "uncertainty")):
        raise ValueError("估计依据和不确定性各为1至300字")


def estimate_reservation(client: OpenAICompatibleClient | None, world: TradeWorld, role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if role not in ROLES:
        raise PermissionError("未知估计角色")
    target = "seller" if role == "buyer" else "buyer"
    boundary = "最低" if target == "seller" else "最高"
    observation = world.observe(role)
    rules = world.contract_rules()
    messages = [{"role": "system", "content": (
        f"你是 {role}。本世谈判已结束，请独立估计对手 {target} 的客观接受边界。"
        f"固定合同为3天交付、30天保障：估计对手满足其世界效用下限、可以合法接受的{boundary}价格。"
        "这里询问世界规则的合法边界，不是其理想成交价、下一轮报价、谈判底价主张或是否愿意立即成交。"
        "只根据下方你本人有权读取的本世信息推断；你不知道对手未公开的效用和下限。"
        f"价格必须是{rules['price_min']}至{rules['price_max']}之间、步长{rules['price_step']}的整数。"
        "只输出JSON，严格包含 price（整数）、reason（简短依据）、uncertainty（简短不确定性），后两段各不超过300字。"
        "这是事后评测，不会反馈答案，也不影响任何后续谈判。")},
        {"role": "user", "content": json.dumps(observation, ensure_ascii=False, sort_keys=True)}]
    calls = []
    if client is None:
        value = {"price": rules["price_min"], "reason": "离线网格占位估计，不代表模型推断。", "uncertainty": "未调用模型。"}
    else:
        for attempt in range(2):
            try:
                raw, record = client.complete(messages)
            except ModelCallError as exc:
                for call in exc.call_records:
                    call.update(role=role, purpose="estimation", timeline_id=world.audit_timeline_id(AUDITOR))
                exc.call_records = calls + exc.call_records
                raise
            record.update(role=role, purpose="estimation", format_attempt=attempt, timeline_id=world.audit_timeline_id(AUDITOR))
            calls.append(record)
            try:
                value = json.loads(raw)
                validate_estimate(value, rules)
            except (TypeError, ValueError, KeyError) as exc:
                record["validation_error"] = str(exc)
                if attempt == 1:
                    raise AgentOutputError(f"接受边界估计无效: {exc}", calls) from exc
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"仅修正格式与价格网格，不存在答案提示：{exc}"}]
                continue
            break
    # 真值在模型调用之后才计算，且从不加入任何模型消息。
    truth = world.reservation_price(target, 3, 30, viewer=AUDITOR)
    if truth is None:
        raise ValueError("固定合同在当前网格没有可接受价格")
    return {"role": role, "target_role": target, "delivery_days": 3, "warranty_days": 30,
            **value, "true_boundary": truth, "absolute_error": abs(value["price"] - truth),
            "source": "agent_estimate" if client else "offline_fixture"}, calls


def metric(run: dict[str, Any] | None, name: str) -> float | None:
    if run is None:
        return None
    if name.endswith("_estimation_error"):
        role = name.split("_")[0]
        item = run.get("estimations", {}).get(role)
        return float(item["absolute_error"]) if item else None
    if "metrics" not in run:
        return None
    return float(run["metrics"]["deal" if name == "agreement_rate" else name])


def descriptive(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def paired_statistics(differences: list[float], *, samples: int = 10000, seed: int = 2026) -> dict[str, Any]:
    result = {**descriptive(differences), "differences": differences, "ci95": None}
    if len(differences) < 2:
        return result
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(samples))

    def percentile(q: float) -> float:
        index = (len(draws) - 1) * q
        lower = int(index)
        return draws[lower] + (draws[min(lower + 1, len(draws) - 1)] - draws[lower]) * (index - lower)

    result["ci95"] = [percentile(0.025), percentile(0.975)]
    return result


def ablation_statistics(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [{r["group"]: r for r in e["runs"]} for e in experiments]
    groups = {}
    for group in GROUPS:
        groups[group] = {name: descriptive([v for row in rows if (v := metric(row.get(group), name)) is not None]) for name in METRICS}
    contrasts = {}
    for treatment, reference in (("rewind_h", "control"), ("rewind_hr", "control"), ("rewind_hr", "rewind_h")):
        contrasts[f"{treatment}-minus-{reference}"] = {
            name: paired_statistics([a - b for row in rows
                                     if (a := metric(row.get(treatment), name)) is not None
                                     and (b := metric(row.get(reference), name)) is not None]) for name in METRICS}
    changes = {group: {role: paired_statistics([
        a - b for row in rows if (a := metric(row.get(group), f"{role}_estimation_error")) is not None
        and (b := metric(row.get("baseline"), f"{role}_estimation_error")) is not None]) for role in ROLES} for group in GROUPS}
    return {"groups": groups, "paired_contrasts": contrasts, "error_change_from_first_life": changes,
            "ci_method": "paired percentile bootstrap; resample independent repeats; B=10000; seed=2026; n<2 unavailable; exploratory, unadjusted for multiple comparisons"}
