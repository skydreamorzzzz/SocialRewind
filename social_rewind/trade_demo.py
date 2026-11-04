from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleClient, OpenAICompatibleConfig
from .negotiation import (
    AgentTurn,
    LLMNegotiationAgent,
    NegotiationAgent,
    NegotiationRun,
    NegotiationRunner,
    ScriptedAgent,
    render_control_prompt,
    render_rebirth_prompt,
)
from .trade import AUDITOR, TradeMemory, TradeWorld, fictional_phone_trade


def _turn(
    kind: str,
    speech: str,
    fact: str,
    *,
    price: int | None = None,
    offer_id: str | None = None,
    evidence: tuple[int, ...] = (),
) -> AgentTurn:
    return AgentTurn(
        kind=kind,
        speech=speech,
        intent="我想在自己的价格约束内推进谈判。",
        known_facts=(fact,),
        uncertainty="我不确定对方下一步是否会让步。",
        price=price,
        offer_id=offer_id,
        evidence_event_ids=evidence,
    )


def _offline_agents(branch: int | str) -> tuple[ScriptedAgent, ScriptedAgent]:
    if branch in (1, "timeline_1", "control"):
        buyer_turns = [
            _turn("offer", "我先报 3400 元。", "挂牌价是 4200 元。", price=3400),
            _turn("offer", "我可以提高到 3600 元。", "卖方刚才正式报价 4000 元。", price=3600, evidence=(2,)),
            _turn("accept", "我明确接受你 3700 元的有效报价。", "当前有效报价 o0004 是 3700 元。", offer_id="o0004", evidence=(4,)),
        ]
        seller_turns = [
            _turn("offer", "3400 太低，我正式报价 4000 元。", "买方正式报价 3400 元。", price=4000, evidence=(1,)),
            _turn("offer", "我最后可以到 3700 元。", "买方已提高到 3600 元。", price=3700, evidence=(3,)),
        ]
    else:
        buyer_turns = [
            _turn("offer", "我们都记得上一世。这次我直接报 3550 元。", "上一世最终以 3700 元成交。", price=3550),
            _turn("offer", "我这次到 3650 元，希望更快结束。", "卖方这一世正式报价 3750 元。", price=3650, evidence=(2,)),
        ]
        seller_turns = [
            _turn("offer", "我也记得。我们少绕弯，我报 3750 元。", "买方这一世开价 3550 元。", price=3750, evidence=(1,)),
            _turn("accept", "我明确接受你 3650 元的有效报价。", "当前有效报价 o0003 是 3650 元。", offer_id="o0003", evidence=(3,)),
        ]
    return ScriptedAgent(buyer_turns), ScriptedAgent(seller_turns)


def _live_agents(client: OpenAICompatibleClient) -> tuple[NegotiationAgent, NegotiationAgent]:
    return LLMNegotiationAgent(client), LLMNegotiationAgent(client)


def _print_turn(label: int | str):
    def callback(role: str, turn: AgentTurn, receipt: dict[str, Any]) -> None:
        rounds = receipt["result"].get("rounds")
        detail = f" price={turn.price}" if turn.price is not None else ""
        if turn.offer_id is not None:
            detail += f" offer_id={turn.offer_id}"
        print(
            f"[{label}][Round {rounds}][{role}] {turn.kind}{detail}: {turn.speech} "
            f"| decision={turn.intent}"
        )

    return callback


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"[{label} Result] status={metrics['status']} price={metrics['deal_price']} "
        f"buyer_utility={metrics['buyer_utility']} seller_utility={metrics['seller_utility']} "
        f"rounds={metrics['rounds']} quality={metrics['nash_quality']:.4f} "
        f"efficiency={metrics['efficiency_discount']:.4f}"
    )


def _verify_replay(
    scenario, world: TradeWorld
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = world.export_events(AUDITOR)
    final_state = world.audit_state(AUDITOR)
    replayed = TradeWorld.replay(scenario, events)
    assert replayed.audit_state(AUDITOR) == final_state
    assert replayed.export_events(AUDITOR) == events
    return events, final_state


def _key_excerpts(run: NegotiationRun) -> list[dict[str, Any]]:
    excerpts: list[dict[str, Any]] = []
    for receipt in run.receipts:
        if receipt["kind"] not in ("offer", "accept", "reject", "exit"):
            continue
        excerpts.append(
            {
                "event_id": receipt["event_id"],
                "actor": receipt["actor"],
                "kind": receipt["kind"],
                "input": receipt["input"],
                "decision_intent": receipt["decision"]["intent"],
            }
        )
    return excerpts


def _model_config(run: NegotiationRun) -> dict[str, Any] | None:
    if not run.model_calls:
        return None
    configs = [
        {key: value for key, value in call["request"].items() if key != "messages"}
        for call in run.model_calls
    ]
    if any(config != configs[0] for config in configs[1:]):
        raise AssertionError("同一实验分支内的模型配置不一致")
    return configs[0]


def _print_comparison(runs: dict[str, NegotiationRun]) -> None:
    labels = ("Timeline 1", "Control", "Rewind")
    rows = (
        ("是否成交", lambda m: "是" if m["deal"] else "否"),
        ("成交价格", lambda m: m["deal_price"]),
        ("买方收益", lambda m: m["buyer_utility"]),
        ("卖方收益", lambda m: m["seller_utility"]),
        ("谈判轮数", lambda m: m["rounds"]),
        ("交易质量 Q", lambda m: f"{m['nash_quality']:.4f}"),
    )
    print("\n=== Comparison ===")
    print(f"{'指标':<14} | {'Timeline 1':<12} | {'Control':<12} | {'Rewind':<12}")
    print("-" * 62)
    for name, getter in rows:
        values = [str(getter(runs[label].metrics)) for label in labels]
        print(f"{name:<14} | {values[0]:<12} | {values[1]:<12} | {values[2]:<12}")
    print("\n[关键报价与决策说明]")
    for label in ("Control", "Rewind"):
        print(f"{label}:")
        for excerpt in _key_excerpts(runs[label]):
            print(
                f"  - {excerpt['actor']} {excerpt['kind']} {excerpt['input']} "
                f"| {excerpt['decision_intent']}"
            )


def run_demo(
    *,
    live: bool = False,
    client: OpenAICompatibleClient | None = None,
    max_rounds: int = 20,
    log_path: Path | None = None,
) -> dict[str, Any]:
    if live and client is None:
        raise ValueError("live=True 时必须显式提供模型客户端")
    scenario = fictional_phone_trade(max_rounds=max_rounds)
    world = TradeWorld(scenario)
    s0 = world.audit_state(AUDITOR)
    runner = NegotiationRunner()

    print("=== Timeline 1：第一世（无前世记忆） ===")
    buyer1, seller1 = _live_agents(client) if live else _offline_agents(1)
    run1 = runner.run(world, buyer1, seller1, on_turn=_print_turn("Timeline 1"))
    _print_metrics("Timeline 1", run1.metrics)

    buyer_memory = world.capture_memory("buyer")
    seller_memory = world.capture_memory("seller")
    memories = {"buyer": buyer_memory, "seller": seller_memory}

    print("\n=== Rewind：世界恢复 S0 ===")
    world.reset()
    assert world.audit_state(AUDITOR) == s0
    fresh = TradeWorld(scenario)
    assert world.observe("buyer") == fresh.observe("buyer")
    assert world.observe("seller") == fresh.observe("seller")
    print(f"S0 fingerprint={world.s0_fingerprint}；普通观察不含 Timeline 1 历史。")

    print("\n=== Timeline 2：第二世（双方共同重生） ===")
    for role, memory in memories.items():
        counterpart = "seller" if role == "buyer" else "buyer"
        prompt = render_rebirth_prompt(memory, counterpart)
        print(
            f"[{role} Rebirth Context] authorized_events={len(memory.visible_events)} "
            f"outcome={memory.outcome}"
        )
        print(prompt)
    buyer2, seller2 = _live_agents(client) if live else _offline_agents(2)
    run2 = runner.run(
        world,
        buyer2,
        seller2,
        memories=memories,
        on_turn=_print_turn("Timeline 2"),
    )
    _print_metrics("Timeline 2", run2.metrics)

    audit_events = world.export_events(AUDITOR)
    replayed = TradeWorld.replay(scenario, audit_events)
    assert replayed.audit_state(AUDITOR) == world.audit_state(AUDITOR)
    assert replayed.export_events(AUDITOR) == audit_events
    print("\n[Replay] 两世完整审计日志已确定性回放，未重新调用模型。")
    print("[Research Note] 两次谈判仅展示机制，不证明策略学习有效。")

    output = {
        "fictional": True,
        "mode": "live_llm" if live else "offline_scripted",
        "scenario": asdict(scenario),
        "s0_fingerprint": world.s0_fingerprint,
        "timeline_1": _run_dict(run1),
        "rewind_memories": {role: memory.to_dict() for role, memory in memories.items()},
        "timeline_2": _run_dict(run2),
        "audit_events": audit_events,
        "final_state": world.audit_state(AUDITOR),
        "replay_verified": True,
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[Log] 已写入 {log_path}")
    return output


def run_compare_demo(
    *,
    live: bool = False,
    client: OpenAICompatibleClient | None = None,
    max_rounds: int = 20,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """从同一第一世派生无记忆 Control 与双边记忆 Rewind。"""

    if live and client is None:
        raise ValueError("live=True 时必须显式提供模型客户端")
    scenario = fictional_phone_trade(max_rounds=max_rounds)
    timeline1_world = TradeWorld(scenario)
    control_world = TradeWorld(scenario)
    rewind_world = TradeWorld(scenario)
    worlds = (timeline1_world, control_world, rewind_world)
    s0_states = [world.audit_state(AUDITOR) for world in worlds]
    s0_fingerprints = [world.s0_fingerprint for world in worlds]
    if not (s0_states[0] == s0_states[1] == s0_states[2]):
        raise AssertionError("Timeline 1、Control 与 Rewind 的 S0 不一致")
    if len(set(s0_fingerprints)) != 1:
        raise AssertionError("三个实验世界的 S0 fingerprint 不一致")

    runner = NegotiationRunner()
    print("=== Timeline 1：共同的第一世 ===")
    buyer1, seller1 = _live_agents(client) if live else _offline_agents("timeline_1")
    run1 = runner.run(
        timeline1_world,
        buyer1,
        seller1,
        on_turn=_print_turn("Timeline 1"),
    )
    _print_metrics("Timeline 1", run1.metrics)
    memories = {
        "buyer": timeline1_world.capture_memory("buyer"),
        "seller": timeline1_world.capture_memory("seller"),
    }

    control_context = render_control_prompt()
    print("\n=== Control：相同 S0，无可用前世记忆 ===")
    print(control_context)
    buyer_control, seller_control = (
        _live_agents(client) if live else _offline_agents("control")
    )
    control_run = runner.run(
        control_world,
        buyer_control,
        seller_control,
        session_context=control_context,
        on_turn=_print_turn("Control"),
    )
    _print_metrics("Control", control_run.metrics)

    print("\n=== Rewind：相同 S0，双方携带各自合法前世记忆 ===")
    for role, memory in memories.items():
        counterpart = "seller" if role == "buyer" else "buyer"
        print(
            f"[{role} Rebirth Context] authorized_events={len(memory.visible_events)} "
            f"outcome={memory.outcome}"
        )
        print(render_rebirth_prompt(memory, counterpart))
    buyer_rewind, seller_rewind = (
        _live_agents(client) if live else _offline_agents("rewind")
    )
    rewind_run = runner.run(
        rewind_world,
        buyer_rewind,
        seller_rewind,
        memories=memories,
        on_turn=_print_turn("Rewind"),
    )
    _print_metrics("Rewind", rewind_run.metrics)

    runs = {
        "Timeline 1": run1,
        "Control": control_run,
        "Rewind": rewind_run,
    }
    _print_comparison(runs)

    audit_worlds: dict[str, dict[str, Any]] = {}
    for label, world in (
        ("timeline_1", timeline1_world),
        ("control", control_world),
        ("rewind", rewind_world),
    ):
        events, final_state = _verify_replay(scenario, world)
        audit_worlds[label] = {
            "events": events,
            "final_state": final_state,
            "replay_verified": True,
        }
    print("\n[Replay] 三个世界均已从各自事件日志确定性回放，未重新调用模型。")

    configs = [_model_config(run1), _model_config(control_run), _model_config(rewind_run)]
    model_config_equal = configs[0] == configs[1] == configs[2]
    if live and not model_config_equal:
        raise AssertionError("三个真实模型分支的模型配置不一致")
    output = {
        "fictional": True,
        "mode": "compare_live_llm" if live else "compare_offline_scripted",
        "scenario": asdict(scenario),
        "design_note": (
            "Control 与 Rewind 共享同一 Timeline 1 和相同 S0；比较同时包含记忆内容与轮回叙事情境差异，"
            "单次结果不支持因果或策略学习结论。"
        ),
        "s0_equal": True,
        "s0_fingerprint": s0_fingerprints[0],
        "model_config_equal": model_config_equal,
        "model_config": configs[0],
        "timeline_1": _run_dict(run1),
        "control": {
            **_run_dict(control_run),
            "context": control_context,
            "memory_injected": False,
            "key_excerpts": _key_excerpts(control_run),
        },
        "rewind": {
            **_run_dict(rewind_run),
            "memory_injected": True,
            "key_excerpts": _key_excerpts(rewind_run),
        },
        "rewind_memories": {role: memory.to_dict() for role, memory in memories.items()},
        "audit_worlds": audit_worlds,
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[Log] 已写入 {log_path}")
    print("[Research Note] 单次对照只提供初步观察，不证明轮回记忆带来因果改善。")
    return output


def _run_dict(run: NegotiationRun) -> dict[str, Any]:
    return {
        "timeline_id": run.timeline_id,
        "metrics": run.metrics,
        "model_calls": list(run.model_calls),
        "receipts": list(run.receipts),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行双人轮回交易谈判演示")
    parser.add_argument("--live", action="store_true", help="使用真实 OpenAI-compatible LLM")
    parser.add_argument("--compare", action="store_true", help="运行 Timeline 1/Control/Rewind 对照实验")
    parser.add_argument("--api-base", default=os.getenv("SOCIALREWIND_API_BASE"))
    parser.add_argument("--model", default=os.getenv("SOCIALREWIND_MODEL"))
    parser.add_argument("--api-key-env", default="SOCIALREWIND_API_KEY")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    client = None
    if args.live:
        api_key = os.getenv(args.api_key_env)
        if not api_key or not args.api_base or not args.model:
            parser.error("--live 需要 --api-base、--model 以及 --api-key-env 指向的非空密钥")
        extra_body = {"thinking": {"type": "disabled"}} if args.disable_thinking else {}
        client = OpenAICompatibleClient(
            OpenAICompatibleConfig(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                extra_body=extra_body,
            )
        )
    runner = run_compare_demo if args.compare else run_demo
    runner(live=args.live, client=client, max_rounds=args.max_rounds, log_path=args.log)


if __name__ == "__main__":
    main()
