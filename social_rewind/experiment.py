"""v2 最小实验入口：多维合同、共享前世的多轮重置、批量统计与本地审计页。"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import hashlib
from itertools import permutations
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import ModelCallError, OpenAICompatibleClient, OpenAICompatibleConfig
from .negotiation import (
    AgentOutputError, AgentTurn, IllegalAgentAction, LLMNegotiationAgent,
    NegotiationRunner, ScriptedAgent, render_control_prompt, summarize_experience, validate_summary,
)
from .trade import AUDITOR, ROLES, DiscretePreferences, TradeMemory, TradeScenario, TradeWorld, phone_negotiation, scenario_from_dict
from .evaluation import GROUPS, RESET_FRAME, ablation_statistics, estimate_reservation, validate_estimate


def _scripted_agents(scenario: TradeScenario) -> tuple[ScriptedAgent, ScriptedAgent]:
    """仅用于离线机制测试。使用同一脚本，不模拟或证明记忆改善。"""
    world = TradeWorld(scenario)
    c = scenario.contract
    feasible = [
        (p, d, w) for p in range(c.price_min, c.price_max + 1, c.price_step)
        for d in c.delivery_days for w in c.warranty_days
        if all(world._utility(r, p, d, w) >= getattr(getattr(c, r), "minimum_utility", 0) for r in ROLES)
    ]

    def turn(kind: str, speech: str, **kwargs: Any) -> AgentTurn:
        return AgentTurn(kind, speech, "我按离线测试脚本检查沟通与合同流程。",
                         ("本运行使用确定性的测试脚本，不代表策略探索。",), "对方的真实偏好未知。", **kwargs)

    if not feasible:
        return ScriptedAgent([turn("exit", "离线无可行合同，结束测试。")]), ScriptedAgent([])
    p, d, w = feasible[len(feasible) // 2]
    return (
        ScriptedAgent([turn("message", "能讨论交付时间和售后保障吗？"),
                       turn("offer", "这是一个完整合同方案。", price=p, delivery_days=d, warranty_days=w)]),
        ScriptedAgent([turn("message", "可以，请提出完整方案。"),
                       turn("accept", "我接受当前合同。", offer_id="o0001")]),
    )


def summarize(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for experiment in experiments:
        for run in experiment["runs"]:
            if run["status"] == "completed":
                key = (experiment["scenario"]["scenario_id"], run["group"], run["life"])
                buckets.setdefault(key, []).append(run["metrics"])
    rows = []
    for (scenario, group, life), metrics in sorted(buckets.items()):
        row: dict[str, Any] = {"scenario": scenario, "group": group, "life": life, "n": len(metrics),
                               "agreement_rate": statistics.fmean(m["deal"] for m in metrics)}
        for field in ("buyer_utility", "seller_utility", "social_welfare", "pareto_efficient", "rounds"):
            values = [m[field] for m in metrics]
            row[field] = {"mean": statistics.fmean(values),
                          "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0}
        rows.append(row)
    return rows


def save_results(output: dict[str, Any], log_path: Path, html_path: Path | None) -> None:
    from .timeline_view import write_timeline_page

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if html_path is not None:
        write_timeline_page(output, html_path)


def verify_results(output: dict[str, Any]) -> None:
    """从磁盘加载的结果也可验证；不访问 API。"""
    if output.get("design") == "memory_ablation":
        verify_ablation_results(output)
        return
    for experiment in output["experiments"]:
        scenario = scenario_from_dict(experiment["scenario"])
        initial = TradeWorld(scenario)
        if initial.s0_fingerprint != experiment["s0_fingerprint"]:
            raise AssertionError("S0 fingerprint 不一致")
        for branch in experiment["worlds"].values():
            baseline_events = experiment["worlds"]["baseline"]["events"]
            if branch["events"][:len(baseline_events)] != baseline_events:
                raise AssertionError("对照分支没有共享同一第一世")
            replayed = TradeWorld.replay(scenario, branch["events"])
            if replayed.audit_state(AUDITOR) != branch["final_state"]:
                raise AssertionError("最终状态与回放不一致")
        for run in experiment["runs"]:
            if run["initial_state"] != initial.audit_state(AUDITOR):
                raise AssertionError("实验分支未从相同 S0 开始")
            replayed = TradeWorld.replay(scenario, experiment["worlds"][run["group"]]["events"][:run["audit_end"]])
            if run["status"] == "completed" and replayed.metrics(AUDITOR) != run["metrics"]:
                raise AssertionError("时间线指标与回放不一致")
            if run.get("summaries_enabled") and run["status"] == "completed" and set(run["reflections"]) != set(ROLES):
                raise AssertionError("本世必须保留双方个人总结")
            for role, summary in run.get("reflections", {}).items():
                validate_summary(summary, replayed.capture_memory(role))
            if run["group"] != "rewind":
                if run["memories"] or run["memory_injected"]:
                    raise AssertionError("Control 或第一世不应注入记忆")
                continue
            events = experiment["worlds"]["rewind"]["events"][:run["audit_end"]]
            if set(run["memories"]) != set(ROLES) or not run["memory_injected"]:
                raise AssertionError("Rewind 必须给双方注入各自记忆")
            for source_life in range(1, run["life"]):
                cutoff = next(i for i, e in enumerate(events) if e["kind"] == "reset" and e["timeline_id"] == source_life)
                source = TradeWorld.replay(scenario, events[:cutoff])
                for role in ROLES:
                    saved = copy.deepcopy(run["memories"][role])
                    earlier = saved.pop("previous_lives")
                    if len(earlier) != run["life"] - 2:
                        raise AssertionError("累计记忆世数不一致")
                    snapshot = earlier[source_life - 1] if source_life < run["life"] - 1 else saved
                    expected = source.capture_memory(role).to_dict()
                    expected.pop("previous_lives")
                    source_group = "baseline" if source_life == 1 else "rewind"
                    source_run = next(r for r in experiment["runs"] if r["group"] == source_group and r["life"] == source_life)
                    expected["personal_summary"] = source_run.get("reflections", {}).get(role)
                    if "personal_summary" not in snapshot and not source_run.get("summaries_enabled"):
                        # 仅历史 v2 日志允许缺少总结；新版不能通过删字段绕过继承检查。
                        expected.pop("personal_summary")
                    if snapshot != expected:
                        raise AssertionError("记忆包含未授权内容或与实际历史不一致")


def run_experiments(
    scenarios: list[TradeScenario], *, lives: int = 3, repeats: int = 1,
    live: bool = False, client: OpenAICompatibleClient | None = None,
    log_path: Path | None = None, html_path: Path | None = None,
) -> dict[str, Any]:
    if lives < 1 or repeats < 1:
        raise ValueError("lives 和 repeats 至少为 1；lives=1 表示单次谈判")
    if live and client is None:
        raise ValueError("真实实验必须显式提供模型客户端")
    if not scenarios or any(s.contract is None for s in scenarios):
        raise ValueError("v2 需要至少一组三维合同场景")
    output: dict[str, Any] = {
        "schema_version": 3, "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live_llm" if live else "offline_scripted", "status": "running",
        "config": {"lives": lives, "repeats": repeats,
                   "design": "同一第一世；Control 每世无历史或总结，Rewind 累积本人授权历史与个人总结；每世新会话；重复间不共享经验",
                   "statistics": "按场景/组/世汇总；未成交收益为0；Pareto效率基于完整有限合同网格和退出点；不跨偏好混合",
                   "limitation": "记忆可用性与轮回叙事情境同时改变；样本不保证显著性或因果改善"},
        "experiments": [], "summary": [],
    }
    runner = NegotiationRunner()
    model_config = None
    for scenario in scenarios:
        for repeat in range(1, repeats + 1):
            first_world = TradeWorld(scenario)
            s0 = first_world.audit_state(AUDITOR)
            experiment: dict[str, Any] = {
                "id": f"{scenario.scenario_id}/repeat-{repeat}", "repeat": repeat,
                "scenario": asdict(scenario), "s0_fingerprint": first_world.s0_fingerprint,
                "runs": [], "worlds": {},
            }
            output["experiments"].append(experiment)

            def execute(world: TradeWorld, group: str, life: int, memories: dict[str, TradeMemory] | None = None) -> None:
                nonlocal model_config
                if world.audit_state(AUDITOR) != s0 or any(world.export_events(r) for r in ROLES):
                    raise AssertionError("重启必须恢复同一 S0 且普通观察无历史")
                context = render_control_prompt() if group == "control" else None
                agents = (LLMNegotiationAgent(client), LLMNegotiationAgent(client)) if live else _scripted_agents(scenario)
                record: dict[str, Any] = {
                    "group": group, "life": life, "status": "running", "initial_state": copy.deepcopy(s0),
                    "memory_injected": bool(memories), "memories": {r: m.to_dict() for r, m in (memories or {}).items()},
                    "session_context": context, "model_calls": [], "receipts": [],
                    "summaries_enabled": isinstance(scenario.contract.buyer, DiscretePreferences), "reflections": {},
                }
                experiment["runs"].append(record)
                try:
                    run = runner.run(world, *agents, memories=memories, session_context=context)
                    record.update(status="completed", timeline_id=run.timeline_id, metrics=run.metrics,
                                  model_calls=list(run.model_calls), receipts=list(run.receipts))
                    if record["summaries_enabled"]:
                        for role in ROLES:
                            try:
                                summary, calls = summarize_experience(client if live else None, world.capture_memory(role))
                            except (AgentOutputError, ModelCallError) as exc:
                                exc.call_records = record["model_calls"] + exc.call_records
                                exc.receipts = record["receipts"]
                                record["error_phase"] = "reflection"
                                raise
                            record["reflections"][role] = summary
                            record["model_calls"].extend(calls)
                    for sequence, call in enumerate(record["model_calls"], start=1):
                        call["call_sequence"] = sequence
                        call.setdefault("purpose", "negotiation")
                        cfg = {k: v for k, v in call["request"].items() if k != "messages"}
                        if model_config is not None and model_config != cfg:
                            raise AssertionError("同一实验的模型采样配置发生变化")
                        model_config = cfg
                    output["config"]["model_request"] = model_config
                    m = run.metrics
                    print(f"{experiment['id']} {group} life={life}: {m['status']} terms={m['deal_terms']} "
                          f"buyer={m['buyer_utility']} seller={m['seller_utility']} welfare={m['social_welfare']} "
                          f"pareto={m['pareto_efficient']} rounds={m['rounds']}", flush=True)
                except (AgentOutputError, IllegalAgentAction, ModelCallError) as exc:
                    record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                                  model_calls=exc.call_records, receipts=getattr(exc, "receipts", []))
                    for sequence, call in enumerate(record["model_calls"], start=1):
                        call["call_sequence"] = sequence
                        call.setdefault("purpose", "negotiation")
                    output["status"] = "failed"
                    raise
                finally:
                    events = world.export_events(AUDITOR)
                    record["audit_end"] = len(events)
                    experiment["worlds"][group] = {"events": events, "final_state": world.audit_state(AUDITOR)}
                    verify_results(output)
                    output["summary"] = summarize(output["experiments"])
                    if log_path is not None:
                        save_results(output, log_path, html_path)

            execute(first_world, "baseline", 1)
            first_events = first_world.export_events(AUDITOR)
            # 交替组运行顺序，所有重复均保留；分支从同一第一世事件精确复制。
            order = ("control", "rewind") if repeat % 2 else ("rewind", "control")
            experiment["branch_order"] = list(order)
            for group in order if lives > 1 else ():
                world = TradeWorld.replay(scenario, first_events)
                history: dict[str, list[dict[str, Any]]] = {r: [] for r in ROLES}
                for life in range(2, lives + 1):
                    memories = None
                    if group == "rewind":
                        source_group = "baseline" if life == 2 else "rewind"
                        source_run = next(r for r in experiment["runs"] if r["group"] == source_group and r["life"] == life - 1)
                        current = {r: replace(world.capture_memory(r), personal_summary=source_run["reflections"].get(r)) for r in ROLES}
                        memories = {r: replace(m, previous_lives=tuple(copy.deepcopy(history[r]))) for r, m in current.items()}
                        for role, memory in current.items():
                            snapshot = memory.to_dict()
                            snapshot.pop("previous_lives")
                            history[role].append(snapshot)
                    world.reset()
                    execute(world, group, life, memories)
    output["status"] = "completed"
    output["replay_verified"] = True
    if log_path is not None:
        save_results(output, log_path, html_path)
    print("\n统计（未成交计0收益；各配置分开；std为重复间样本标准差）：")
    for row in output["summary"]:
        print(f"{row['scenario']} {row['group']} life={row['life']} n={row['n']} "
              f"Agreement={row['agreement_rate']:.2f} Buyer={row['buyer_utility']['mean']:.1f} "
              f"Seller={row['seller_utility']['mean']:.1f} Welfare={row['social_welfare']['mean']:.1f} "
              f"Pareto={row['pareto_efficient']['mean']:.2f} Rounds={row['rounds']['mean']:.1f}")
    return output


def verify_ablation_results(output: dict[str, Any]) -> None:
    for experiment in output["experiments"]:
        scenario = scenario_from_dict(experiment["scenario"])
        initial = TradeWorld(scenario)
        if scenario != phone_negotiation() or initial.s0_fingerprint != experiment["s0_fingerprint"]:
            raise AssertionError("冻结案例或 S0 被改变")
        base_events = experiment["worlds"]["baseline"]["events"]
        base_world = TradeWorld.replay(scenario, base_events)
        baseline = experiment["runs"][0]
        for run in experiment["runs"]:
            if run["initial_state"] != initial.audit_state(AUDITOR):
                raise AssertionError("分支初始状态不一致")
            branch = experiment["worlds"][run["group"]]
            world = TradeWorld.replay(scenario, branch["events"])
            if world.audit_state(AUDITOR) != branch["final_state"] or ("metrics" in run and world.metrics(AUDITOR) != run["metrics"]):
                raise AssertionError("世界或指标回放不一致")
            if branch["events"][:len(base_events)] != base_events:
                raise AssertionError("分支不是来自同一第一世")
            expected_memories = {}
            if run["group"] in ("rewind_h", "rewind_hr"):
                for role in ROLES:
                    memory = base_world.capture_memory(role)
                    if run["group"] == "rewind_hr":
                        memory = replace(memory, personal_summary=baseline["reflections"][role])
                    expected_memories[role] = memory.to_dict()
            if run["memories"] != expected_memories or run["memory_injected"] != bool(expected_memories):
                raise AssertionError("消融记忆包含未授权历史、Reflection 或事后评测")
            if run["group"] != "baseline" and run["session_context"] != RESET_FRAME:
                raise AssertionError("三组重启叙事不一致")
            for role, summary in run["reflections"].items():
                validate_summary(summary, world.capture_memory(role))
            for role, estimation in run["estimations"].items():
                validate_estimate({k: estimation[k] for k in ("price", "reason", "uncertainty")}, world.contract_rules())
                target = "seller" if role == "buyer" else "buyer"
                truth = world.reservation_price(target, 3, 30, viewer=AUDITOR)
                if (estimation["role"] != role or estimation["target_role"] != target
                        or (estimation["delivery_days"], estimation["warranty_days"]) != (3, 30)
                        or estimation["true_boundary"] != truth or estimation["absolute_error"] != abs(estimation["price"] - truth)):
                    raise AssertionError("估计真值、归属或误差不一致")
                if estimation["source"] == "agent_estimate":
                    valid_calls = [c for c in run["model_calls"] if c.get("role") == role and c.get("purpose") == "estimation"
                                   and "validation_error" not in c and "error" not in c]
                    if not valid_calls or json.loads(valid_calls[-1]["response"]["content"]) != {k: estimation[k] for k in ("price", "reason", "uncertainty")}:
                        raise AssertionError("保存的估计不是模型原始输出")
            if run["status"] == "completed" and (set(run["estimations"]) != set(ROLES) or set(run["reflections"]) != set(ROLES)):
                raise AssertionError("完整场次缺少评测或 Reflection")
            # 按保存的请求检查新会话、相同基础 Prompt 和估计的授权输入，不重新请求模型。
            for role in ROLES:
                context = initial.scenario_context(role)
                memory = None
                if role in expected_memories:
                    memory = base_world.capture_memory(role)
                    if run["group"] == "rewind_hr":
                        memory = replace(memory, personal_summary=baseline["reflections"][role])
                if run["group"] != "baseline":
                    context["shared_reset_frame"] = RESET_FRAME
                agent = LLMNegotiationAgent(None)
                agent.start(role, context, memory)
                calls = [c for c in run["model_calls"] if c.get("role") == role and c["purpose"] == "negotiation"]
                if calls and (len(calls[0]["request"]["messages"]) != 2 or any(c["request"]["messages"][0] != agent._messages[0] for c in calls)):
                    raise AssertionError("Agent 会话或冻结 Prompt 不一致")
                for call in run["model_calls"]:
                    if call.get("role") == role and call["purpose"] == "estimation":
                        expected = json.dumps(world.observe(role), ensure_ascii=False, sort_keys=True)
                        if call["request"]["messages"][1] != {"role": "user", "content": expected}:
                            raise AssertionError("估计调用包含超出本世授权观察的信息")
    if "ablation_statistics" in output and ablation_statistics(output["experiments"]) != output["ablation_statistics"]:
        raise AssertionError("派生统计与原始配对数据不一致")


def run_ablation(*, repeats: int = 10, live: bool = False, client: OpenAICompatibleClient | None = None,
                 log_path: Path | None = None, html_path: Path | None = None) -> dict[str, Any]:
    if repeats < 1 or (live and client is None):
        raise ValueError("repeats 必须为正，真实运行必须提供客户端")
    scenario = phone_negotiation()
    output: dict[str, Any] = {
        "schema_version": 4, "design": "memory_ablation", "mode": "live_llm" if live else "offline_scripted",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "config": {"requested_repeats": repeats, "shared_reset_frame": RESET_FRAME,
                   "case_sha256": hashlib.sha256(json.dumps(asdict(scenario), ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                   "estimation_contract": {"delivery_days": 3, "warranty_days": 30},
                   "posthoc_only": True, "ci": "paired bootstrap percentile 95%, B=10000, seed=2026; exploratory"},
        "experiments": [], "summary": [],
    }
    runner = NegotiationRunner()
    model_config = None
    fatal = False

    def checkpoint(final: bool = False) -> None:
        runs = [r for e in output["experiments"] for r in e["runs"]]
        output["counts"] = {"requested_repeats": repeats, "attempted_repeats": len(output["experiments"]),
                            "complete_repeats": sum(len(e["runs"]) == 4 and all(r["status"] == "completed" for r in e["runs"]) for e in output["experiments"]),
                            "attempted_runs": len(runs), "completed_runs": sum(r["status"] == "completed" for r in runs),
                            "failed_runs": sum(r["status"] == "failed" for r in runs),
                            "model_calls": sum(len(r["model_calls"]) for r in runs)}
        output["summary"] = summarize(output["experiments"])
        if final:
            output["ablation_statistics"] = ablation_statistics(output["experiments"])
        if log_path is not None:
            save_results(output, log_path, html_path)

    for repeat in range(1, repeats + 1):
        first = TradeWorld(scenario)
        experiment = {"id": f"phone-ablation/repeat-{repeat}", "repeat": repeat, "scenario": asdict(scenario),
                      "s0_fingerprint": first.s0_fingerprint, "runs": [], "worlds": {}}
        output["experiments"].append(experiment)

        def execute(world: TradeWorld, group: str, memories: dict[str, TradeMemory] | None = None) -> dict[str, Any]:
            nonlocal model_config, fatal
            if world.audit_state(AUDITOR) != TradeWorld(scenario).audit_state(AUDITOR) or any(world.export_events(r) for r in ROLES):
                raise AssertionError("分支没有从同一 S0 开始")
            record = {"group": group, "life": 1 if group == "baseline" else 2, "status": "running",
                      "initial_state": world.audit_state(AUDITOR), "memory_injected": bool(memories),
                      "memories": {r: m.to_dict() for r, m in (memories or {}).items()}, "reflections": {}, "estimations": {},
                      "model_calls": [], "receipts": [], "session_context": None if group == "baseline" else RESET_FRAME}
            experiment["runs"].append(record)
            phase = "negotiation"
            try:
                agents = (LLMNegotiationAgent(client), LLMNegotiationAgent(client)) if live else _scripted_agents(scenario)
                run = runner.run(world, *agents, memories=memories, shared_reset_frame=record["session_context"])
                record.update(metrics=run.metrics, model_calls=list(run.model_calls), receipts=list(run.receipts), timeline_id=run.timeline_id)
                # 先独立生成 Reflection，再做事后估计；评测从不污染 Reflection 或后续记忆。
                for phase in ("reflection", "estimation"):
                    for role in ROLES:
                        try:
                            if phase == "reflection":
                                value, calls = summarize_experience(client if live else None, world.capture_memory(role))
                                record["reflections"][role] = value
                            else:
                                value, calls = estimate_reservation(client if live else None, world, role)
                                record["estimations"][role] = value
                        except (AgentOutputError, ModelCallError) as exc:
                            exc.call_records = record["model_calls"] + exc.call_records
                            exc.receipts = record["receipts"]
                            raise
                        record["model_calls"].extend(calls)
                record["status"] = "completed"
            except (AgentOutputError, IllegalAgentAction, ModelCallError) as exc:
                record.update(status="failed", error_phase=phase, error={"type": type(exc).__name__, "message": str(exc)},
                              model_calls=exc.call_records, receipts=getattr(exc, "receipts", []))
                fatal = any(c.get("error", {}).get("http_status") in (401, 402, 403, 429) for c in exc.call_records if isinstance(c.get("error"), dict))
            finally:
                for sequence, call in enumerate(record["model_calls"], start=1):
                    call["call_sequence"] = sequence
                    call.setdefault("purpose", "negotiation")
                    cfg = {k: v for k, v in call["request"].items() if k != "messages"}
                    if model_config is not None and cfg != model_config:
                        raise AssertionError("组间模型采样配置不一致")
                    model_config = cfg
                output["config"]["model_request"] = model_config
                events = world.export_events(AUDITOR)
                record["audit_end"] = len(events)
                experiment["worlds"][group] = {"events": events, "final_state": world.audit_state(AUDITOR)}
                verify_ablation_results(output)
                checkpoint()
            print(f"repeat={repeat} {group} {record['status']} "
                  f"metrics={record.get('metrics')} estimates={record['estimations']}", flush=True)
            return record

        baseline = execute(first, "baseline")
        if fatal:
            break
        if baseline["status"] != "completed":
            continue
        order = list(tuple(permutations(GROUPS))[(repeat - 1) % 6])
        experiment["branch_order"] = order
        for group in order:
            memories = None
            if group != "control":
                memories = {role: first.capture_memory(role) for role in ROLES}
                if group == "rewind_hr":
                    memories = {role: replace(memory, personal_summary=baseline["reflections"][role]) for role, memory in memories.items()}
            world = TradeWorld.replay(scenario, first.export_events(AUDITOR))
            world.reset()
            execute(world, group, memories)
            if fatal:
                break
        if fatal:
            break
    output["status"] = "completed" if all(len(e["runs"]) == 4 and all(r["status"] == "completed" for r in e["runs"]) for e in output["experiments"]) and len(output["experiments"]) == repeats else "partial"
    output["replay_verified"] = True
    checkpoint(final=True)
    print("ABLATION_COUNTS", json.dumps(output["counts"], ensure_ascii=False), flush=True)
    print_ablation_statistics(output)
    return output


def print_ablation_statistics(output: dict[str, Any]) -> None:
    stats = output["ablation_statistics"]
    print("三组统计：mean ± sample_std (n)；误差分别为买方/卖方估计对手边界")
    for group, values in stats["groups"].items():
        print(group, " | ".join(f"{key}: {item['mean']} ± {item['sample_std']} (n={item['n']})" for key, item in values.items()))
    print("配对差值：mean [95% CI]；误差负值表示更准，n<2 的 CI 不可估计")
    for contrast, values in stats["paired_contrasts"].items():
        print(contrast, " | ".join(f"{key}: {item['mean']} {item['ci95']} (n={item['n']})" for key, item in values.items()))
    print("第一世→第二世估计误差变化：", json.dumps(stats["error_change_from_first_life"], ensure_ascii=False))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SocialRewind v2：自主三维谈判与多世对照")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="共同第一世与三个第二世记忆消融分支")
    parser.add_argument("--lives", type=int, default=3, help="总世数；1表示单次谈判")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--api-base", default=os.getenv("SOCIALREWIND_API_BASE"))
    parser.add_argument("--model", default=os.getenv("SOCIALREWIND_MODEL"))
    parser.add_argument("--api-key-env", default="SOCIALREWIND_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, help="仅在服务支持时使用；不保证确定性采样")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--log", type=Path, default=Path("runs/v2.json"))
    parser.add_argument("--html", type=Path, default=Path("runs/v2.html"))
    parser.add_argument("--replay", type=Path, help="只验证已有v2日志并重新生成本地HTML，不联网")
    args = parser.parse_args()
    if args.log.resolve() == args.html.resolve() or (args.replay and args.replay.resolve() == args.html.resolve()):
        parser.error("HTML 不能覆盖 JSON 日志")
    if args.replay:
        from .timeline_view import write_timeline_page
        output = json.loads(args.replay.read_text(encoding="utf-8"))
        verify_results(output)
        write_timeline_page(output, args.html)
        if output.get("design") == "memory_ablation":
            print_ablation_statistics(output)
        print(f"精确回放通过；页面：{args.html.resolve()}")
        return
    scenarios = [phone_negotiation(args.max_rounds)]
    client = None
    if args.live:
        key = os.getenv(args.api_key_env)
        if not key or not args.api_base or not args.model:
            parser.error("--live 需要 API base、model 和密钥环境变量；不会回退脚本")
        extra: dict[str, Any] = {"thinking": {"type": "disabled"}} if args.disable_thinking else {}
        if args.seed is not None:
            extra["seed"] = args.seed
        client = OpenAICompatibleClient(OpenAICompatibleConfig(
            api_base=args.api_base, api_key=key, model=args.model, temperature=args.temperature,
            max_tokens=args.max_tokens, extra_body=extra))
    if args.ablation:
        if args.max_rounds != 20:
            parser.error("消融实验冻结 max_rounds=20")
        run_ablation(repeats=args.repeats, live=args.live, client=client, log_path=args.log, html_path=args.html)
    else:
        run_experiments(scenarios, lives=args.lives, repeats=args.repeats, live=args.live,
                        client=client, log_path=args.log, html_path=args.html)
    print(f"日志：{args.log.resolve()}\n本地审计页面：{args.html.resolve()}")


if __name__ == "__main__":
    main()
