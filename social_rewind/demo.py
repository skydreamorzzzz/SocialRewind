from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .judge import MockJudge
from .models import Action, DecisionNote, TimelineMemory
from .policy import choose_opening
from .scenario import fictional_dubbing_scenario
from .world import World


def _note(
    intent: str,
    fact: str,
    uncertainty: str,
    evidence_event_ids: tuple[int, ...] = (),
) -> DecisionNote:
    return DecisionNote(
        intent=intent,
        known_facts=(fact,),
        uncertainty=uncertainty,
        evidence_event_ids=evidence_event_ids,
    )


def _act(world: World, kind: str, actor: str, payload: dict[str, object], note: DecisionNote) -> dict:
    event = world.act(Action(kind=kind, actor=actor, payload=payload, decision=note))
    if not event["accepted"]:
        raise RuntimeError(event["result"]["refusal"])
    return event


def run_demo(log_path: Path | None = None) -> None:
    scenario = fictional_dubbing_scenario()
    world = World(scenario, MockJudge())
    s0 = world.audit_state("auditor")
    producer_observation = world.observe("producer")
    actor_observation = world.observe("voice_actor")
    print(f"[S0] 第 {s0['day']} 天，可用预算 {s0['budget']['available']}，案例明确标记为虚构。")

    _act(
        world,
        "send_message",
        "producer",
        {"to": "voice_actor", "content": "发布窗口今天关闭；你能否用两小时完成重录？"},
        _note(
            "我想先确认关键档期。",
            producer_observation["public_facts"][1],
            "我不知道她今天的档期。",
        ),
    )
    reply = _act(
        world,
        "send_message",
        "voice_actor",
        {"to": "producer", "content": "我今天可以安排两小时，明天没有档期。"},
        _note(
            "我想明确自己的可用档期。",
            actor_observation["self"]["private_facts"][0],
            "我不确定两小时能否达到目标质量。",
        ),
    )
    print(f"[交涉] 私信已送达；模拟日仍为 {world.observe('producer')['day']}。")

    proposed = _act(
        world,
        "propose_commitment",
        "producer",
        {
            "to": "voice_actor",
            "task_id": "final_voice_take",
            "amount": 300,
            "terms": "今天完成两小时重录，目标质量 good。",
        },
        _note(
            "我愿意用项目预算换取按时交付。",
            "演员在可见消息中自述今天可安排两小时。",
            "Judge 尚未评价成品。",
            (reply["event_id"],),
        ),
    )
    commitment_id = proposed["result"]["commitment_id"]
    accepted = _act(
        world,
        "respond_commitment",
        "voice_actor",
        {"commitment_id": commitment_id, "response": "accept"},
        _note(
            "我接受可在安全工时内完成的承诺。",
            "制作人提出了 300 元的两小时录音承诺。",
            "成品质量仍需复核。",
            (proposed["event_id"],),
        ),
    )
    print(f"[承诺] {commitment_id} 已接受，300 元从预留转为已支出。")

    submitted = _act(
        world,
        "submit_work",
        "voice_actor",
        {
            "task_id": "final_voice_take",
            "claimed_hours": 2,
            "evidence": [
                {
                    "id": "ev-clean-take",
                    "kind": "audio_review_note",
                    "description": "两句对白已重录，底噪与爆音检查通过。",
                }
            ],
        },
        _note(
            "我提交两小时内完成的最终对白。",
            "我已接受这项录音承诺。",
            "主观质量由 Judge 判定。",
            (accepted["event_id"],),
        ),
    )
    _act(
        world,
        "advance_day",
        "producer",
        {},
        _note(
            "我明确结束今天并触发结算。",
            "执行者提交了我可见的工作证据。",
            "Judge 可能判定需要返工。",
            (submitted["event_id"],),
        ),
    )
    final_state = world.audit_state("auditor")
    timeline = world.export_events("auditor")
    verdict = timeline[-1]["result"]["judge_records"][0]["output"]
    print(
        f"[结算] 进入第 {final_state['day']} 天；质量={verdict['quality']}，"
        f"有效工时={verdict['effective_hours']}，任务={final_state['tasks']['final_voice_take']['status']}。"
    )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            for event in timeline:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"[日志] 已写入仅含虚构数据的 {log_path}。")

    world.reset()
    assert world.audit_state("auditor") == s0
    print("[重置] 世界硬状态与 S0 完全相同；既有审计事件仍只追加保留。")

    replayed = World.replay(scenario, timeline)
    assert replayed.audit_state("auditor") == final_state
    print("[回放] 未调用在线 Judge；最终状态逐字段一致。")

    baseline = choose_opening("producer", None)
    memory = TimelineMemory("producer", "先确认演员档期能减少无效谈判。")
    informed = choose_opening("producer", memory)
    assert world.audit_state("auditor") == s0
    print(f"[外部经验] 下轮首问从 {baseline.payload['to']} 改为 {informed.payload['to']}，S0 未被改写。")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行可审查社会世界的最小端到端演示")
    parser.add_argument("--log", type=Path, help="可选：写出完整审计 JSONL")
    args = parser.parse_args()
    run_demo(args.log)


if __name__ == "__main__":
    main()
