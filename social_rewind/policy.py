from __future__ import annotations

from .models import Action, DecisionNote, TimelineMemory


def choose_opening(actor: str, memory: TimelineMemory | None) -> Action:
    """最小脚本策略：展示外部经验如何改变下一轮选择。"""

    if actor != "producer":
        raise ValueError("示例策略只定义了 producer")
    if memory is None:
        target = "director"
        content = "请确认你对最终对白的最低质量要求。"
        basis = ("项目今天截止",)
    else:
        if memory.owner != actor:
            raise ValueError("不能读取其他角色的跨时间线经验")
        target = "voice_actor"
        content = "我记得先确认档期能减少返工；你今天可录多久？"
        basis = ("项目今天截止", f"上轮经验：{memory.lesson}")
    return Action(
        kind="send_message",
        actor=actor,
        payload={"to": target, "content": content},
        decision=DecisionNote(
            intent="我想先取得最影响交付的信息。",
            known_facts=basis,
            uncertainty="我尚不确定对方会如何回应。",
        ),
    )
