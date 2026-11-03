from __future__ import annotations

from .models import Scenario


def fictional_dubbing_scenario() -> Scenario:
    """完全虚构的配音社案例，不含真实访谈或个人资料。"""

    return Scenario(
        scenario_id="fictional-dubbing-club-v1",
        title="虚构案例：星桥配音社短片交付",
        fictional=True,
        public_facts=(
            "本案例全部人物、组织与项目均为虚构数据。",
            "团队需在第 1 天结束前完成最终对白，否则错过联合发布窗口。",
            "导演追求至少 good 的质量，制作人同时受 1000 元项目预算约束。",
        ),
        roles={
            "producer": {
                "name": "制作人林岚（虚构）",
                "goal": "按时交付且不超预算",
                "permissions": ["propose_budget_commitment", "advance_time"],
                "availability": {"1": 2, "2": 2},
            },
            "director": {
                "name": "导演周澄（虚构）",
                "goal": "获得至少 good 的表演质量",
                "permissions": [],
                "availability": {"1": 4, "2": 2},
            },
            "voice_actor": {
                "name": "配音演员许遥（虚构）",
                "goal": "完成项目但避免嗓音过度负荷",
                "permissions": [],
                "availability": {"1": 3, "2": 0},
            },
        },
        private_facts={
            "producer": ("备用宣发预算只有 200 元，不能挪作制作支出。",),
            "director": ("粗剪中有两句情绪层次不足，希望完整重录。",),
            "voice_actor": ("今天最多适合连续录音 2 小时，明天没有档期。",),
        },
        project_budget=1000,
        tasks={
            "final_voice_take": {
                "title": "最终对白录制",
                "deadline_day": 1,
                "target_quality": "good",
            }
        },
        max_daily_interactions=8,
    )
