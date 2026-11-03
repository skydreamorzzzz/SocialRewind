from __future__ import annotations

import copy
import json
import math
import unittest

from social_rewind import Action, DecisionNote, MockJudge, ReplayError, World
from social_rewind.judge import JudgeOutputError, validate_verdict
from social_rewind.models import TimelineMemory
from social_rewind.policy import choose_opening
from social_rewind.scenario import fictional_dubbing_scenario


def note(
    intent: str = "我将执行这项行动。",
    evidence_event_ids: tuple[int, ...] = (),
) -> DecisionNote:
    return DecisionNote(
        intent=intent,
        known_facts=("项目今天截止。",),
        uncertainty="我不确定对方的选择。",
        evidence_event_ids=evidence_event_ids,
    )


def action(kind: str, actor: str, **payload: object) -> Action:
    return Action(kind=kind, actor=actor, payload=payload, decision=note())


class InvalidJudge:
    def evaluate(self, request: dict[str, object]) -> dict[str, object]:
        return {
            "effective_hours": 999,
            "quality": "legendary",
            "reason": "越权输出",
            "evidence_ids": ["不存在"],
            "model_metadata": {"provider": "bad", "model": "bad", "call_id": "bad"},
        }


class WorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = fictional_dubbing_scenario()
        self.judge = MockJudge()
        self.world = World(self.scenario, self.judge)

    def propose_and_accept(self, world: World | None = None) -> str:
        world = world or self.world
        proposed = world.act(
            action(
                "propose_commitment",
                "producer",
                to="voice_actor",
                task_id="final_voice_take",
                amount=300,
                terms="第 1 天提交最终对白，目标质量 good。",
            )
        )
        self.assertTrue(proposed["accepted"])
        commitment_id = proposed["result"]["commitment_id"]
        accepted = world.act(
            action(
                "respond_commitment",
                "voice_actor",
                commitment_id=commitment_id,
                response="accept",
            )
        )
        self.assertTrue(accepted["accepted"])
        return commitment_id

    def submit(self, world: World | None = None) -> None:
        world = world or self.world
        result = world.act(
            action(
                "submit_work",
                "voice_actor",
                task_id="final_voice_take",
                claimed_hours=2,
                evidence=[
                    {
                        "id": "ev-clean-take",
                        "kind": "audio_review_note",
                        "description": "两句对白已重录，底噪与爆音检查通过。",
                    }
                ],
            )
        )
        self.assertTrue(result["accepted"])

    def complete_timeline(self) -> None:
        self.world.act(
            action("send_message", "producer", to="voice_actor", content="请确认今天能否完成重录。")
        )
        self.propose_and_accept()
        self.submit()
        result = self.world.act(action("advance_day", "producer"))
        self.assertTrue(result["accepted"])

    def test_private_negotiation_does_not_advance_day(self) -> None:
        before = self.world.audit_state("auditor")["day"]
        event = self.world.act(
            action("send_message", "producer", to="voice_actor", content="你今天可录多久？")
        )
        self.assertTrue(event["accepted"])
        self.assertEqual(before, self.world.audit_state("auditor")["day"])
        self.assertTrue(event["result"]["day_unchanged"])

    def test_observation_does_not_leak_other_private_information(self) -> None:
        secret_message = "仅限双方：嗓音状态需要保密。"
        self.world.act(
            action("send_message", "producer", to="voice_actor", content=secret_message)
        )
        director_view = json.dumps(self.world.observe("director"), ensure_ascii=False)
        actor_view = json.dumps(self.world.observe("voice_actor"), ensure_ascii=False)
        self.assertNotIn(secret_message, director_view)
        self.assertNotIn("今天最多适合连续录音 2 小时", director_view)
        self.assertNotIn("我将执行这项行动", director_view)
        self.assertIn(secret_message, actor_view)
        self.assertIn("今天最多适合连续录音 2 小时", actor_view)
        with self.assertRaises(PermissionError):
            self.world.audit_state("director")

    def test_overspend_is_rejected_without_state_change(self) -> None:
        before = self.world.audit_state("auditor")
        event = self.world.act(
            action(
                "propose_commitment",
                "producer",
                to="voice_actor",
                task_id="final_voice_take",
                amount=1001,
                terms="超预算提案",
            )
        )
        self.assertFalse(event["accepted"])
        self.assertEqual("insufficient_budget", event["result"]["refusal"]["code"])
        self.assertNotIn("state_diff", event)
        self.assertEqual([], self.world.export_events("auditor")[-1]["state_diff"])
        self.assertEqual(before, self.world.audit_state("auditor"))

    def test_commitment_can_be_proposed_and_accepted(self) -> None:
        commitment_id = self.propose_and_accept()
        state = self.world.audit_state("auditor")
        self.assertEqual("accepted", state["commitments"][commitment_id]["status"])
        self.assertEqual("assigned", state["tasks"]["final_voice_take"]["status"])
        self.assertEqual(700, state["budget"]["available"])
        self.assertEqual(0, state["budget"]["reserved"])
        self.assertEqual(300, state["budget"]["spent"])

    def test_rejected_commitment_releases_reserved_budget(self) -> None:
        proposed = self.world.act(
            action(
                "propose_commitment",
                "producer",
                to="voice_actor",
                task_id="final_voice_take",
                amount=300,
                terms="第 1 天完成重录。",
            )
        )
        commitment_id = proposed["result"]["commitment_id"]
        rejected = self.world.act(
            action(
                "respond_commitment",
                "voice_actor",
                commitment_id=commitment_id,
                response="reject",
            )
        )
        state = self.world.audit_state("auditor")
        self.assertTrue(rejected["accepted"])
        self.assertEqual("rejected", state["commitments"][commitment_id]["status"])
        self.assertEqual(1000, state["budget"]["available"])
        self.assertEqual(0, state["budget"]["reserved"])
        self.assertEqual("open", state["tasks"]["final_voice_take"]["status"])

    def test_daily_private_interaction_limit_is_enforced(self) -> None:
        for index in range(self.scenario.max_daily_interactions):
            event = self.world.act(
                action(
                    "send_message",
                    "producer",
                    to="director",
                    content=f"第 {index + 1} 次有限交互。",
                )
            )
            self.assertTrue(event["accepted"])
        before = self.world.audit_state("auditor")
        refused = self.world.act(
            action("send_message", "producer", to="director", content="超过上限的交互。")
        )
        self.assertFalse(refused["accepted"])
        self.assertEqual("interaction_limit", refused["result"]["refusal"]["code"])
        self.assertEqual(before, self.world.audit_state("auditor"))

    def test_judged_work_is_settled_and_traceable(self) -> None:
        self.propose_and_accept()
        self.submit()
        receipt = self.world.act(action("advance_day", "producer"))
        record = self.world.export_events("auditor")[-1]["result"]["judge_records"][0]
        state = self.world.audit_state("auditor")
        self.assertEqual(["ok"], receipt["result"]["settlements"])
        self.assertEqual("ok", record["status"])
        self.assertEqual("ev-clean-take", record["input"]["evidence"][0]["id"])
        self.assertEqual(["ev-clean-take"], record["output"]["evidence_ids"])
        self.assertEqual("mock", record["output"]["model_metadata"]["provider"])
        self.assertEqual("completed", state["tasks"]["final_voice_take"]["status"])
        self.assertEqual(1, self.judge.calls)

    def test_reset_restores_exact_s0_and_keeps_append_only_audit(self) -> None:
        s0 = self.world.audit_state("auditor")
        self.complete_timeline()
        event_count = len(self.world.export_events("auditor"))
        reset_event = self.world.reset()
        self.assertEqual(s0, self.world.audit_state("auditor"))
        self.assertEqual("reset", reset_event["kind"])
        self.assertEqual(event_count + 1, len(self.world.export_events("auditor")))

    def test_reset_hides_prior_timeline_from_agents_but_not_auditor(self) -> None:
        secret = "上一条时间线的私密谈判内容"
        old_receipt = self.world.act(
            action("send_message", "producer", to="voice_actor", content=secret)
        )
        fresh_world = World(self.scenario, MockJudge())
        s0 = fresh_world.audit_state("auditor")
        self.world.reset()

        current_producer_view = self.world.observe("producer")
        producer_view = json.dumps(current_producer_view, ensure_ascii=False)
        actor_events = self.world.export_events("voice_actor")
        audit_events = self.world.export_events("auditor")
        self.assertNotIn(secret, producer_view)
        self.assertNotIn(secret, json.dumps(actor_events, ensure_ascii=False))
        self.assertIn(secret, json.dumps(audit_events, ensure_ascii=False))
        self.assertEqual([], actor_events)
        self.assertNotIn("timeline_id", self.world.observe("producer"))
        self.assertEqual(fresh_world.observe("producer"), current_producer_view)
        self.assertEqual(s0, self.world.audit_state("auditor"))

        before = self.world.audit_state("auditor")
        refused = self.world.act(
            Action(
                kind="send_message",
                actor="producer",
                payload={"to": "director", "content": "尝试引用旧时间线。"},
                decision=note(evidence_event_ids=(old_receipt["event_id"],)),
            )
        )
        self.assertFalse(refused["accepted"])
        self.assertEqual("unobservable_evidence", refused["result"]["refusal"]["code"])
        self.assertEqual(before, self.world.audit_state("auditor"))

        self.world.act(
            action("send_message", "producer", to="director", content="当前时间线的新消息。")
        )
        events = self.world.export_events("auditor")
        replayed = World.replay(self.scenario, events)
        self.assertEqual(self.world.audit_state("auditor"), replayed.audit_state("auditor"))
        self.assertEqual(events, replayed.export_events("auditor"))

    def test_replay_reconstructs_exact_final_state_without_calling_judge(self) -> None:
        self.complete_timeline()
        final_state = self.world.audit_state("auditor")
        events = self.world.export_events("auditor")
        replayed = World.replay(self.scenario, events)
        self.assertEqual(final_state, replayed.audit_state("auditor"))
        self.assertEqual(events, replayed.export_events("auditor"))

    def test_external_memory_changes_next_policy_but_not_s0(self) -> None:
        s0 = self.world.audit_state("auditor")
        baseline = choose_opening("producer", None)
        memory = TimelineMemory(owner="producer", lesson="先问演员档期，再谈质量。")
        informed = choose_opening("producer", memory)
        self.assertNotEqual(baseline.payload, informed.payload)
        self.assertEqual("voice_actor", informed.payload["to"])
        self.assertEqual(s0, self.world.audit_state("auditor"))

    def test_action_receipt_hides_audit_diff_and_judge_details(self) -> None:
        self.propose_and_accept()
        self.submit()
        receipt = self.world.act(action("advance_day", "producer"))
        self.assertNotIn("state_diff", receipt)
        self.assertNotIn("judge_records", receipt["result"])
        self.assertEqual(["ok"], receipt["result"]["settlements"])

        visible_event = self.world.export_events("producer")[-1]
        self.assertNotIn("state_diff", visible_event)
        self.assertNotIn("result", visible_event)
        self.assertEqual(["ok"], visible_event["result_summary"]["settlements"])

        audit_event = self.world.export_events("auditor")[-1]
        self.assertIn("state_diff", audit_event)
        self.assertIn("judge_records", audit_event["result"])

    def test_decision_references_visible_event_and_marks_facts_as_self_reported(self) -> None:
        message = self.world.act(
            action("send_message", "producer", to="voice_actor", content="请确认档期。")
        )
        receipt = self.world.act(
            Action(
                kind="send_message",
                actor="voice_actor",
                payload={"to": "producer", "content": "今天可以录两小时。"},
                decision=note(evidence_event_ids=(message["event_id"],)),
            )
        )
        self.assertTrue(receipt["accepted"])
        decision = self.world.export_events("auditor")[-1]["decision"]
        self.assertEqual([message["event_id"]], decision["evidence_event_ids"])
        self.assertEqual("actor_self_report", decision["known_facts_source"])

    def test_invalid_judge_output_is_explicit_failure_without_score(self) -> None:
        world = World(self.scenario, InvalidJudge())
        self.propose_and_accept(world)
        self.submit(world)
        world.act(action("advance_day", "producer"))
        record = world.export_events("auditor")[-1]["result"]["judge_records"][0]
        task = world.audit_state("auditor")["tasks"]["final_voice_take"]
        self.assertEqual("failed", record["status"])
        self.assertEqual("JudgeOutputError", record["error"]["type"])
        self.assertEqual("evaluation_failed", task["status"])
        self.assertIsNone(task["quality"])
        self.assertIsNone(task["effective_hours"])
        replayed = World.replay(self.scenario, world.export_events("auditor"))
        self.assertEqual(world.audit_state("auditor"), replayed.audit_state("auditor"))

    def test_judge_rejects_non_finite_effective_hours(self) -> None:
        request = {
            "claimed_hours": 2,
            "evidence": [{"id": "ev-1"}],
        }
        base_verdict = {
            "quality": "good",
            "reason": "测试",
            "evidence_ids": ["ev-1"],
            "model_metadata": {"provider": "mock", "model": "test", "call_id": "1"},
        }
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(JudgeOutputError):
                validate_verdict(request, {"effective_hours": value, **base_verdict})

    def test_tampered_replay_fails_loudly(self) -> None:
        self.complete_timeline()
        events = self.world.export_events("auditor")
        tampered = copy.deepcopy(events)
        tampered[-1]["result"]["judge_records"][0]["output"]["quality"] = "poor"
        with self.assertRaises(ReplayError):
            World.replay(self.scenario, tampered)


if __name__ == "__main__":
    unittest.main()
