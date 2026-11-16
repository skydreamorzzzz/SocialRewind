from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path

from social_rewind.experiment import run_experiments, verify_results
from social_rewind.llm import ModelCallError
from social_rewind.models import Action, DecisionNote
from social_rewind.negotiation import AgentOutputError, summarize_experience
from social_rewind.trade import AUDITOR, TradeWorld, phone_negotiation, scenario_from_dict


def act(kind, role="buyer", **payload):
    return Action(kind, role, {"text": "公开发言", **payload}, DecisionNote("我想协商。", ("本人自述",), "对方需求未知"))


class PhoneClient:
    def __init__(self):
        self.requests = []

    def complete(self, messages):
        self.requests.append(copy.deepcopy(messages))
        if "你刚作为" in messages[0]["content"]:
            memory = json.loads(messages[1]["content"])
            value = {"observations": ["我观察到双方交换了公开发言与方案。"],
                     "beliefs": f"{memory['role']}_PRIVATE_BELIEF：对方可能完全不在意价格，尚未验证。",
                     "next_strategy": "我下一世可能先询问交付需求，也可能维持原报价。", "evidence_event_ids": [1]}
        else:
            observation = json.loads(messages[-1]["content"].rsplit("\n", 1)[1])
            role = observation["role"]
            if len(observation["events"]) < 2:
                action = {"kind": "message"}
            elif role == "buyer":
                action = {"kind": "offer", "price": 3800, "delivery_days": 3, "warranty_days": 30}
            else:
                action = {"kind": "accept", "offer_id": observation["active_offer"]["id"]}
            value = {"decision": {"intent": f"{role}_PRIVATE_DECISION", "known_facts": ["本人自述"],
                                   "uncertainty": "对方需求未知"}, "speech": "公开协商", "action": action}
        raw = json.dumps(value, ensure_ascii=False)
        return raw, {"request": {"model": "fake", "temperature": 0.3, "max_tokens": 1200,
                                  "messages": copy.deepcopy(messages)}, "response": {"content": raw}}


class PhoneCaseTests(unittest.TestCase):
    def test_all_nine_discrete_combinations(self):
        scenario = phone_negotiation()
        for days, bu, su in ((1, 120, -220), (3, 0, 0), (7, -250, 20)):
            for warranty, bw, sw in ((0, 0, 0), (30, 140, -50), (90, 170, -170)):
                with self.subTest(days=days, warranty=warranty):
                    world = TradeWorld(scenario)
                    self.assertTrue(world.act(act("offer", price=3800, delivery_days=days, warranty_days=warranty))["accepted"])
                    self.assertTrue(world.act(act("accept", "seller", offer_id="o0001"))["accepted"])
                    metrics = world.metrics(AUDITOR)
                    self.assertEqual(4100 - 3800 + bu + bw, metrics["buyer_utility"])
                    self.assertEqual(3800 - 3300 + su + sw, metrics["seller_utility"])
                    self.assertEqual(800 + bu + bw + su + sw, metrics["social_welfare"])
                    self.assertEqual(days == 3 and warranty == 30, metrics["pareto_efficient"])

    def test_minimum_utility_and_boundary_are_objective_not_aspiration(self):
        scenario = phone_negotiation()
        scenario = replace(scenario, contract=replace(scenario.contract, buyer=replace(scenario.contract.buyer, minimum_utility=300)))
        world = TradeWorld(scenario)
        before = world.audit_state(AUDITOR)
        refused = world.act(act("offer", price=4300, delivery_days=1, warranty_days=30))
        self.assertFalse(refused["accepted"])  # 收益为60，虽为正但低于私人下限300。
        self.assertEqual(before, world.audit_state(AUDITOR))
        limit = next(t for t in world.observe("buyer")["private"]["price_limits_by_terms"]
                     if t["delivery_days"] == 1 and t["warranty_days"] == 30)
        self.assertEqual(4050, limit["maximum_price"])
        self.assertEqual([], world.export_events("seller"))

    def test_role_cards_and_tables_are_private_and_do_not_include_joint_optimum(self):
        world = TradeWorld(phone_negotiation())
        buyer = json.dumps(world.observe("buyer"), ensure_ascii=False)
        seller = json.dumps(world.observe("seller"), ensure_ascii=False)
        self.assertIn("担心隐藏故障", buyer)
        self.assertNotIn("迁移个人数据", buyer)
        self.assertIn("迁移个人数据", seller)
        self.assertNotIn("担心隐藏故障", seller)
        self.assertNotIn("pareto", buyer)
        self.assertNotIn("social_welfare", seller)
        self.assertEqual(phone_negotiation(), scenario_from_dict(json.loads(json.dumps(asdict(phone_negotiation())))))

    def test_invalid_discrete_tables_fail_loudly(self):
        scenario = phone_negotiation()
        for pref in (replace(scenario.contract.buyer, delivery_utilities={"1": 120}),
                     replace(scenario.contract.buyer, warranty_utilities={"0": 0, "30": float("nan"), "90": 170}),
                     replace(scenario.contract.buyer, minimum_utility=-1)):
            with self.assertRaises(ValueError):
                TradeWorld(replace(scenario, contract=replace(scenario.contract, buyer=pref)))

    def test_summary_can_be_wrong_but_receives_only_authorized_evidence(self):
        world = TradeWorld(phone_negotiation())
        world.act(act("offer", price=3800, delivery_days=3, warranty_days=30))
        world.act(act("accept", "seller", offer_id="o0001"))
        client = PhoneClient()
        before = world.audit_state(AUDITOR)
        summary, calls = summarize_experience(client, world.capture_memory("buyer"))
        self.assertIn("完全不在意价格", summary["beliefs"])
        self.assertEqual("agent_self_report", summary["source"])
        self.assertEqual("reflection", calls[0]["purpose"])
        self.assertNotIn("迁移个人数据", json.dumps(client.requests, ensure_ascii=False))
        self.assertEqual(before, world.audit_state(AUDITOR))

    def test_third_life_inherits_personal_summaries_without_control_or_counterparty_leak(self):
        client = PhoneClient()
        with redirect_stdout(StringIO()):
            result = run_experiments([phone_negotiation()], live=True, client=client)
        experiment = result["experiments"][0]
        self.assertEqual(10, sum(c.get("purpose") == "reflection" for r in experiment["runs"] for c in r["model_calls"]))
        for run in experiment["runs"]:
            self.assertEqual({"buyer", "seller"}, set(run["reflections"]))
            for role, opposite in (("buyer", "seller"), ("seller", "buyer")):
                negotiation = next(c for c in run["model_calls"] if c["role"] == role and c["purpose"] == "negotiation")
                messages = negotiation["request"]["messages"]
                self.assertEqual(2, len(messages))
                text = json.dumps(messages, ensure_ascii=False)
                self.assertNotIn(opposite + "_PRIVATE_BELIEF", text)
                self.assertNotIn(opposite + "_PRIVATE_DECISION", text)
                if run["group"] == "control":
                    self.assertNotIn("PRIVATE_BELIEF", text)
                    self.assertFalse(run["memories"])
                if run["group"] == "rewind":
                    self.assertIn(role + "_PRIVATE_BELIEF", text)
                    if run["life"] == 3:
                        memory = run["memories"][role]
                        self.assertEqual(2, memory["personal_summary"]["source_timeline_id"])
                        self.assertEqual(1, memory["previous_lives"][0]["personal_summary"]["source_timeline_id"])
        verify_results(result)
        tampered = copy.deepcopy(result)
        next(r for r in tampered["experiments"][0]["runs"] if r["group"] == "rewind")["memories"]["buyer"]["personal_summary"]["beliefs"] = "篡改总结"
        with self.assertRaisesRegex(AssertionError, "未授权"):
            verify_results(tampered)
        stripped = copy.deepcopy(result)
        del next(r for r in stripped["experiments"][0]["runs"] if r["group"] == "rewind")["memories"]["buyer"]["personal_summary"]
        with self.assertRaisesRegex(AssertionError, "未授权"):
            verify_results(stripped)

    def test_summary_failure_preserves_confirmed_world_and_complete_calls(self):
        class FailureClient(PhoneClient):
            def complete(self, messages):
                if "你刚作为" in messages[0]["content"]:
                    raise ModelCallError("模拟总结失败", [{"request": {"messages": copy.deepcopy(messages)}, "error": "timeout"}])
                return super().complete(messages)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "failed.json"
            with redirect_stdout(StringIO()), self.assertRaises(ModelCallError):
                run_experiments([phone_negotiation()], live=True, client=FailureClient(), log_path=log)
            result = json.loads(log.read_text(encoding="utf-8"))
            run = result["experiments"][0]["runs"][0]
            self.assertEqual("reflection", run["error_phase"])
            self.assertEqual("failed", run["status"])
            self.assertEqual({}, run["reflections"])
            self.assertEqual(5, len(run["model_calls"]))
            self.assertEqual(list(range(1, 6)), [c["call_sequence"] for c in run["model_calls"]])
            self.assertTrue(run["metrics"]["deal"])
            self.assertEqual("deal", result["experiments"][0]["worlds"]["baseline"]["final_state"]["status"])
            verify_results(result)

    def test_invalid_summary_reference_is_rejected_not_fabricated(self):
        class InvalidClient(PhoneClient):
            def complete(self, messages):
                raw, record = super().complete(messages[:2])
                value = json.loads(raw)
                value["evidence_event_ids"] = [999]
                raw = json.dumps(value)
                record["response"]["content"] = raw
                return raw, record
        world = TradeWorld(phone_negotiation())
        world.act(act("exit"))
        with self.assertRaises(AgentOutputError) as caught:
            summarize_experience(InvalidClient(), world.capture_memory("buyer"))
        self.assertEqual(2, len(caught.exception.call_records))

    def test_persisted_v2_success_and_failure_remain_replayable(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        for name in ("v2_live.json", "v2_live_attempt1_failed.json"):
            with self.subTest(name=name):
                verify_results(json.loads((examples / name).read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
