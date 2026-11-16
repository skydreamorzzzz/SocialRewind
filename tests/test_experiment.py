from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from social_rewind.experiment import run_experiments, summarize, verify_results
from social_rewind.llm import ModelCallError, OpenAICompatibleClient, OpenAICompatibleConfig
from social_rewind.models import Action, DecisionNote
from social_rewind.negotiation import AgentOutputError, LLMNegotiationAgent
from social_rewind.timeline_view import write_timeline_page
from social_rewind.trade import AUDITOR, ContractPreferences, TradeWorld, multidimensional_trade, scenario_from_dict


def action(kind: str, actor: str = "buyer", **payload) -> Action:
    return Action(kind, actor, {"text": "公开发言", **payload},
                  DecisionNote(f"{actor} PRIVATE_INTENT", ("角色自述",), "对方偏好未知"))


class AutonomousFakeClient:
    """模拟当前观察驱动的 Agent；不读取世界、对手私人信息或外部日志。"""
    def __init__(self) -> None:
        self.requests = []

    def complete(self, messages):
        self.requests.append(copy.deepcopy(messages))
        observation = json.loads(messages[-1]["content"].rsplit("\n", 1)[1])
        role = observation["role"]
        if len(observation["events"]) < 2:
            selected = {"kind": "message"}
        elif role == "buyer":
            selected = {"kind": "offer", "price": 3600, "delivery_days": 1, "warranty_days": 30}
        else:
            selected = {"kind": "accept", "offer_id": observation["active_offer"]["id"]}
        raw = json.dumps({"decision": {"intent": f"{role} PRIVATE_INTENT", "known_facts": ["自述"],
                                       "uncertainty": "对方偏好未知"},
                          "speech": "PUBLIC_OFFER" if selected["kind"] == "offer" else "公开回应",
                          "action": selected}, ensure_ascii=False)
        return raw, {"request": {"model": "fake", "temperature": 0.3, "max_tokens": 1200,
                                  "messages": copy.deepcopy(messages)}, "response": {"content": raw}}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.scenario = multidimensional_trade()
        self.world = TradeWorld(self.scenario)

    def deal(self, price, delivery, warranty):
        offer = self.world.act(action("offer", price=price, delivery_days=delivery, warranty_days=warranty))
        self.assertTrue(offer["accepted"])
        receipt = self.world.act(action("accept", "seller", offer_id="o0001"))
        self.assertTrue(receipt["accepted"])
        return self.world.metrics(AUDITOR)

    def test_contract_utilities_and_pareto_frontier(self):
        m = self.deal(3600, 1, 90)
        self.assertEqual((570, 210, 780), (m["buyer_utility"], m["seller_utility"], m["social_welfare"]))
        self.assertTrue(m["pareto_efficient"])
        self.assertEqual({"price": 3600, "delivery_days": 1, "warranty_days": 90}, m["deal_terms"])

    def test_inefficient_contract_can_be_improved_by_trading_terms(self):
        m = self.deal(3600, 3, 0)
        self.assertEqual((160, 340, 500), (m["buyer_utility"], m["seller_utility"], m["social_welfare"]))
        self.assertFalse(m["pareto_efficient"])

    def test_multidimensional_preference_can_justify_price_above_base_value(self):
        m = self.deal(4000, 1, 90)
        self.assertEqual(170, m["buyer_utility"])
        self.assertEqual(610, m["seller_utility"])

    def test_invalid_terms_are_atomic_private_refusals(self):
        for extra in ({"price": 3555, "delivery_days": 1, "warranty_days": 0},
                      {"price": 3600, "delivery_days": True, "warranty_days": 0},
                      {"price": 3600, "delivery_days": 2, "warranty_days": 0},
                      {"price": 3600, "delivery_days": 1, "warranty_days": 365},
                      {"price": 3600}, {"price": float("nan"), "delivery_days": 1, "warranty_days": 0},
                      {"price": 4500, "delivery_days": 7, "warranty_days": 0}):
            with self.subTest(extra=extra):
                world = TradeWorld(self.scenario)
                before = world.audit_state(AUDITOR)
                receipt = world.act(action("offer", **extra))
                self.assertFalse(receipt["accepted"])
                self.assertEqual(before, world.audit_state(AUDITOR))
                self.assertEqual([], world.export_events("seller"))
                self.assertNotIn("state_diff", receipt)

    def test_accept_checks_own_utility_and_cannot_revive_old_offer(self):
        self.world.act(action("offer", price=3000, delivery_days=1, warranty_days=0))
        before = self.world.audit_state(AUDITOR)
        refused = self.world.act(action("accept", "seller", offer_id="o0001"))
        self.assertEqual("negative_own_utility", refused["result"]["refusal"]["code"])
        self.assertEqual(before, self.world.audit_state(AUDITOR))
        self.world.act(action("offer", "seller", price=3600, delivery_days=1, warranty_days=30))
        self.world.act(action("offer", price=3500, delivery_days=1, warranty_days=30))
        refused = self.world.act(action("accept", "seller", offer_id="o0002"))
        self.assertEqual("inactive_offer", refused["result"]["refusal"]["code"])

    def test_private_preferences_and_joint_metrics_are_never_in_agent_views(self):
        self.deal(3600, 1, 30)
        for role, opposite in (("buyer", "seller"), ("seller", "buyer")):
            view = self.world.observe(role)
            text = json.dumps(view)
            self.assertNotIn(opposite + "_utility", text)
            self.assertNotIn("social_welfare", text)
            self.assertNotIn("pareto_efficient", text)
            self.assertNotIn("contract_config", text)
            self.assertNotIn(opposite + " PRIVATE_INTENT", text)
            self.assertEqual(asdict(getattr(self.scenario.contract, role)), view["private"]["preferences"])

    def test_scenario_json_roundtrip_and_invalid_grid(self):
        restored = scenario_from_dict(json.loads(json.dumps(asdict(self.scenario))))
        self.assertEqual(self.scenario, restored)
        for contract in (replace(self.scenario.contract, price_step=0),
                         replace(self.scenario.contract, delivery_days=()),
                         replace(self.scenario.contract, warranty_days=(0, 0)),
                         replace(self.scenario.contract, buyer=ContractPreferences(3900, float("inf"), 3)),
                         replace(self.scenario.contract, price_max=1000000, price_step=1)):
            with self.assertRaises(ValueError):
                TradeWorld(replace(self.scenario, contract=contract))
        other = TradeWorld(replace(self.scenario, contract=replace(self.scenario.contract, buyer=ContractPreferences(3900, 50, 3))))
        self.assertNotEqual(self.world.s0_fingerprint, other.s0_fingerprint)

    def test_no_agreement_outcome_and_unavailable_surplus(self):
        config = replace(self.scenario.contract, buyer=ContractPreferences(1000, 70, 0))
        world = TradeWorld(replace(self.scenario, contract=config))
        world.act(action("exit"))
        m = world.metrics(AUDITOR)
        self.assertFalse(m["deal"])
        self.assertEqual(0, m["social_welfare"])
        self.assertTrue(m["pareto_efficient"])  # 无可行正收益合同时，退出点也可在 Pareto 前沿。


class ExperimentTests(unittest.TestCase):
    def run_quiet(self, *args, **kwargs):
        with redirect_stdout(StringIO()):
            return run_experiments(*args, **kwargs)

    def test_three_lives_control_and_rewind_use_same_s0_and_legal_cumulative_memory(self):
        fake = AutonomousFakeClient()
        result = self.run_quiet([multidimensional_trade()], live=True, client=fake)
        experiment = result["experiments"][0]
        self.assertEqual(5, len(experiment["runs"]))
        first = experiment["runs"][0]
        for run in experiment["runs"]:
            self.assertEqual(first["initial_state"], run["initial_state"])
            for role in ("buyer", "seller"):
                first_call = next(c for c in run["model_calls"] if c["role"] == role)
                self.assertEqual(2, len(first_call["request"]["messages"]))
                system = first_call["request"]["messages"][0]["content"]
                self.assertNotIn(("seller" if role == "buyer" else "buyer") + " PRIVATE_INTENT", system)
                if run["group"] == "control":
                    self.assertFalse(run["memory_injected"])
                    self.assertEqual({}, run["memories"])
                    self.assertNotIn("PUBLIC_OFFER", system)
                    self.assertNotIn("PRIVATE_INTENT", system)
                if run["group"] == "rewind":
                    memory = run["memories"][role]
                    self.assertEqual(run["life"] - 2, len(memory["previous_lives"]))
                    self.assertEqual(run["life"] - 1, memory["source_timeline_id"])
                    self.assertIn("PUBLIC_OFFER", system)
                    if run["life"] == 2:
                        own_events = memory["visible_events"]
                        self.assertEqual(len(first["receipts"]), len(own_events))
                    else:
                        self.assertEqual(1, memory["previous_lives"][0]["source_timeline_id"])
        with patch.object(OpenAICompatibleClient, "complete", side_effect=AssertionError("回放不应联网")):
            verify_results(result)
        tampered = copy.deepcopy(result)
        rewind = next(r for r in tampered["experiments"][0]["runs"] if r["group"] == "rewind")
        rewind["memories"]["buyer"]["visible_events"][1]["decision"] = {"intent": "未经授权的对手私人说明"}
        with self.assertRaisesRegex(AssertionError, "未授权"):
            verify_results(tampered)

    def test_repeat_statistics_separate_profiles_and_have_correct_denominators(self):
        result = self.run_quiet([multidimensional_trade(p) for p in ("urgent", "warranty")], repeats=2)
        self.assertEqual(4, len(result["experiments"]))
        self.assertEqual(10, len(result["summary"]))
        self.assertTrue(all(row["n"] == 2 for row in result["summary"]))
        self.assertTrue(all(row["buyer_utility"]["sample_std"] == 0 for row in result["summary"]))
        self.assertEqual(["rewind", "control"], result["experiments"][1]["branch_order"])
        # 不成交的收益也进入均值，不能仅统计成功交易。
        modified = copy.deepcopy(result["experiments"][:2])
        modified[0]["runs"][0]["metrics"].update(deal=False, buyer_utility=0, seller_utility=0, social_welfare=0)
        baseline = next(row for row in summarize(modified) if row["group"] == "baseline")
        self.assertEqual(0.5, baseline["agreement_rate"])

    def test_single_live_negotiation_and_missing_client(self):
        with self.assertRaises(ValueError):
            self.run_quiet([multidimensional_trade()], live=True)
        result = self.run_quiet([multidimensional_trade()], lives=1, live=True, client=AutonomousFakeClient())
        self.assertEqual(1, len(result["experiments"][0]["runs"]))
        self.assertTrue(result["replay_verified"])

    def test_model_failure_retains_all_prior_calls_and_replayable_partial_world(self):
        class BrokenClient(AutonomousFakeClient):
            def complete(self, messages):
                if self.requests:
                    raise ModelCallError("模拟服务失败", [{"request": {"messages": copy.deepcopy(messages)}, "error": "timeout"}])
                return super().complete(messages)

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "failed.json"
            with self.assertRaises(ModelCallError):
                self.run_quiet([multidimensional_trade()], live=True, client=BrokenClient(), log_path=log)
            saved = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual("failed", saved["status"])
            run = saved["experiments"][0]["runs"][0]
            self.assertEqual(2, len(run["model_calls"]))
            self.assertEqual(1, len(run["receipts"]))
            self.assertNotIn("metrics", run)
            self.assertEqual([], saved["summary"])
            verify_results(saved)

    def test_invalid_model_format_does_not_fabricate_results(self):
        class InvalidClient:
            def complete(self, messages):
                return "not JSON", {"request": {"messages": copy.deepcopy(messages)}}
        with self.assertRaises(AgentOutputError) as raised:
            self.run_quiet([multidimensional_trade()], live=True, client=InvalidClient())
        self.assertEqual(2, len(raised.exception.call_records))

    def test_model_prompt_has_own_formula_and_complete_contract_schema(self):
        world = TradeWorld(multidimensional_trade())
        client = AutonomousFakeClient()
        agent = LLMNegotiationAgent(client)
        agent.start("buyer", world.scenario_context("buyer"), None)
        agent.respond(world.observe("buyer"))
        system = client.requests[0][0]["content"]
        self.assertIn("value-price-delay*(delivery_days-1)", system)
        self.assertIn('"delivery_days":整数', system)
        self.assertNotIn('"value": 3300', system)
        self.assertNotIn("price-value+delay", system)
        raw = client.requests[0][-1]["content"]
        self.assertIn('"delivery_days": 1', raw)

    def test_own_price_boundaries_match_grid_and_refuse_observed_bad_offer(self):
        world = TradeWorld(multidimensional_trade())
        private = world.observe("buyer")["private"]
        limit = next(t for t in private["price_limits_by_terms"] if t["delivery_days"] == 1 and t["warranty_days"] == 30)
        self.assertEqual(3950, limit["maximum_price"])
        refused = world.act(action("offer", price=4000, delivery_days=1, warranty_days=30))
        self.assertEqual("negative_own_utility", refused["result"]["refusal"]["code"])
        for role in ("buyer", "seller"):
            view = world.observe(role)
            pref = getattr(world._scenario.contract, role)
            for row in view["private"]["price_limits_by_terms"]:
                threshold = pref.value - pref.delay * (row["delivery_days"] - 1) + pref.warranty * row["warranty_days"]
                valid = [p for p in range(3000, 4501, 50) if (p <= threshold if role == "buyer" else p >= threshold)]
                key = "maximum_price" if role == "buyer" else "minimum_price"
                self.assertEqual((max(valid) if role == "buyer" else min(valid)) if valid else None, row[key])

    def test_html_is_standalone_and_escapes_model_html(self):
        result = self.run_quiet([multidimensional_trade()])
        payload = '</script><script>alert("xss")</script>'
        result["experiments"][0]["worlds"]["baseline"]["events"][0]["input"]["text"] = payload
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "timeline.html"
            write_timeline_page(result, page)
            html = page.read_text(encoding="utf-8")
            self.assertNotIn(payload, html)
            self.assertIn("\\u003c/script\\u003e", html)
            self.assertNotIn("fetch(", html)
            self.assertIn("textContent", html)

    def test_http_failure_record_never_contains_authorization(self):
        import urllib.error
        config = OpenAICompatibleConfig("https://example.invalid", "FAKE_SECRET", "fake")
        client = OpenAICompatibleClient(config)
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(ModelCallError) as raised:
                client.complete([{"role": "user", "content": "test"}])
        encoded = json.dumps(raised.exception.call_records)
        self.assertNotIn("FAKE_SECRET", encoded)
        self.assertNotIn("Authorization", encoded)


if __name__ == "__main__":
    unittest.main()
