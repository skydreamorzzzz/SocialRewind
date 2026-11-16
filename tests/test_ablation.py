from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from test_phone_case import PhoneClient
from social_rewind.evaluation import GROUPS, RESET_FRAME, ablation_statistics, estimate_reservation, paired_statistics, validate_estimate
from social_rewind.experiment import run_ablation, verify_results
from social_rewind.llm import ModelCallError
from social_rewind.negotiation import AgentOutputError
from social_rewind.trade import AUDITOR, TradeWorld, phone_negotiation


class AblationClient(PhoneClient):
    def complete(self, messages):
        if "客观接受边界" in messages[0]["content"]:
            self.requests.append(copy.deepcopy(messages))
            role = json.loads(messages[1]["content"])["role"]
            raw = json.dumps({"price": 3650 if role == "buyer" else 4150,
                              "reason": f"{role}_PRIVATE_ESTIMATION_DO_NOT_INHERIT", "uncertainty": "对手参数未知。"})
            return raw, {"request": {"model": "fake", "temperature": 0.3, "max_tokens": 1200,
                                      "messages": copy.deepcopy(messages)}, "response": {"content": raw}}
        return super().complete(messages)


class AblationTests(unittest.TestCase):
    def quiet(self, **kwargs):
        with redirect_stdout(StringIO()):
            return run_ablation(**kwargs)

    def test_truth_is_computed_from_world_and_restricted_to_auditor(self):
        world = TradeWorld(phone_negotiation())
        self.assertEqual(3350, world.reservation_price("seller", 3, 30, viewer=AUDITOR))
        self.assertEqual(4200, world.reservation_price("buyer", 3, 30, viewer=AUDITOR))
        for role in ("buyer", "seller"):
            with self.assertRaises(PermissionError):
                world.reservation_price("seller", 3, 30, viewer=role)
        # 不是对方是否愿意立即接受，更不取决于本场成交价。
        self.assertEqual("negotiating", world.audit_state(AUDITOR)["status"])

    def test_frozen_scenario_and_baseline_prompt_match_prior_real_log(self):
        old = json.loads((Path(__file__).resolve().parents[1] / "examples/phone_case_live.json").read_text(encoding="utf-8"))
        output = self.quiet(repeats=1, live=True, client=AblationClient())
        experiment = output["experiments"][0]
        self.assertEqual(old["experiments"][0]["scenario"], json.loads(json.dumps(experiment["scenario"])))
        self.assertEqual(old["experiments"][0]["s0_fingerprint"], experiment["s0_fingerprint"])
        old_system = old["experiments"][0]["runs"][0]["model_calls"][0]["request"]["messages"][0]
        current = experiment["runs"][0]["model_calls"][0]["request"]["messages"][0]
        self.assertEqual(old_system, current)

    def test_three_branches_share_first_life_and_differ_only_in_memory_slot(self):
        output = self.quiet(repeats=1, live=True, client=AblationClient())
        experiment = output["experiments"][0]
        baseline = experiment["runs"][0]
        by_group = {r["group"]: r for r in experiment["runs"]}
        systems = {role: [] for role in ("buyer", "seller")}
        for group in GROUPS:
            run = by_group[group]
            self.assertEqual(baseline["initial_state"], run["initial_state"])
            self.assertEqual(RESET_FRAME, run["session_context"])
            for role in ("buyer", "seller"):
                calls = [c for c in run["model_calls"] if c["purpose"] == "negotiation" and c["role"] == role]
                messages = calls[0]["request"]["messages"]
                self.assertEqual(2, len(messages))
                system = messages[0]["content"]
                before, payload = system.split("【本人授权记忆数据】\n")
                _, after = json.JSONDecoder().raw_decode(payload)
                systems[role].append((before, payload[after:]))
                if group == "control":
                    self.assertFalse(run["memories"])
                else:
                    memory = run["memories"][role]
                    self.assertEqual([], memory["previous_lives"])
                    expected = baseline["reflections"][role] if group == "rewind_hr" else None
                    self.assertEqual(expected, memory["personal_summary"])
                    self.assertNotIn(("seller" if role == "buyer" else "buyer") + "_PRIVATE_BELIEF", json.dumps(memory))
                self.assertNotIn("PRIVATE_ESTIMATION_DO_NOT_INHERIT", json.dumps(messages))
            self.assertEqual(300, run["estimations"]["buyer"]["absolute_error"])
            self.assertEqual(50, run["estimations"]["seller"]["absolute_error"])
        self.assertTrue(all(len(set(values)) == 1 for values in systems.values()))
        verify_results(output)

    def test_estimator_input_is_only_current_authorized_observation_and_does_not_mutate_world(self):
        world = TradeWorld(phone_negotiation())
        client = AblationClient()
        before = world.audit_state(AUDITOR)
        estimate, records = estimate_reservation(client, world, "buyer")
        self.assertEqual(json.loads(json.dumps(world.observe("buyer"))), json.loads(records[0]["request"]["messages"][1]["content"]))
        self.assertNotIn("true_boundary", json.dumps(records[0]["request"]))
        self.assertNotIn("迁移个人数据", json.dumps(records[0]["request"], ensure_ascii=False))
        self.assertEqual(3350, estimate["true_boundary"])
        self.assertEqual(before, world.audit_state(AUDITOR))
        self.assertEqual([], world.export_events(AUDITOR))

    def test_estimate_rejects_off_grid_boolean_nan_and_extra_fields(self):
        rules = TradeWorld(phone_negotiation()).contract_rules()
        for price in (3025, True, float("nan"), float("inf"), 2999, 4501):
            with self.subTest(price=price), self.assertRaises(ValueError):
                validate_estimate({"price": price, "reason": "自述", "uncertainty": "未知"}, rules)
        with self.assertRaises(ValueError):
            validate_estimate({"price": 3500, "reason": "自述", "uncertainty": "未知", "true_boundary": 3350}, rules)

    def test_saved_memory_or_truth_tampering_is_detected(self):
        output = self.quiet(repeats=1, live=True, client=AblationClient())
        for kind in ("reflection", "truth", "evaluation_input", "estimate_output"):
            changed = copy.deepcopy(output)
            run = next(r for r in changed["experiments"][0]["runs"] if r["group"] == "rewind_h")
            if kind == "reflection":
                run["memories"]["buyer"]["personal_summary"] = {"hidden": "not allowed"}
            elif kind == "truth":
                run["estimations"]["buyer"]["true_boundary"] = 3400
            elif kind == "estimate_output":
                run["estimations"]["buyer"].update(price=3500, absolute_error=150)
            else:
                call = next(c for c in run["model_calls"] if c["purpose"] == "estimation")
                call["request"]["messages"][1]["content"] = '{"true_boundary":3350}'
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                verify_results(changed)

    def test_bootstrap_pairs_and_missing_values_have_explicit_n(self):
        self.assertEqual([5, 5], paired_statistics([5, 5])["ci95"])
        self.assertIsNone(paired_statistics([5])["ci95"])
        self.assertIsNone(paired_statistics([])["mean"])
        output = self.quiet(repeats=2)
        second = output["experiments"][1]
        control = next(r for r in second["runs"] if r["group"] == "control")
        control["estimations"].pop("buyer")
        stats = ablation_statistics(output["experiments"])
        self.assertEqual(1, stats["groups"]["control"]["buyer_estimation_error"]["n"])
        self.assertEqual(1, stats["paired_contrasts"]["rewind_h-minus-control"]["buyer_estimation_error"]["n"])
        self.assertEqual(2, stats["paired_contrasts"]["rewind_h-minus-control"]["buyer_utility"]["n"])
        self.assertEqual(0, stats["error_change_from_first_life"]["rewind_hr"]["seller"]["mean"])

    def test_api_budget_failure_stops_and_reports_actual_counts(self):
        class BudgetClient(AblationClient):
            def complete(self, messages):
                raw, record = super().complete(messages)
                if "客观接受边界" in messages[0]["content"]:
                    record["error"] = {"http_status": 402}
                    raise ModelCallError("模拟预算耗尽", [record])
                return raw, record
        output = self.quiet(repeats=10, live=True, client=BudgetClient())
        self.assertEqual("partial", output["status"])
        self.assertEqual(1, output["counts"]["attempted_repeats"])
        self.assertEqual(0, output["counts"]["complete_repeats"])
        self.assertEqual(1, output["counts"]["failed_runs"])
        self.assertEqual("estimation", output["experiments"][0]["runs"][0]["error_phase"])
        self.assertTrue(output["experiments"][0]["runs"][0]["metrics"]["deal"])
        self.assertEqual(0, output["ablation_statistics"]["groups"]["control"]["buyer_utility"]["n"])

    def test_partial_repeats_are_not_replaced_and_pages_are_standalone(self):
        class OnceIllegal(AblationClient):
            def __init__(self):
                super().__init__()
                self.once = False

            def complete(self, messages):
                raw, record = super().complete(messages)
                if not self.once:
                    self.once = True
                    value = json.loads(raw)
                    value["action"] = {"kind": "offer", "price": 4125, "delivery_days": 3, "warranty_days": 30}
                    raw = json.dumps(value)
                    record["response"]["content"] = raw
                return raw, record
        with tempfile.TemporaryDirectory() as tmp:
            log, page = Path(tmp) / "log.json", Path(tmp) / "page.html"
            output = self.quiet(repeats=2, live=True, client=OnceIllegal(), log_path=log, html_path=page)
            self.assertEqual(2, output["counts"]["attempted_repeats"])
            self.assertEqual(1, output["counts"]["complete_repeats"])
            self.assertEqual(5, output["counts"]["attempted_runs"])
            self.assertEqual("partial", output["status"])
            self.assertEqual(1, len(output["experiments"][0]["runs"]))
            saved = json.loads(log.read_text(encoding="utf-8"))
            verify_results(saved)
            html = page.read_text(encoding="utf-8")
            self.assertIn("Rewind-HR", html)
            self.assertIn("买方估计卖方边界", html)
            self.assertNotIn("fetch(", html)


if __name__ == "__main__":
    unittest.main()
