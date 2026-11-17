from __future__ import annotations

import copy
import http.client
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path

from test_phone_case import PhoneClient
from social_rewind.llm import ModelCallError, OpenAICompatibleClient, OpenAICompatibleConfig
from social_rewind.models import Action, DecisionNote
from social_rewind.scenario_v3 import strategy_scenario
from social_rewind.strategy_tree import METHODS, MODES, PLAN_KEYS, WELFARE_REWARD, SearchConfig, StrategyTree, batch_statistics, run_v3, verify_v3
from social_rewind.trade import AUDITOR, TradeWorld, phone_negotiation, scenario_from_dict


def synthetic(welfare=840, status="completed"):
    return {"status": status, **({"metrics": {"social_welfare": welfare, "deal": welfare > 0}} if status == "completed" else {})}


class StrategyClient(PhoneClient):
    def complete(self, messages):
        if messages[0]["content"].startswith("你作为"):
            self.requests.append(copy.deepcopy(messages))
            role = json.loads(messages[1]["content"])["own_context"]["role"]
            raw = json.dumps({key: f"{role}_PRIVATE_PLAN_{len(self.requests)}_{key}" for key in PLAN_KEYS})
            return raw, {"request": {"model": "fake", "temperature": 0.3, "max_tokens": 1200,
                                      "messages": copy.deepcopy(messages)}, "response": {"content": raw}}
        return super().complete(messages)


class V3WorldTests(unittest.TestCase):
    def test_nine_interaction_contracts_and_maximum_welfare(self):
        buyers = {1: (250, 390, 420), 3: (0, 200, 240), 7: (-180, 0, 40)}
        sellers = {1: (-200, -350, -520), 3: (0, -40, -230), 7: (40, -10, -175)}
        welfare = []
        for d in (1, 3, 7):
            for i, w in enumerate((0, 30, 90)):
                world = TradeWorld(strategy_scenario())
                note = DecisionNote("我提出方案。", ("本人自述",), "对手未知")
                self.assertTrue(world.act(Action("offer", "buyer", {"text": "方案", "price": 3900, "delivery_days": d, "warranty_days": w}, note))["accepted"])
                self.assertTrue(world.act(Action("accept", "seller", {"text": "接受", "offer_id": "o0001"}, note))["accepted"])
                m = world.metrics(AUDITOR)
                self.assertEqual(200 + buyers[d][i], m["buyer_utility"])
                self.assertEqual(600 + sellers[d][i], m["seller_utility"])
                welfare.append((m["social_welfare"], d, w))
                self.assertEqual((d, w) == (3, 30), m["pareto_efficient"])
        self.assertEqual((960, 3, 30), max(welfare))
        self.assertNotEqual(sellers[1][1] - sellers[1][0], sellers[3][1] - sellers[3][0])
        self.assertEqual(strategy_scenario(), scenario_from_dict(json.loads(json.dumps(asdict(strategy_scenario())))))
        self.assertEqual(140, phone_negotiation().contract.buyer.warranty_utilities["30"])

    def test_private_table_and_illegal_action_isolation(self):
        world = TradeWorld(strategy_scenario())
        buyer = json.dumps(world.observe("buyer"), ensure_ascii=False)
        self.assertNotIn("完成基础检测", buyer)
        self.assertNotIn("social_welfare", buyer)
        self.assertNotIn('"960"', buyer)
        before = world.audit_state(AUDITOR)
        note = DecisionNote("我报价。", ("自述",), "未知")
        for payload in ({"price": 4125, "delivery_days": 3, "warranty_days": 30},
                        {"price": 4500, "delivery_days": 7, "warranty_days": 0},
                        {"price": 3900, "delivery_days": 3, "warranty_days": 30, "extra_guarantee": "gift"}):
            self.assertFalse(world.act(Action("offer", "buyer", {"text": "方案", **payload}, note))["accepted"])
            self.assertEqual(before, world.audit_state(AUDITOR))
        self.assertEqual([], world.export_events("seller"))
        bad = strategy_scenario()
        with self.assertRaises(ValueError):
            TradeWorld(replace(bad, contract=replace(bad.contract, buyer=replace(bad.contract.buyer, adjustment={"1": {"0": 0}}))))


class TreeAlgorithmTests(unittest.TestCase):
    # 保留原v3的福利奖励断言；v3.1默认Nash及其选择在test_strategy_v31覆盖。
    def make_two(self, first, second):
        tree = StrategyTree("guided", SearchConfig(), synthetic(), reward_version=WELFARE_REWARD)
        for score in (first, second):
            node = tree.add(tree.select())
            tree.finish(node.node_id, synthetic(score))
        return tree

    def test_ucb_depends_on_observed_rewards_not_fixed_expansion(self):
        a, b = self.make_two(600, 960), self.make_two(960, 600)
        sa, sb = a.select(), b.select()
        self.assertEqual("n2", sa["parent_id"])
        self.assertEqual("n1", sb["parent_id"])
        route = sa["route"][0]
        self.assertEqual("ucb_observed_subtree_value", route["reason"])
        self.assertTrue(all("ucb" in c for c in route["candidates"]))

    def test_backprop_and_budget_depth_limits(self):
        for method in METHODS:
            tree = StrategyTree(method, SearchConfig(), synthetic(960), reward_version=WELFARE_REWARD)
            while (selection := tree.select()) is not None:
                node = tree.add(selection)
                tree.finish(node.node_id, synthetic(960))
            self.assertEqual(7, len(tree.nodes))
            self.assertEqual(7, tree.nodes["n0"].visits)
            self.assertAlmostEqual(7 * 0.8, tree.nodes["n0"].value_sum)
            self.assertLessEqual(max(n.depth for n in tree.nodes.values()), 6 if method == "linear" else 3)
            if method == "linear":
                self.assertEqual(6, max(n.depth for n in tree.nodes.values()))
            self.assertTrue(all(len(tree.children(n.node_id)) <= (1 if method == "linear" else 2) for n in tree.nodes.values()))
            self.assertTrue(all(len({c.branch_mode for c in tree.children(n.node_id)}) == len(tree.children(n.node_id)) for n in tree.nodes.values()))
        with self.assertRaises(ValueError):
            SearchConfig(max_nodes=8).validate()

    def test_ties_are_seeded_and_failures_do_not_fabricate_samples(self):
        a, b = self.make_two(840, 840), self.make_two(840, 840)
        self.assertEqual(a.select(), b.select())
        tree = StrategyTree("guided", SearchConfig(), synthetic(840), reward_version=WELFARE_REWARD)
        for _ in range(2):
            child = tree.add(tree.select())
            tree.finish(child.node_id, synthetic(status="failed"))
            self.assertIsNone(child.reward)
        self.assertIsNone(tree.select())
        self.assertEqual(1, tree.nodes["n0"].visits)
        self.assertEqual(3, tree.nodes["n0"].attempts)


class V3ExperimentTests(unittest.TestCase):
    def quiet(self, **kwargs):
        with redirect_stdout(StringIO()):
            return run_v3(**kwargs)

    def test_default_equal_budget_and_validation_excluded_from_search(self):
        result = self.quiet()
        self.assertEqual(22, result["counts"]["negotiations_started"])
        self.assertEqual(22, result["counts"]["terminal_negotiations"])
        experiment = result["experiments"][0]
        for method, meta in experiment["methods"].items():
            self.assertEqual(7, len(meta["tree"]["nodes"]))
            self.assertEqual(7, len(meta["tree"]["best_curve"]))
            self.assertEqual(6, meta["statistics"]["additional_attempts"])
            self.assertEqual(960, meta["statistics"]["best_observed_welfare"])
            self.assertEqual(850, experiment["runs"][meta["validation_run_id"]]["metrics"]["social_welfare"])
        self.assertEqual(result["experiments"][0]["methods"]["random"]["tree"]["config"], result["experiments"][0]["methods"]["guided"]["tree"]["config"])
        verify_v3(json.loads(json.dumps(result)))

    def test_private_path_plan_and_prompt_isolation(self):
        result = self.quiet(live=True, client=StrategyClient())
        experiment = result["experiments"][0]
        for method, meta in experiment["methods"].items():
            node_map = {n["node_id"]: n for n in meta["tree"]["nodes"]}
            for node in list(node_map.values())[1:]:
                record = experiment["runs"][node["run_id"]]
                ancestor_ids = []
                parent = node_map[node["parent_id"]]
                while True:
                    ancestor_ids.append(parent["node_id"])
                    if parent["parent_id"] is None:
                        break
                    parent = node_map[parent["parent_id"]]
                for role, opposite in (("buyer", "seller"), ("seller", "buyer")):
                    self.assertEqual(list(reversed(ancestor_ids)), [m["source_node_id"] for m in record["path_memories"][role]])
                    context = json.dumps(record["private_contexts"][role], ensure_ascii=False)
                    self.assertNotIn(opposite + "_PRIVATE_PLAN", context)
                    self.assertNotIn(opposite + "_PRIVATE_BELIEF", context)
                    self.assertNotIn(opposite + "_PRIVATE_DECISION", context)
                    self.assertNotIn('"social_welfare"', context)
                    self.assertNotIn('"reward"', context)
                    self.assertNotIn('"ucb"', context)
                self.assertEqual({"buyer", "seller"}, set(record["plans"]))
            validation = experiment["runs"][meta["validation_run_id"]]
            winner = experiment["runs"][node_map[meta["winner_id"]]["run_id"]]
            self.assertEqual(winner["private_contexts"], validation["private_contexts"])
            self.assertEqual(0, sum(c["purpose"] == "planning" for c in validation["model_calls"]))
        verify_v3(result)

    def test_incomplete_budget_is_retained_but_not_treated_as_equal_budget(self):
        result = self.quiet()
        metadata = result["experiments"][0]["methods"]["linear"]
        metadata["statistics"]["budget_complete"] = False
        stats = batch_statistics(result["experiments"])
        self.assertEqual(0, stats["search"]["linear"]["social_welfare"]["n"])
        self.assertEqual(1, stats["all_attempted_search"]["linear"]["social_welfare"]["n"])
        self.assertEqual(0, stats["paired"]["guided-minus-linear"]["social_welfare"]["n"])
        self.assertEqual(1, stats["paired"]["guided-minus-random"]["social_welfare"]["n"])
        self.assertEqual(1, stats["validation"]["linear"]["social_welfare"]["n"])

    def test_tampered_selection_path_and_winner_validation_are_rejected(self):
        result = self.quiet()
        for field in ("selection", "memory", "validation", "reward", "counts"):
            changed = copy.deepcopy(result)
            e = changed["experiments"][0]
            m = e["methods"]["guided"]
            if field == "selection":
                m["tree"]["selections"][-1]["parent_id"] = "n0"
            elif field == "memory":
                e["runs"]["guided-n3"]["path_memories"]["buyer"].append({"source_node_id": "sibling"})
            elif field == "validation":
                e["runs"][m["validation_run_id"]]["private_contexts"]["buyer"]["winner_result"] = {"social_welfare": 960}
            elif field == "counts":
                changed["counts"]["usage"]["calls"] += 1
            else:
                m["tree"]["nodes"][0]["value_sum"] += 1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                verify_v3(changed)

    def test_interruption_is_saved_and_replayable(self):
        class InterruptedClient(StrategyClient):
            def complete(self, messages):
                if messages[0]["content"].startswith("你作为"):
                    raise KeyboardInterrupt()
                return super().complete(messages)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interrupted.json"
            result = self.quiet(live=True, client=InterruptedClient(), log_path=path)
            self.assertEqual("partial", result["status"])
            self.assertTrue(path.exists())
            self.assertEqual(1, result["counts"]["failed_or_interrupted"])
            verify_v3(json.loads(path.read_text(encoding="utf-8")))

    def test_api_quota_failure_preserves_calls_and_stops_without_fake_results(self):
        class QuotaClient(StrategyClient):
            def __init__(self, failing_role):
                super().__init__()
                self.failing_role = failing_role

            def complete(self, messages):
                if (messages[0]["content"].startswith("你作为")
                    and json.loads(messages[1]["content"])["own_context"]["role"] == self.failing_role):
                    raise ModelCallError("quota", [{"request": {"model": "fake", "temperature": 0.3,
                        "max_tokens": 1200, "messages": copy.deepcopy(messages)}, "error": {"http_status": 402}}])
                return super().complete(messages)
        for role in ("buyer", "seller"):
            with self.subTest(role=role):
                result = self.quiet(live=True, client=QuotaClient(role), repeats=5)
                self.assertEqual("partial", result["status"])
                self.assertEqual(1, result["counts"]["attempted_repeats"])
                self.assertEqual(1, result["counts"]["negotiations_started"])
                record = result["experiments"][0]["runs"]["linear-n1"]
                self.assertEqual(402, record["model_calls"][-1]["error"]["http_status"])
                self.assertEqual({"buyer"} if role == "seller" else set(), set(record["plans"]))
                self.assertEqual({}, record["private_contexts"])
                self.assertNotIn("metrics", record)
                self.assertIsNone(result["experiments"][0]["methods"]["linear"]["tree"]["nodes"][-1]["reward"])
                verify_v3(result)

    def test_transport_failure_is_explicit_and_does_not_retry(self):
        client = OpenAICompatibleClient(OpenAICompatibleConfig("https://example.invalid", "test-key", "fake"))
        response = MagicMock()
        response.read.side_effect = http.client.IncompleteRead(b'{"partial":')
        connection = MagicMock()
        connection.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=connection) as request:
            with self.assertRaises(ModelCallError) as caught:
                client.complete([{"role": "user", "content": "test"}])
        self.assertEqual(1, request.call_count)
        record = caught.exception.call_records[0]
        self.assertEqual('{"partial":', record["error"]["partial_response"])
        self.assertEqual("IncompleteRead", record["error"]["type"])
        self.assertNotIn("test-key", json.dumps(record))

    def test_remaining_repeats_keep_original_ids_and_order(self):
        result = self.quiet(repeats=2, repeat_start=2, config=SearchConfig(max_nodes=2))
        self.assertEqual([2, 3], [e["repeat"] for e in result["experiments"]])
        self.assertEqual(["linear", "guided", "random"], result["experiments"][0]["method_order"])
        self.assertEqual(["random", "linear", "guided"], result["experiments"][1]["method_order"])
        verify_v3(result)

    def test_html_uses_real_records_and_escapes_model_content(self):
        from social_rewind.tree_view import write_tree_page
        result = self.quiet(config=SearchConfig(max_nodes=3))
        payload = '</script><script>alert("x")</script>'
        result["experiments"][0]["runs"]["root"]["events"][0]["input"]["text"] = payload
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.html"
            write_tree_page(result, path)
            html = path.read_text(encoding="utf-8")
            self.assertNotIn(payload, html)
            self.assertNotIn("fetch(", html)
            self.assertIn("未探索候选（未运行）", html)
            self.assertIn("独立重执行验证", html)


if __name__ == "__main__":
    unittest.main()
