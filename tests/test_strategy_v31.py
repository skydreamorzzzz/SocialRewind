from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path

from test_strategy_tree import StrategyClient
from social_rewind.negotiation import LEGACY_STRATEGY_PROMPT, STRATEGY_PROMPT, strategy_mode_instruction
from social_rewind.scenario_v3 import strategy_scenario
from social_rewind.strategy_tree import (NASH_REWARD, WELFARE_REWARD, SearchConfig, StrategyTree,
    batch_statistics, choose_winner, nash_reward, propose_plan, run_v3, verify_v3)
from social_rewind.tree_view import write_tree_page


def outcome(buyer: int, seller: int, *, deal: bool = True):
    return {"status": "completed", "metrics": {"deal": deal, "buyer_utility": buyer,
            "seller_utility": seller, "social_welfare": buyer + seller if deal else 0}}


class NashRewardTests(unittest.TestCase):
    def test_zero_balanced_no_deal_and_missing(self):
        self.assertEqual(0, nash_reward(outcome(0, 960)["metrics"]))
        self.assertGreater(nash_reward(outcome(350, 500)["metrics"]), nash_reward(outcome(0, 960)["metrics"]))
        self.assertAlmostEqual(math.sqrt(300 * 660) / 500, nash_reward(outcome(300, 660)["metrics"]))
        self.assertEqual(0, nash_reward(outcome(-20, 500)["metrics"]))
        self.assertEqual(0, nash_reward(outcome(300, 660, deal=False)["metrics"]))
        self.assertIsNone(nash_reward(None))

    def test_ucb_backprop_winner_and_welfare_are_separate(self):
        root = outcome(350, 500)
        tree = StrategyTree("guided", SearchConfig(), root)
        self.assertEqual(NASH_REWARD, tree.reward_version)
        records = {"root": root}
        for record in (outcome(0, 960), outcome(300, 660)):
            child = tree.add(tree.select())
            records[child.run_id] = record
            tree.finish(child.node_id, record)
        self.assertEqual(3, tree.nodes["n0"].visits)
        self.assertAlmostEqual(nash_reward(root["metrics"]) + math.sqrt(300 * 660) / 500, tree.nodes["n0"].value_sum)
        self.assertEqual("n2", tree.select()["parent_id"])
        self.assertEqual("n2", choose_winner(tree, records).node_id)
        self.assertEqual([850, 960, 960], [p["best_observed_welfare"] for p in tree.best_curve])
        self.assertEqual(0, tree.best_curve[1]["nash_reward"])
        self.assertAlmostEqual(math.sqrt(300 * 660) / 500, tree.best_curve[-1]["best_observed_nash_reward"])

    def test_ties_and_technical_failure_keep_original_semantics(self):
        record = outcome(300, 660)
        for version in (WELFARE_REWARD, NASH_REWARD):
            tree = StrategyTree("guided", SearchConfig(), record, reward_version=version)
            child = tree.add(tree.select())
            tree.finish(child.node_id, record)
            self.assertEqual("n1", choose_winner(tree, {"root": record, child.run_id: record}).node_id)
            failed = tree.add(tree.select())
            tree.finish(failed.node_id, {"status": "failed"})
            self.assertIsNone(failed.reward)
            self.assertEqual(2, tree.nodes["n0"].visits)
            self.assertEqual(3, tree.nodes["n0"].attempts)
            if version == NASH_REWARD:
                self.assertIsNone(tree.best_curve[-1]["nash_reward"])
                self.assertEqual(tree.best_curve[-2]["best_observed_nash_reward"], tree.best_curve[-1]["best_observed_nash_reward"])
            else:
                self.assertAlmostEqual(2 * 0.8, tree.nodes["n0"].value_sum)
                self.assertNotIn("best_observed_nash_reward", tree.best_curve[-1])


class V31ExperimentTests(unittest.TestCase):
    def run_fixture(self, **kwargs):
        with redirect_stdout(StringIO()):
            return run_v3(config=SearchConfig(max_nodes=3), live=True, client=StrategyClient(), **kwargs)

    def test_modes_require_testable_actions_but_allow_exploitation(self):
        explore = strategy_mode_instruction("exploration", STRATEGY_PROMPT)
        for text in ("公开回应支持或削弱", "第一个适当机会", "至少一次", "允许失败", "非法合同"):
            self.assertIn(text, explore)
        exploit = strategy_mode_instruction("exploitation", STRATEGY_PROMPT)
        self.assertIn("可以直接成交", exploit)
        self.assertIn("主观猜测", exploit)
        self.assertNotIn("至少一次", exploit)
        client = StrategyClient()
        propose_plan(client, "buyer", "exploration", [], {"role": "buyer"})
        self.assertIn(explore, client.requests[0][0]["content"])

    def test_identical_mode_prompts_private_conflicts_and_no_extra_calls(self):
        new = self.run_fixture()
        old = self.run_fixture(prompt_version=LEGACY_STRATEGY_PROMPT, reward_version=WELFARE_REWARD)
        self.assertEqual(old["counts"]["usage"]["by_purpose"], new["counts"]["usage"]["by_purpose"])
        self.assertEqual(old["search_config"], new["search_config"])
        self.assertEqual(old["scenario"], new["scenario"])
        e = new["experiments"][0]
        templates = {}
        for record in e["runs"].values():
            for call in record["model_calls"]:
                raw = json.dumps(call["request"], ensure_ascii=False)
                for forbidden in ('"reward"', '"nash_reward"', '"social_welfare"', '"ucb"', "nash-v1"):
                    self.assertNotIn(forbidden, raw)
                if call["purpose"] == "planning":
                    templates.setdefault((call["role"], record["branch_mode"]), set()).add(call["request"]["messages"][0]["content"])
                if call["purpose"] == "negotiation" and record["plans"]:
                    self.assertIn(strategy_mode_instruction(record["branch_mode"], STRATEGY_PROMPT), call["request"]["messages"][0]["content"])
        self.assertEqual(4, len(templates))
        self.assertTrue(all(len(values) == 1 for values in templates.values()))
        root_systems = {role: next(c for c in e["runs"]["root"]["model_calls"] if c["role"] == role and c["purpose"] == "negotiation")["request"]["messages"][0]["content"] for role in ("buyer", "seller")}
        self.assertIn("3天会造成不便", root_systems["buyer"])
        self.assertNotIn("迁移", root_systems["buyer"])
        self.assertIn("其他条件合理也可考虑", root_systems["seller"])
        self.assertNotIn("主力手机", root_systems["seller"])
        for system in root_systems.values():
            self.assertIn("不要仅因冲突提高退出概率", system)
            self.assertIn("不编造未建模的硬性截止日期", system)
        verify_v3(new)

    def test_versioned_statistics_validation_and_mixed_batches(self):
        new = self.run_fixture()
        self.assertEqual((STRATEGY_PROMPT, NASH_REWARD), (new["prompt_version"], new["reward_version"]))
        for e in new["experiments"]:
            for method, metadata in e["methods"].items():
                validation = e["runs"][metadata["validation_run_id"]]
                winner = e["runs"][next(n for n in metadata["tree"]["nodes"] if n["node_id"] == metadata["winner_id"])["run_id"]]
                self.assertEqual(winner["private_contexts"], validation["private_contexts"])
                self.assertEqual(winner["path_memories"], validation["path_memories"])
                self.assertEqual(nash_reward(validation["metrics"]), metadata["statistics"]["validation_nash_reward"])
                self.assertEqual(0, sum(c["purpose"] == "planning" for c in validation["model_calls"]))
                self.assertIn("nash_reward", new["statistics"]["search"][method])
                self.assertIn("best_observed_nash_reward", new["statistics"]["search"][method])
                self.assertNotIn("best_observed_nash_reward", new["statistics"]["validation"][method])
        old = self.run_fixture(prompt_version=LEGACY_STRATEGY_PROMPT, reward_version=WELFARE_REWARD)
        with self.assertRaisesRegex(ValueError, "不能混合统计"):
            batch_statistics(old["experiments"] + new["experiments"])

    def test_tampered_nash_planning_prompt_or_versions_fail(self):
        result = self.run_fixture()
        for field in ("nash", "prompt", "version"):
            changed = copy.deepcopy(result)
            e = changed["experiments"][0]
            if field == "nash":
                e["methods"]["guided"]["tree"]["best_curve"][-1]["best_observed_nash_reward"] += 0.1
            elif field == "prompt":
                next(c for c in e["runs"]["guided-n1"]["model_calls"] if c["purpose"] == "planning")["request"]["messages"][0]["content"] = "wrong version"
            else:
                changed["reward_version"] = WELFARE_REWARD
            with self.subTest(field=field), self.assertRaises((AssertionError, ValueError)):
                verify_v3(changed)

    def test_original_v3_pilot_and_formal_replay_without_rewriting(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        for name in ("v3_pilot.json", "v3_formal.json"):
            path = examples / name
            before = path.read_bytes()
            result = json.loads(before)
            self.assertNotIn("prompt_version", result)
            self.assertEqual(json.loads(json.dumps(asdict(strategy_scenario()))), result["scenario"])
            verify_v3(result)
            self.assertEqual(before, path.read_bytes())

    def test_page_labels_nash_and_welfare_separately(self):
        result = self.run_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v31.html"
            write_tree_page(result, path)
            page = path.read_text(encoding="utf-8")
            self.assertIn("SocialRewind v3.1", page)
            self.assertIn("Best-observed Nash Reward", page)
            self.assertIn("Best-observed Welfare", page)
            self.assertIn('"reward_version": "nash-v1"', page)


if __name__ == "__main__":
    unittest.main()
