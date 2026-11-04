from __future__ import annotations

import copy
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from social_rewind.models import Action, DecisionNote
from social_rewind.negotiation import (
    AgentOutputError,
    AgentTurn,
    LLMNegotiationAgent,
    NegotiationRunner,
    ScriptedAgent,
    render_control_prompt,
    render_rebirth_prompt,
)
from social_rewind.trade import (
    AUDITOR,
    TradeReplayError,
    TradeWorld,
    fictional_phone_trade,
)
from social_rewind.trade_demo import run_compare_demo, run_demo


def note(
    fact: str = "这是角色根据当前观察形成的自述。",
    evidence: tuple[int, ...] = (),
) -> DecisionNote:
    return DecisionNote(
        intent="我想推进交易。",
        known_facts=(fact,),
        uncertainty="我不确定对方是否让步。",
        evidence_event_ids=evidence,
    )


def act(kind: str, actor: str, **payload: object) -> Action:
    return Action(kind=kind, actor=actor, payload=payload, decision=note())


def turn(
    kind: str,
    speech: str,
    *,
    price: int | None = None,
    offer_id: str | None = None,
    evidence: tuple[int, ...] = (),
) -> AgentTurn:
    return AgentTurn(
        kind=kind,
        speech=speech,
        intent="我想推进交易。",
        known_facts=("这是我的角色自述。",),
        uncertainty="我不知道对方下一步。",
        price=price,
        offer_id=offer_id,
        evidence_event_ids=evidence,
    )


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
        self.calls += 1
        self.requests.append(copy.deepcopy(messages))
        if not self.outputs:
            raise AssertionError("FakeClient 没有剩余输出")
        output = self.outputs.pop(0)
        return output, {
            "provider": "fake",
            "request": {"messages": copy.deepcopy(messages)},
            "response": {"content": output},
        }


class TradeWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = fictional_phone_trade()
        self.world = TradeWorld(self.scenario)

    def test_formal_offer_and_accept_compute_raw_metrics(self) -> None:
        offer = self.world.act(act("offer", "buyer", text="我报 3600。", price=3600))
        self.assertTrue(offer["accepted"])
        accepted = self.world.act(
            act("accept", "seller", text="我明确接受。", offer_id="o0001")
        )
        self.assertTrue(accepted["accepted"])
        metrics = self.world.metrics(AUDITOR)
        self.assertEqual("deal", metrics["status"])
        self.assertEqual(3600, metrics["deal_price"])
        self.assertEqual(300, metrics["buyer_utility"])
        self.assertEqual(300, metrics["seller_utility"])
        self.assertEqual(1.0, metrics["nash_quality"])
        self.assertEqual(1, metrics["rounds"])

    def test_natural_language_acceptance_does_not_create_deal_or_day(self) -> None:
        self.world.act(act("message", "buyer", text="你的挂牌价我接受了。"))
        self.world.act(act("message", "seller", text="好的。"))
        state = self.world.audit_state(AUDITOR)
        self.assertEqual("negotiating", state["status"])
        self.assertEqual(1, state["rounds"])
        self.assertNotIn("day", state)

    def test_private_price_constraints_reject_without_leaking_attempt(self) -> None:
        before = self.world.audit_state(AUDITOR)
        refused = self.world.act(
            act("offer", "buyer", text="我报 4000。", price=4000)
        )
        self.assertFalse(refused["accepted"])
        self.assertEqual("buyer_limit_exceeded", refused["result"]["refusal"]["code"])
        self.assertEqual(before, self.world.audit_state(AUDITOR))
        self.assertEqual([], self.world.export_events("seller"))

        self.world.act(act("offer", "buyer", text="我报 3500。", price=3500))
        seller_before = self.world.audit_state(AUDITOR)
        seller_refused = self.world.act(
            act("offer", "seller", text="我报 3200。", price=3200)
        )
        self.assertFalse(seller_refused["accepted"])
        self.assertEqual(seller_before, self.world.audit_state(AUDITOR))
        buyer_events = self.world.export_events("buyer")
        self.assertEqual(2, len(buyer_events))
        self.assertNotIn("3200", json.dumps(buyer_events, ensure_ascii=False))

    def test_observations_protect_counterparty_private_value_and_utility(self) -> None:
        buyer = json.dumps(self.world.observe("buyer"), ensure_ascii=False)
        seller = json.dumps(self.world.observe("seller"), ensure_ascii=False)
        self.assertNotIn("minimum_price", buyer)
        self.assertNotIn("seller_utility", buyer)
        self.assertNotIn("maximum_price", seller)
        self.assertNotIn("buyer_utility", seller)
        self.assertIn("maximum_price", buyer)
        self.assertIn("minimum_price", seller)

    def test_reject_exit_and_timeout_are_deterministic(self) -> None:
        rejected = self.world.act(
            act("reject", "buyer", text="我拒绝挂牌价。", offer_id="o0000")
        )
        self.assertEqual("rejected", rejected["result"]["status"])
        exited = self.world.act(act("exit", "seller", text="我退出谈判。"))
        self.assertEqual("exited", exited["result"]["status"])
        self.assertEqual("seller_exit", self.world.audit_state(AUDITOR)["termination_reason"])

        timeout_world = TradeWorld(fictional_phone_trade(max_rounds=1))
        timeout_world.act(act("message", "buyer", text="先聊聊。"))
        timeout = timeout_world.act(act("message", "seller", text="暂不报价。"))
        self.assertEqual("timeout", timeout["result"]["status"])
        self.assertEqual("timeout", timeout_world.audit_state(AUDITOR)["status"])

    def test_accepting_offer_outside_own_limit_is_rejected(self) -> None:
        before = self.world.audit_state(AUDITOR)
        receipt = self.world.act(
            act("accept", "buyer", text="接受挂牌。", offer_id="o0000")
        )
        self.assertFalse(receipt["accepted"])
        self.assertEqual("buyer_limit_exceeded", receipt["result"]["refusal"]["code"])
        self.assertEqual(before, self.world.audit_state(AUDITOR))

    def test_unknown_actor_gets_structured_refusal(self) -> None:
        before = self.world.audit_state(AUDITOR)
        receipt = self.world.act(act("message", "intruder", text="尝试加入。"))
        self.assertFalse(receipt["accepted"])
        self.assertEqual("unknown_actor", receipt["result"]["refusal"]["code"])
        self.assertEqual(before, self.world.audit_state(AUDITOR))

    def test_live_demo_never_silently_falls_back_to_script(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须显式提供模型客户端"):
            run_demo(live=True, client=None)
        with self.assertRaisesRegex(ValueError, "必须显式提供模型客户端"):
            run_compare_demo(live=True, client=None)

    def test_offline_compare_uses_same_s0_and_isolates_control(self) -> None:
        with redirect_stdout(StringIO()):
            result = run_compare_demo(live=False)
        self.assertTrue(result["s0_equal"])
        self.assertTrue(result["model_config_equal"])
        self.assertFalse(result["control"]["memory_injected"])
        self.assertTrue(result["rewind"]["memory_injected"])
        self.assertEqual(
            result["timeline_1"]["metrics"], result["control"]["metrics"]
        )
        self.assertNotEqual(
            result["control"]["metrics"], result["rewind"]["metrics"]
        )
        self.assertNotIn("3700 元成交", result["control"]["context"])
        self.assertIn("没有保留任何可用的前世经历", result["control"]["context"])
        self.assertTrue(
            all(branch["replay_verified"] for branch in result["audit_worlds"].values())
        )

    def test_role_memory_excludes_opponent_decision_and_private_value(self) -> None:
        self.world.act(act("offer", "buyer", text="3400。", price=3400))
        secret = "卖方私人决策：绝不能透露底价"
        seller_action = Action(
            kind="offer",
            actor="seller",
            payload={"text": "4000。", "price": 4000},
            decision=note(secret, evidence=(1,)),
        )
        self.world.act(seller_action)
        buyer_memory = self.world.capture_memory("buyer")
        seller_memory = self.world.capture_memory("seller")
        buyer_text = json.dumps(buyer_memory.to_dict(), ensure_ascii=False)
        seller_text = json.dumps(seller_memory.to_dict(), ensure_ascii=False)
        self.assertNotIn(secret, buyer_text)
        self.assertNotIn("最低接受价", buyer_text)
        self.assertIn(secret, seller_text)
        self.assertIn("最低接受价", seller_text)

        buyer_prompt = render_rebirth_prompt(buyer_memory, "seller")
        seller_prompt = render_rebirth_prompt(seller_memory, "buyer")
        self.assertIn("seller 也重生了", buyer_prompt)
        self.assertIn("buyer 也重生了", seller_prompt)
        self.assertNotIn(secret, buyer_prompt)

    def test_reset_is_exact_and_agent_history_starts_empty(self) -> None:
        fresh_buyer = self.world.observe("buyer")
        s0 = self.world.audit_state(AUDITOR)
        self.world.act(act("offer", "buyer", text="3400。", price=3400))
        self.world.act(act("offer", "seller", text="4000。", price=4000))
        self.world.reset()
        self.assertEqual(s0, self.world.audit_state(AUDITOR))
        self.assertEqual(fresh_buyer, self.world.observe("buyer"))
        self.assertEqual([], self.world.export_events("buyer"))
        self.assertGreater(len(self.world.export_events(AUDITOR)), 0)

    def test_selected_memory_must_be_visible_to_owner(self) -> None:
        self.world.act(act("offer", "buyer", text="3400。", price=3400))
        memory = self.world.capture_memory("buyer", event_ids=[1])
        self.assertEqual(1, len(memory.visible_events))
        with self.assertRaises(PermissionError):
            self.world.capture_memory("seller", event_ids=[2])

    def test_two_timeline_runner_and_replay_need_no_model(self) -> None:
        runner = NegotiationRunner()
        buyer1 = ScriptedAgent(
            [turn("offer", "3400。", price=3400), turn("accept", "接受。", offer_id="o0002")]
        )
        seller1 = ScriptedAgent([turn("offer", "3700。", price=3700)])
        first = runner.run(self.world, buyer1, seller1)
        self.assertEqual("deal", first.metrics["status"])
        memories = {
            "buyer": self.world.capture_memory("buyer"),
            "seller": self.world.capture_memory("seller"),
        }
        s0 = TradeWorld(self.scenario).audit_state(AUDITOR)
        self.world.reset()
        self.assertEqual(s0, self.world.audit_state(AUDITOR))
        buyer2 = ScriptedAgent([turn("offer", "3600。", price=3600)])
        seller2 = ScriptedAgent([turn("accept", "接受。", offer_id="o0001")])
        second = runner.run(self.world, buyer2, seller2, memories=memories)
        self.assertEqual("deal", second.metrics["status"])
        self.assertIs(buyer2.memory, memories["buyer"])
        self.assertIs(seller2.memory, memories["seller"])

        events = self.world.export_events(AUDITOR)
        replayed = TradeWorld.replay(self.scenario, events)
        self.assertEqual(self.world.audit_state(AUDITOR), replayed.audit_state(AUDITOR))
        self.assertEqual(events, replayed.export_events(AUDITOR))

    def test_tampered_trade_log_fails_replay(self) -> None:
        self.world.act(act("offer", "buyer", text="3600。", price=3600))
        self.world.act(act("accept", "seller", text="接受。", offer_id="o0001"))
        events = self.world.export_events(AUDITOR)
        tampered = copy.deepcopy(events)
        tampered[-1]["result"]["deal_price"] = 3500
        with self.assertRaises(TradeReplayError):
            TradeWorld.replay(self.scenario, tampered)


class LLMAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = TradeWorld(fictional_phone_trade())

    @staticmethod
    def valid_output() -> str:
        return json.dumps(
            {
                "decision": {
                    "intent": "我先提出一个报价。",
                    "known_facts": ["挂牌价为 4200 元。"],
                    "uncertainty": "卖方是否让步未知。",
                    "evidence_event_ids": [],
                },
                "speech": "我报价 3500 元。",
                "action": {"kind": "offer", "price": 3500},
            },
            ensure_ascii=False,
        )

    def test_invalid_format_has_one_bounded_repair(self) -> None:
        client = FakeClient(["不是 JSON", self.valid_output()])
        agent = LLMNegotiationAgent(client, max_format_repairs=1)
        agent.start("buyer", self.world.scenario_context("buyer"), None)
        reply = agent.respond(self.world.observe("buyer"))
        self.assertEqual(2, client.calls)
        self.assertEqual(2, len(reply.call_records))
        self.assertIn("validation_error", reply.call_records[0])
        self.assertEqual("offer", reply.turn.kind)
        self.assertEqual(3500, reply.turn.price)

    def test_invalid_format_fails_without_fabricating_action(self) -> None:
        client = FakeClient(["坏格式", "仍然坏"])
        agent = LLMNegotiationAgent(client, max_format_repairs=1)
        agent.start("buyer", self.world.scenario_context("buyer"), None)
        before = self.world.audit_state(AUDITOR)
        with self.assertRaises(AgentOutputError) as caught:
            agent.respond(self.world.observe("buyer"))
        self.assertEqual(2, len(caught.exception.call_records))
        self.assertEqual(before, self.world.audit_state(AUDITOR))

    def test_control_prompt_discloses_reset_but_no_prior_history(self) -> None:
        client = FakeClient([self.valid_output()])
        agent = LLMNegotiationAgent(client)
        context = self.world.scenario_context("buyer")
        context["experiment_context"] = render_control_prompt()
        agent.start("buyer", context, None)
        agent.respond(self.world.observe("buyer"))
        self.assertEqual(1, client.calls)
        request_messages = client.requests[0]
        self.assertIn("世界已经恢复", request_messages[0]["content"])
        self.assertIn("没有保留任何可用的前世经历", request_messages[0]["content"])
        self.assertNotIn("上一世最终以", request_messages[0]["content"])
        self.assertNotIn("o0004", request_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
