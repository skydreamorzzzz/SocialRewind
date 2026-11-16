from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .models import Action


AUDITOR = "auditor"
ROLES = ("buyer", "seller")
TRADE_ACTIONS = {"message", "offer", "accept", "reject", "exit"}


class TradeReplayError(RuntimeError):
    pass


class _Refusal(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = {"code": code, "message": message, "details": details}


@dataclass(frozen=True)
class ContractPreferences:
    """buyer: value-price-delay*(d-1)+warranty*w；seller: price-value+delay*(d-1)-warranty*w。"""

    value: int
    delay: int
    warranty: int


@dataclass(frozen=True)
class DiscretePreferences:
    value: int
    delivery_utilities: dict[str, int]
    warranty_utilities: dict[str, int]
    role_card: tuple[str, ...]
    minimum_utility: int = 0


def _terms_adjustment(pref: ContractPreferences | DiscretePreferences, role: str, d: int, w: int) -> int:
    if isinstance(pref, DiscretePreferences):
        return pref.delivery_utilities[str(d)] + pref.warranty_utilities[str(w)]
    adjustment = -pref.delay * (d - 1) + pref.warranty * w
    return adjustment if role == "buyer" else -adjustment


@dataclass(frozen=True)
class ContractConfig:
    buyer: ContractPreferences | DiscretePreferences
    seller: ContractPreferences | DiscretePreferences
    price_min: int = 3000
    price_max: int = 4500
    price_step: int = 50
    delivery_days: tuple[int, ...] = (1, 3, 7)
    warranty_days: tuple[int, ...] = (0, 30, 90)


@dataclass(frozen=True)
class TradeScenario:
    scenario_id: str
    title: str
    fictional: bool
    product: dict[str, Any]
    listed_price: int
    buyer_max_price: int
    seller_min_price: int
    max_rounds: int = 20
    gamma: float = 0.99
    contract: ContractConfig | None = None


@dataclass(frozen=True)
class TradeMemory:
    """世界外、按角色授权的上一世可见经历。"""

    role: str
    source_timeline_id: int
    visible_events: tuple[dict[str, Any], ...]
    outcome: dict[str, Any]
    own_private_fact: str
    previous_lives: tuple[dict[str, Any], ...] = ()
    personal_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["visible_events"] = list(self.visible_events)
        return _canonical(value)


def fictional_phone_trade(max_rounds: int = 20) -> TradeScenario:
    return TradeScenario(
        scenario_id="fictional-used-phone-v1",
        title="虚构案例：二手手机交易",
        fictional=True,
        product={
            "name": "星云 X1 手机（虚构）",
            "condition": "使用 10 个月，屏幕无裂纹，电池健康度 91%",
            "included": "手机、保护壳、充电线",
        },
        listed_price=4200,
        buyer_max_price=3900,
        seller_min_price=3300,
        max_rounds=max_rounds,
    )


def multidimensional_trade(profile: str = "urgent", max_rounds: int = 20) -> TradeScenario:
    """历史 v2 线性配置，仅用于兼容既有测试和审计记录。"""
    profiles = {
        "urgent": (ContractPreferences(3900, 70, 3), ContractPreferences(3300, 20, 1)),
        "warranty": (ContractPreferences(3700, 5, 6), ContractPreferences(3300, 20, 2)),
    }
    if profile not in profiles:
        raise ValueError(f"未知偏好配置: {profile}")
    buyer, seller = profiles[profile]
    return TradeScenario(
        scenario_id=f"fictional-phone-v2-{profile}", title=f"虚构三维合同：{profile}",
        fictional=True, product=fictional_phone_trade().product, listed_price=4200,
        buyer_max_price=buyer.value, seller_min_price=seller.value, max_rounds=max_rounds,
        contract=ContractConfig(buyer, seller),
    )


def phone_negotiation(max_rounds: int = 20) -> TradeScenario:
    """同一虚构手机案例的离散效用与角色卡，不导入真实成交数据。"""
    buyer = DiscretePreferences(4100, {"1": 120, "3": 0, "7": -250},
                                {"0": 0, "30": 140, "90": 170}, (
        "我希望以合理价格买到可靠的手机，担心隐藏故障。",
        "我偏好尽快交付，但可以为了更好的价格等待。",
        "我希望有适当售后保障，90 天相比 30 天的额外价值有限。",
        "我的理想目标是较低成交价、较快交付、至少 30 天保障。",
        "我不知道卖方的最低接受条件和履约成本。",
    ))
    seller = DiscretePreferences(3300, {"1": -220, "3": 0, "7": 20},
                                 {"0": 0, "30": -50, "90": -170}, (
        "我希望尽可能提高成交收益，也希望尽快完成交易。",
        "我需要迁移个人数据，1 天交付有额外加急成本，3 天更合适。",
        "我可以提供售后保障，但期限越长，潜在成本越高。",
        "我的理想目标是较高成交价、3 天交付、控制售后责任。",
        "我不知道买方的预算、时间偏好与保障估值。",
    ))
    return TradeScenario(
        scenario_id="fictional-phone-v2-discrete", title="星云 X1：交付与保障的条件交换",
        fictional=True, product=fictional_phone_trade().product, listed_price=4200,
        buyer_max_price=buyer.value, seller_min_price=seller.value, max_rounds=max_rounds,
        contract=ContractConfig(buyer, seller),
    )


def scenario_from_dict(value: dict[str, Any]) -> TradeScenario:
    data = copy.deepcopy(value)
    if data.get("contract") is not None:
        contract = data["contract"]
        for role in ROLES:
            pref = contract[role]
            if "delivery_utilities" in pref:
                pref["role_card"] = tuple(pref["role_card"])
                contract[role] = DiscretePreferences(**pref)
            else:
                contract[role] = ContractPreferences(**pref)
        contract["delivery_days"] = tuple(contract["delivery_days"])
        contract["warranty_days"] = tuple(contract["warranty_days"])
        data["contract"] = ContractConfig(**contract)
    return TradeScenario(**data)


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _diff(before: Any, after: Any, path: str = "$") -> list[dict[str, Any]]:
    if type(before) is not type(after):
        return [{"path": path, "before": before, "after": after}]
    if isinstance(before, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}"
            if key not in before:
                changes.append({"path": child, "before": None, "after": after[key]})
            elif key not in after:
                changes.append({"path": child, "before": before[key], "after": None})
            else:
                changes.extend(_diff(before[key], after[key], child))
        return changes
    if isinstance(before, list):
        return [] if before == after else [{"path": path, "before": before, "after": after}]
    return [] if before == after else [{"path": path, "before": before, "after": after}]


class TradeWorld:
    """单买家、单卖家、单商品的确定性交易世界。"""

    def __init__(self, scenario: TradeScenario) -> None:
        self._validate_scenario(scenario)
        self._scenario = copy.deepcopy(scenario)
        initial = self._make_s0(scenario)
        self._s0_json = json.dumps(initial, ensure_ascii=False, sort_keys=True)
        self._state: dict[str, Any] = json.loads(self._s0_json)
        self._feasible_utilities: list[tuple[int, int]] = []
        if scenario.contract is not None:
            c = scenario.contract
            for p in range(c.price_min, c.price_max + 1, c.price_step):
                for d in c.delivery_days:
                    for w in c.warranty_days:
                        pair = (self._utility("buyer", p, d, w), self._utility("seller", p, d, w))
                        if all(u >= getattr(getattr(c, r), "minimum_utility", 0) for r, u in zip(ROLES, pair)):
                            self._feasible_utilities.append(pair)
        self._events: list[dict[str, Any]] = []
        self._timeline_id = 1
        self._assert_invariants(self._state)

    @staticmethod
    def _validate_scenario(scenario: TradeScenario) -> None:
        if not scenario.fictional:
            raise ValueError("交易原型只接受明确标记为虚构的场景")
        prices = (scenario.listed_price, scenario.buyer_max_price, scenario.seller_min_price)
        if any(isinstance(price, bool) or not isinstance(price, int) or price <= 0 for price in prices):
            raise ValueError("所有价格必须是正整数")
        if scenario.contract is None and scenario.listed_price < scenario.seller_min_price:
            raise ValueError("挂牌价不得低于卖方最低接受价")
        if scenario.contract is None and scenario.buyer_max_price <= scenario.seller_min_price:
            raise ValueError("质量指标要求买方最高价严格高于卖方最低价")
        if type(scenario.max_rounds) is not int or not 1 <= scenario.max_rounds <= 20:
            raise ValueError("max_rounds 必须在 1 到 20 之间")
        if (
            isinstance(scenario.gamma, bool)
            or not isinstance(scenario.gamma, (int, float))
            or not 0 < scenario.gamma <= 1
        ):
            raise ValueError("gamma 必须在 (0, 1] 范围内")
        c = scenario.contract
        if c is not None:
            numbers = (c.price_min, c.price_max, c.price_step, *c.delivery_days, *c.warranty_days,
                       c.buyer.value, c.seller.value)
            if any(type(n) is not int or n < 0 for n in numbers):
                raise ValueError("合同范围与偏好必须是非负整数")
            if not 0 < c.price_min <= c.price_max or c.price_step <= 0:
                raise ValueError("合同价格区间和步长无效")
            if (c.price_max - c.price_min) % c.price_step:
                raise ValueError("价格区间必须可被步长整除")
            if (not c.delivery_days or not c.warranty_days or min(c.delivery_days) < 1
                    or len(set(c.delivery_days)) != len(c.delivery_days)
                    or len(set(c.warranty_days)) != len(c.warranty_days)):
                raise ValueError("交付/保障选项必须非空、不重复，交付至少一天")
            for pref in (c.buyer, c.seller):
                if isinstance(pref, DiscretePreferences):
                    for table, options in ((pref.delivery_utilities, c.delivery_days), (pref.warranty_utilities, c.warranty_days)):
                        if set(table) != {str(n) for n in options} or any(type(v) is not int for v in table.values()):
                            raise ValueError("离散效用表必须完整覆盖公开选项，效用必须为整数")
                    if type(pref.minimum_utility) is not int or pref.minimum_utility < 0:
                        raise ValueError("最低可接受效用必须为非负整数")
                    if not pref.role_card or any(not isinstance(s, str) or not s.strip() for s in pref.role_card):
                        raise ValueError("私人角色卡不能为空")
                elif any(type(n) is not int or n < 0 for n in (pref.delay, pref.warranty)):
                    raise ValueError("历史线性偏好系数必须为非负整数")
            count = (1 + (c.price_max - c.price_min) // c.price_step) * len(c.delivery_days) * len(c.warranty_days)
            if count > 10000:
                raise ValueError("MVP 合同空间最多 10000 个组合")
            if not c.price_min <= scenario.listed_price <= c.price_max or (scenario.listed_price - c.price_min) % c.price_step:
                raise ValueError("挂牌价必须在合同价格网格内")
            listing_utility = scenario.listed_price - c.seller.value + _terms_adjustment(
                c.seller, "seller", c.delivery_days[0], c.warranty_days[0])
            if listing_utility < getattr(c.seller, "minimum_utility", 0):
                raise ValueError("挂牌合同必须满足卖方非负效用")

    @staticmethod
    def _make_s0(scenario: TradeScenario) -> dict[str, Any]:
        listing = {
            "id": "o0000",
            "actor": "seller",
            "price": scenario.listed_price,
            "status": "active",
            "round": 0,
            "source": "listing",
        }
        state = {
            "scenario_id": scenario.scenario_id,
            "status": "negotiating",
            "termination_reason": None,
            "turns_taken": 0,
            "rounds": 0,
            "next_actor": "buyer",
            "offer_serial": 0,
            "offers": {"o0000": listing},
            "active_offer_id": "o0000",
            "deal_price": None,
        }
        if scenario.contract is not None:
            state["contract_config"] = _canonical(asdict(scenario.contract))
            state["deal_terms"] = None
            listing["delivery_days"] = scenario.contract.delivery_days[0]
            listing["warranty_days"] = scenario.contract.warranty_days[0]
        return state

    def _utility(self, role: str, price: int, delivery_days: int, warranty_days: int) -> int:
        if role not in ROLES or self._scenario.contract is None:
            raise ValueError("utility 仅用于三维合同的 buyer/seller")
        pref = getattr(self._scenario.contract, role)
        base = pref.value - price if role == "buyer" else price - pref.value
        return base + _terms_adjustment(pref, role, delivery_days, warranty_days)

    def contract_rules(self) -> dict[str, Any] | None:
        c = self._scenario.contract
        if c is None:
            return None
        return {"price_min": c.price_min, "price_max": c.price_max, "price_step": c.price_step,
                "delivery_days": list(c.delivery_days), "warranty_days": list(c.warranty_days)}

    @property
    def s0_fingerprint(self) -> str:
        return hashlib.sha256(self._s0_json.encode("utf-8")).hexdigest()

    def audit_state(self, viewer: str) -> dict[str, Any]:
        if viewer != AUDITOR:
            raise PermissionError("完整交易状态仅审计者可读")
        return copy.deepcopy(self._state)

    def audit_timeline_id(self, viewer: str) -> int:
        if viewer != AUDITOR:
            raise PermissionError("时间线编号仅审计者可读")
        return self._timeline_id

    def observe(self, viewer: str) -> dict[str, Any]:
        if viewer not in ROLES and viewer != AUDITOR:
            raise PermissionError("未知观察者")
        if viewer == AUDITOR:
            return {
                "scenario": _canonical(asdict(self._scenario)),
                "state": copy.deepcopy(self._state),
                "metrics": self.metrics(AUDITOR),
                "events": self.export_events(AUDITOR),
            }
        own_value = (
            {"maximum_price": self._scenario.buyer_max_price}
            if viewer == "buyer"
            else {"minimum_price": self._scenario.seller_min_price}
        )
        if self._scenario.contract is not None:
            own_value = {"preferences": asdict(getattr(self._scenario.contract, viewer)),
                         "utility_formula": ("value-price-delay*(delivery_days-1)+warranty*warranty_days"
                                             if viewer == "buyer" else
                                             "price-value+delay*(delivery_days-1)-warranty*warranty_days")}
            # 仅展开角色已知的自身效用，不推荐策略，不使用对手偏好。
            c = self._scenario.contract
            pref = getattr(c, viewer)
            if isinstance(pref, DiscretePreferences):
                own_value["utility_formula"] = ("value-price+delivery_utilities[delivery_days]+warranty_utilities[warranty_days]"
                                                if viewer == "buyer" else
                                                "price-value+delivery_utilities[delivery_days]+warranty_utilities[warranty_days]")
            limits = []
            for d in c.delivery_days:
                for w in c.warranty_days:
                    adjustment = _terms_adjustment(pref, viewer, d, w)
                    minimum = getattr(pref, "minimum_utility", 0)
                    bound = pref.value + adjustment - minimum if viewer == "buyer" else pref.value - adjustment + minimum
                    if viewer == "buyer":
                        price_limit = min(c.price_max, c.price_min + (bound - c.price_min) // c.price_step * c.price_step)
                        available = price_limit >= c.price_min
                    else:
                        grid_steps = (bound - c.price_min + c.price_step - 1) // c.price_step
                        price_limit = max(c.price_min, c.price_min + grid_steps * c.price_step)
                        available = price_limit <= c.price_max
                    limits.append({"delivery_days": d, "warranty_days": w,
                                   "maximum_price" if viewer == "buyer" else "minimum_price": price_limit if available else None})
            own_value["price_limits_by_terms"] = limits
            active = self._active_offer_public()
            own_value["active_offer_utility"] = (
                self._utility(viewer, **{k: active[k] for k in ("price", "delivery_days", "warranty_days")})
                if active else None)
        return {
            "scenario_id": self._scenario.scenario_id,
            "fictional": True,
            "role": viewer,
            "product": copy.deepcopy(self._scenario.product),
            "listed_price": self._scenario.listed_price,
            "max_rounds": self._scenario.max_rounds,
            "private": own_value,
            **({"contract_rules": self.contract_rules()} if self._scenario.contract else {}),
            "status": self._state["status"],
            "termination_reason": self._state["termination_reason"],
            "current_round": self._current_round(),
            "rounds_completed": self._state["rounds"],
            "next_actor": self._state["next_actor"],
            "active_offer": self._active_offer_public(),
            "offers": [self._offer_public(offer) for offer in self._state["offers"].values()],
            "own_result": self.metrics(viewer),
            "events": self.export_events(viewer),
        }

    def scenario_context(self, role: str) -> dict[str, Any]:
        observation = self.observe(role)
        context = {
            key: observation[key]
            for key in ("fictional", "role", "product", "listed_price", "max_rounds", "private")
        }
        if self._scenario.contract is not None:
            context["contract_rules"] = self.contract_rules()
        return context

    def _current_round(self) -> int:
        if self._state["status"] != "negotiating":
            return self._state["rounds"]
        return min(self._state["turns_taken"] // 2 + 1, self._scenario.max_rounds)

    @staticmethod
    def _offer_public(offer: dict[str, Any]) -> dict[str, Any]:
        return {
            key: offer[key]
            for key in ("id", "actor", "price", "status", "round", "source", "delivery_days", "warranty_days")
            if key in offer
        }

    def _active_offer_public(self) -> dict[str, Any] | None:
        offer_id = self._state["active_offer_id"]
        if offer_id is None:
            return None
        return self._offer_public(self._state["offers"][offer_id])

    def metrics(self, viewer: str = AUDITOR) -> dict[str, Any]:
        if viewer not in ROLES and viewer != AUDITOR:
            raise PermissionError("未知指标读取者")
        deal = self._state["status"] == "deal"
        price = self._state["deal_price"]
        rounds = self._state["rounds"]
        discount = self._scenario.gamma ** max(0, rounds - 1)
        public = {
            "status": self._state["status"],
            "deal": deal,
            "deal_price": price,
            "rounds": rounds,
            "efficiency_discount": round(discount, 8),
        }
        if self._scenario.contract is not None:
            terms = self._state["deal_terms"]
            public["deal_terms"] = copy.deepcopy(terms)
            bu = self._utility("buyer", **terms) if deal else 0
            su = self._utility("seller", **terms) if deal else 0
            if viewer in ROLES:
                return {**public, "own_utility": bu if viewer == "buyer" else su}
            dominated = any(b >= bu and s >= su and (b > bu or s > su)
                            for b, s in self._feasible_utilities)
            max_welfare = max((b + s for b, s in self._feasible_utilities), default=0)
            return {**public, "buyer_utility": bu, "seller_utility": su,
                    "social_welfare": bu + su, "pareto_efficient": not dominated,
                    "nash_quality": round(4 * bu * su / max_welfare**2, 8) if max_welfare else 0.0}
        if deal:
            buyer_utility = self._scenario.buyer_max_price - price
            seller_utility = price - self._scenario.seller_min_price
            surplus = self._scenario.buyer_max_price - self._scenario.seller_min_price
            buyer_normalized = buyer_utility / surplus
            seller_normalized = seller_utility / surplus
            quality = 4 * buyer_normalized * seller_normalized
        else:
            buyer_utility = seller_utility = 0
            buyer_normalized = seller_normalized = quality = 0.0
        if viewer == "buyer":
            return {**public, "own_utility": buyer_utility}
        if viewer == "seller":
            return {**public, "own_utility": seller_utility}
        return {
            **public,
            "buyer_utility": buyer_utility,
            "seller_utility": seller_utility,
            "buyer_normalized_utility": round(buyer_normalized, 8),
            "seller_normalized_utility": round(seller_normalized, 8),
            "nash_quality": round(quality, 8),
        }

    def export_events(self, viewer: str) -> list[dict[str, Any]]:
        if viewer == AUDITOR:
            return copy.deepcopy(self._events)
        if viewer not in ROLES:
            raise PermissionError("未知日志读取者")
        visible = self._visible_internal_events(viewer)
        return [
            self._agent_event_view(event, viewer, visible_id)
            for visible_id, event in enumerate(visible, start=1)
        ]

    def _visible_internal_events(self, viewer: str) -> list[dict[str, Any]]:
        return [
            event
            for event in self._events
            if event["timeline_id"] == self._timeline_id
            and ("*" in event["visibility"] or viewer in event["visibility"])
        ]

    @staticmethod
    def _agent_event_view(
        event: dict[str, Any], viewer: str, visible_id: int
    ) -> dict[str, Any]:
        item = {
            "event_id": visible_id,
            "sequence": visible_id,
            "actor": event["actor"],
            "kind": event["kind"],
            "input": copy.deepcopy(event["input"]),
            "accepted": event["accepted"],
            "result_summary": copy.deepcopy(event["result_summary"]),
        }
        if viewer == event["actor"]:
            item["decision"] = copy.deepcopy(event["decision"])
        return item

    def act(self, action: Action) -> dict[str, Any]:
        if not isinstance(action, Action):
            raise TypeError("act 只接受 Action")
        before = copy.deepcopy(self._state)
        candidate = copy.deepcopy(self._state)
        try:
            self._validate_action_envelope(action)
            result = self._apply_action(candidate, action)
            self._finish_turn(candidate, action.actor, result)
            self._assert_invariants(candidate)
        except _Refusal as refusal:
            event = self._append_event(
                action,
                visibility=[action.actor] if action.actor in ROLES else [],
                accepted=False,
                result={"refusal": refusal.reason},
                result_summary={"refusal": refusal.reason},
                state_diff=[],
            )
            return self._action_receipt(event, action.actor)

        self._state = candidate
        event = self._append_event(
            action,
            visibility=list(ROLES),
            accepted=True,
            result=result,
            result_summary=self._result_summary(result),
            state_diff=_diff(before, candidate),
        )
        return self._action_receipt(event, action.actor)

    def _validate_action_envelope(self, action: Action) -> None:
        if action.actor not in ROLES:
            raise _Refusal("unknown_actor", "行动者不是交易参与者", actor=action.actor)
        if action.kind not in TRADE_ACTIONS:
            raise _Refusal("unknown_action", "未知交易行动", kind=action.kind)
        if self._state["status"] != "negotiating":
            raise _Refusal("terminal_state", "谈判已经结束", status=self._state["status"])
        if action.actor != self._state["next_actor"]:
            raise _Refusal(
                "wrong_turn",
                "尚未轮到该角色行动",
                expected=self._state["next_actor"],
                actor=action.actor,
            )
        visible = self._visible_internal_events(action.actor)
        for visible_id in action.decision.evidence_event_ids:
            if visible_id > len(visible):
                raise _Refusal(
                    "unobservable_evidence",
                    "决策引用了当前时间线不可见的事件",
                    event_id=visible_id,
                )

    def _apply_action(
        self, state: dict[str, Any], action: Action
    ) -> dict[str, Any]:
        if action.kind == "message":
            self._exact_keys(action.payload, {"text"})
            self._require_text(action.payload["text"])
            return {"delivered": True}
        if action.kind == "offer":
            expected = {"text", "price"}
            if self._scenario.contract is not None:
                expected |= {"delivery_days", "warranty_days"}
            self._exact_keys(action.payload, expected)
            self._require_text(action.payload["text"])
            price = self._require_price(action.payload["price"])
            if self._scenario.contract is not None:
                self._validate_terms(action.payload)
                self._validate_own_utility(action.actor, action.payload)
            elif action.actor == "buyer" and price > self._scenario.buyer_max_price:
                raise _Refusal("buyer_limit_exceeded", "买方报价超过私人最高愿付价")
            elif action.actor == "seller" and price < self._scenario.seller_min_price:
                raise _Refusal("seller_limit_breached", "卖方报价低于私人最低接受价")
            current_id = state["active_offer_id"]
            if current_id is not None:
                state["offers"][current_id]["status"] = "superseded"
            state["offer_serial"] += 1
            offer_id = f"o{state['offer_serial']:04d}"
            state["offers"][offer_id] = {
                "id": offer_id,
                "actor": action.actor,
                "price": price,
                "status": "active",
                "round": self._current_round(),
                "source": "offer",
            }
            if self._scenario.contract is not None:
                state["offers"][offer_id].update({k: action.payload[k] for k in ("delivery_days", "warranty_days")})
            state["active_offer_id"] = offer_id
            return {"offer_id": offer_id, "price": price, "status": "active",
                    **({k: action.payload[k] for k in ("delivery_days", "warranty_days")}
                       if self._scenario.contract else {})}
        if action.kind in ("accept", "reject"):
            self._exact_keys(action.payload, {"text", "offer_id"})
            self._require_text(action.payload["text"])
            offer = self._opposing_active_offer(state, action.actor, action.payload["offer_id"])
            if action.kind == "reject":
                offer["status"] = "rejected"
                state["active_offer_id"] = None
                return {"offer_id": offer["id"], "status": "rejected"}
            if self._scenario.contract is not None:
                self._validate_own_utility(action.actor, offer)
            elif action.actor == "buyer" and offer["price"] > self._scenario.buyer_max_price:
                raise _Refusal("buyer_limit_exceeded", "买方不能接受超过最高愿付价的报价")
            elif action.actor == "seller" and offer["price"] < self._scenario.seller_min_price:
                raise _Refusal("seller_limit_breached", "卖方不能接受低于最低接受价的报价")
            offer["status"] = "accepted"
            state["status"] = "deal"
            state["termination_reason"] = "explicit_acceptance"
            state["deal_price"] = offer["price"]
            if self._scenario.contract is not None:
                state["deal_terms"] = {k: offer[k] for k in ("price", "delivery_days", "warranty_days")}
            state["active_offer_id"] = None
            return {"offer_id": offer["id"], "status": "deal", "deal_price": offer["price"],
                    **({"deal_terms": copy.deepcopy(state["deal_terms"])} if self._scenario.contract else {})}
        if action.kind == "exit":
            self._exact_keys(action.payload, {"text"})
            self._require_text(action.payload["text"])
            state["status"] = "exited"
            state["termination_reason"] = f"{action.actor}_exit"
            if state["active_offer_id"] is not None:
                state["offers"][state["active_offer_id"]]["status"] = "expired"
            state["active_offer_id"] = None
            return {"status": "exited", "by": action.actor}
        raise AssertionError("unreachable")

    def _finish_turn(
        self, state: dict[str, Any], actor: str, result: dict[str, Any]
    ) -> None:
        state["turns_taken"] += 1
        state["rounds"] = (state["turns_taken"] + 1) // 2
        if state["status"] != "negotiating":
            state["next_actor"] = None
        elif actor == "buyer":
            state["next_actor"] = "seller"
        elif state["rounds"] >= self._scenario.max_rounds:
            state["status"] = "timeout"
            state["termination_reason"] = "max_rounds"
            if state["active_offer_id"] is not None:
                state["offers"][state["active_offer_id"]]["status"] = "expired"
            state["active_offer_id"] = None
            state["next_actor"] = None
            result["status"] = "timeout"
        else:
            state["next_actor"] = "buyer"
        result["rounds"] = state["rounds"]

    @staticmethod
    def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
        allowed = ("delivered", "offer_id", "price", "status", "deal_price", "by", "rounds",
                   "delivery_days", "warranty_days", "deal_terms")
        return {key: copy.deepcopy(result[key]) for key in allowed if key in result}

    def _validate_terms(self, terms: dict[str, Any]) -> None:
        c = self._scenario.contract
        p, d, w = (terms[k] for k in ("price", "delivery_days", "warranty_days"))
        if (type(p) is not int or not c.price_min <= p <= c.price_max
                or (p - c.price_min) % c.price_step or type(d) is not int
                or d not in c.delivery_days or type(w) is not int or w not in c.warranty_days):
            raise _Refusal("invalid_contract", "条款必须符合公开合同网格")

    def _validate_own_utility(self, role: str, terms: dict[str, Any]) -> None:
        pref = getattr(self._scenario.contract, role)
        minimum = getattr(pref, "minimum_utility", 0)
        if self._utility(role, **{k: terms[k] for k in ("price", "delivery_days", "warranty_days")}) < minimum:
            message = ("正式报价和接受不得使自身收益低于私人最低可接受效用" if isinstance(pref, DiscretePreferences)
                       else "正式报价和接受不得使自身收益低于退出收益 0")
            raise _Refusal("negative_own_utility", message)

    @staticmethod
    def _require_text(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise _Refusal("invalid_text", "公开发言不能为空")
        return value

    @staticmethod
    def _require_price(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _Refusal("invalid_price", "价格必须是正整数")
        return value

    @staticmethod
    def _exact_keys(payload: dict[str, Any], expected: set[str]) -> None:
        if set(payload) != expected:
            raise _Refusal(
                "invalid_payload",
                "payload 字段不符合行动契约",
                expected=sorted(expected),
                received=sorted(payload),
            )

    @staticmethod
    def _opposing_active_offer(
        state: dict[str, Any], actor: str, offer_id: object
    ) -> dict[str, Any]:
        if not isinstance(offer_id, str) or offer_id not in state["offers"]:
            raise _Refusal("unknown_offer", "报价不存在", offer_id=offer_id)
        offer = state["offers"][offer_id]
        if offer["status"] != "active" or state["active_offer_id"] != offer_id:
            raise _Refusal("inactive_offer", "只能回应当前有效报价", offer_id=offer_id)
        if offer["actor"] == actor:
            raise _Refusal("own_offer", "不能接受或拒绝自己的报价")
        return offer

    def _append_event(
        self,
        action: Action,
        *,
        visibility: list[str],
        accepted: bool,
        result: dict[str, Any],
        result_summary: dict[str, Any],
        state_diff: list[dict[str, Any]],
    ) -> dict[str, Any]:
        event_id = len(self._events) + 1
        event = {
            "event_id": event_id,
            "sequence": event_id,
            "timeline_id": self._timeline_id,
            "timeline_sequence": 1
            + sum(event["timeline_id"] == self._timeline_id for event in self._events),
            "actor": action.actor,
            "kind": action.kind,
            "visibility": visibility,
            "input": _canonical(action.payload),
            "decision": _canonical(action.decision.to_dict()),
            "accepted": accepted,
            "result": _canonical(result),
            "result_summary": _canonical(result_summary),
            "state_diff": _canonical(state_diff),
        }
        self._events.append(event)
        return event

    def _action_receipt(self, event: dict[str, Any], actor: str) -> dict[str, Any]:
        if actor not in ROLES:
            return {
                "event_id": None,
                "sequence": None,
                "actor": actor,
                "kind": event["kind"],
                "input": copy.deepcopy(event["input"]),
                "decision": copy.deepcopy(event["decision"]),
                "accepted": event["accepted"],
                "result": copy.deepcopy(event["result_summary"]),
            }
        visible = self._visible_internal_events(actor)
        visible_id = visible.index(event) + 1
        receipt = self._agent_event_view(event, actor, visible_id)
        receipt["result"] = receipt.pop("result_summary")
        return receipt

    def reset(self) -> dict[str, Any]:
        before = copy.deepcopy(self._state)
        self._state = json.loads(self._s0_json)
        self._assert_invariants(self._state)
        next_timeline = self._timeline_id + 1
        event_id = len(self._events) + 1
        event = {
            "event_id": event_id,
            "sequence": event_id,
            "timeline_id": self._timeline_id,
            "timeline_sequence": 1
            + sum(event["timeline_id"] == self._timeline_id for event in self._events),
            "actor": AUDITOR,
            "kind": "reset",
            "visibility": [AUDITOR],
            "input": {},
            "decision": {
                "intent": "恢复交易世界 S0 并开始下一世",
                "known_facts": ["当前谈判时间线已经结束"],
                "uncertainty": "无",
                "evidence_event_ids": [],
                "known_facts_source": "actor_self_report",
            },
            "accepted": True,
            "result": {
                "restored_s0": True,
                "s0_fingerprint": self.s0_fingerprint,
                "next_timeline_id": next_timeline,
            },
            "result_summary": {"restored_s0": True},
            "state_diff": _diff(before, self._state),
        }
        self._events.append(event)
        self._timeline_id = next_timeline
        return copy.deepcopy(event)

    def capture_memory(
        self, role: str, event_ids: Iterable[int] | None = None
    ) -> TradeMemory:
        if role not in ROLES:
            raise PermissionError("只能为交易参与者导出授权记忆")
        visible = self.export_events(role)
        if event_ids is not None:
            selected = list(event_ids)
            if any(isinstance(item, bool) or not isinstance(item, int) for item in selected):
                raise ValueError("记忆事件编号必须是整数")
            by_id = {event["event_id"]: event for event in visible}
            if any(item not in by_id for item in selected):
                raise PermissionError("不能选择角色未见过的事件")
            visible = [copy.deepcopy(by_id[item]) for item in selected]
        own_fact = (
            f"我的最高愿付价是 {self._scenario.buyer_max_price} 元。"
            if role == "buyer"
            else f"我的最低接受价是 {self._scenario.seller_min_price} 元。"
        )
        if self._scenario.contract is not None:
            own_fact = json.dumps(self.observe(role)["private"], ensure_ascii=False, sort_keys=True)
        role_metrics = self.metrics(role)
        outcome = {
            key: role_metrics[key]
            for key in ("status", "deal", "deal_price", "rounds", "own_utility")
        }
        if self._scenario.contract is not None:
            outcome["deal_terms"] = copy.deepcopy(role_metrics["deal_terms"])
        return TradeMemory(
            role=role,
            source_timeline_id=self._timeline_id,
            visible_events=tuple(copy.deepcopy(visible)),
            outcome=outcome,
            own_private_fact=own_fact,
        )

    @classmethod
    def replay(
        cls, scenario: TradeScenario, events: Iterable[dict[str, Any]]
    ) -> TradeWorld:
        world = cls(scenario)
        for expected_id, source in enumerate(events, start=1):
            expected = copy.deepcopy(source)
            if expected.get("event_id") != expected_id or expected.get("sequence") != expected_id:
                raise TradeReplayError("审计事件编号必须从 1 开始且连续")
            if expected.get("kind") == "reset":
                actual = world.reset()
            else:
                try:
                    action = Action.from_dict(
                        {
                            "kind": expected["kind"],
                            "actor": expected["actor"],
                            "payload": expected["input"],
                            "decision": expected["decision"],
                        }
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise TradeReplayError(f"交易行动信封无效: {exc}") from exc
                world.act(action)
                actual = copy.deepcopy(world._events[-1])
            if actual != expected:
                raise TradeReplayError(f"交易事件 {expected_id} 无法精确回放")
        return world

    def _assert_invariants(self, state: dict[str, Any]) -> None:
        if state["turns_taken"] < 0:
            raise AssertionError("行动次数不可为负")
        expected_rounds = (state["turns_taken"] + 1) // 2 if state["turns_taken"] else 0
        if state["rounds"] != expected_rounds or state["rounds"] > self._scenario.max_rounds:
            raise AssertionError("谈判轮次不一致")
        active = [offer for offer in state["offers"].values() if offer["status"] == "active"]
        if state["active_offer_id"] is None:
            if active:
                raise AssertionError("存在未登记的有效报价")
        elif len(active) != 1 or active[0]["id"] != state["active_offer_id"]:
            raise AssertionError("有效报价必须唯一")
        if state["status"] == "negotiating":
            expected_actor = "buyer" if state["turns_taken"] % 2 == 0 else "seller"
            if state["next_actor"] != expected_actor:
                raise AssertionError("行动顺序不一致")
        elif state["next_actor"] is not None:
            raise AssertionError("终止状态不可继续行动")
        if state["status"] == "deal":
            price = state["deal_price"]
            if self._scenario.contract is not None:
                terms = state["deal_terms"]
                self._validate_terms(terms)
                if terms["price"] != price or any(self._utility(r, **terms) < getattr(getattr(self._scenario.contract, r), "minimum_utility", 0) for r in ROLES):
                    raise AssertionError("合同必须与成交价一致，且满足双方个体理性")
            elif not self._scenario.seller_min_price <= price <= self._scenario.buyer_max_price:
                raise AssertionError("成交价格违反双方硬约束")
        elif state["deal_price"] is not None:
            raise AssertionError("未成交状态不可出现成交价")
        if self._scenario.contract is not None and state["status"] != "deal" and state["deal_terms"] is not None:
            raise AssertionError("未成交状态不可出现合同")
