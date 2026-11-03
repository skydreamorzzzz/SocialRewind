from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from typing import Any, Iterable

from .judge import QUALITY_LEVELS, Judge, JudgeOutputError, validate_verdict
from .models import Action, Scenario


AUDITOR = "auditor"
PRIVATE_INTERACTIONS = {"send_message", "propose_commitment", "respond_commitment"}
ACTION_KINDS = PRIVATE_INTERACTIONS | {"submit_work", "advance_day"}


class ReplayError(RuntimeError):
    pass


class _Refusal(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = {"code": code, "message": message, "details": details}


class _ReplayOnlyJudge:
    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        raise ReplayError("回放不应在线调用 Judge")


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _diff(before: Any, after: Any, path: str = "$") -> list[dict[str, Any]]:
    if type(before) is not type(after):
        return [{"path": path, "before": before, "after": after}]
    if isinstance(before, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}"
            if key not in before:
                changes.append({"path": child_path, "before": None, "after": after[key]})
            elif key not in after:
                changes.append({"path": child_path, "before": before[key], "after": None})
            else:
                changes.extend(_diff(before[key], after[key], child_path))
        return changes
    if isinstance(before, list):
        return [] if before == after else [{"path": path, "before": before, "after": after}]
    return [] if before == after else [{"path": path, "before": before, "after": after}]


class World:
    """硬约束、私密观察、审计日志与确定性回放的最小边界。"""

    def __init__(self, scenario: Scenario, judge: Judge) -> None:
        if not scenario.fictional:
            raise ValueError("本原型只接受明确标记为虚构的数据")
        self._scenario = copy.deepcopy(scenario)
        initial = self._make_initial_state(scenario)
        self._s0_json = json.dumps(initial, ensure_ascii=False, sort_keys=True)
        self._state: dict[str, Any] = json.loads(self._s0_json)
        self._judge = judge
        self._events: list[dict[str, Any]] = []
        self._assert_invariants(self._state)

    @staticmethod
    def _make_initial_state(scenario: Scenario) -> dict[str, Any]:
        tasks = {
            task_id: {
                "id": task_id,
                "title": spec["title"],
                "deadline_day": spec["deadline_day"],
                "target_quality": spec["target_quality"],
                "status": "open",
                "assignee": None,
                "commitment_id": None,
                "submission_id": None,
                "effective_hours": None,
                "quality": None,
            }
            for task_id, spec in scenario.tasks.items()
        }
        return {
            "scenario_id": scenario.scenario_id,
            "day": 1,
            "budget": {
                "total": scenario.project_budget,
                "available": scenario.project_budget,
                "reserved": 0,
                "spent": 0,
            },
            "tasks": tasks,
            "commitments": {},
            "submissions": {},
            "hours_used": {},
            "interaction_count_by_day": {"1": 0},
        }

    @property
    def s0_fingerprint(self) -> str:
        return hashlib.sha256(self._s0_json.encode("utf-8")).hexdigest()

    def audit_state(self, viewer: str) -> dict[str, Any]:
        if viewer != AUDITOR:
            raise PermissionError("完整世界状态仅实验审计者可读")
        return copy.deepcopy(self._state)

    def observe(self, viewer: str) -> dict[str, Any]:
        if viewer != AUDITOR and viewer not in self._scenario.roles:
            raise PermissionError("未知观察者")
        if viewer == AUDITOR:
            return {
                "scenario": self._scenario_view(),
                "state": copy.deepcopy(self._state),
                "events": self.export_events(AUDITOR),
            }

        role = self._scenario.roles[viewer]
        public_tasks = {
            task_id: {
                key: task[key]
                for key in (
                    "title",
                    "deadline_day",
                    "target_quality",
                    "status",
                    "assignee",
                    "effective_hours",
                    "quality",
                )
            }
            for task_id, task in self._state["tasks"].items()
        }
        own_commitments = {
            key: copy.deepcopy(value)
            for key, value in self._state["commitments"].items()
            if viewer in (value["proposer"], value["recipient"])
        }
        own_submissions = {
            key: copy.deepcopy(value)
            for key, value in self._state["submissions"].items()
            if value["actor"] == viewer
        }
        return {
            "scenario_id": self._scenario.scenario_id,
            "fictional": True,
            "day": self._state["day"],
            "public_facts": list(self._scenario.public_facts),
            "public_budget": copy.deepcopy(self._state["budget"]),
            "tasks": public_tasks,
            "self": {
                "id": viewer,
                "name": role["name"],
                "goal": role["goal"],
                "permissions": list(role["permissions"]),
                "private_facts": list(self._scenario.private_facts[viewer]),
                "availability": copy.deepcopy(role["availability"]),
            },
            "commitments": own_commitments,
            "submissions": own_submissions,
            "events": self.export_events(viewer),
        }

    def _scenario_view(self) -> dict[str, Any]:
        return _canonical(asdict(self._scenario))

    def export_events(self, viewer: str) -> list[dict[str, Any]]:
        if viewer != AUDITOR and viewer not in self._scenario.roles:
            raise PermissionError("未知日志读取者")
        if viewer == AUDITOR:
            return copy.deepcopy(self._events)
        visible: list[dict[str, Any]] = []
        for event in self._events:
            if "*" not in event["visibility"] and viewer not in event["visibility"]:
                continue
            item = {
                "event_id": event["event_id"],
                "sequence": event["sequence"],
                "actor": event["actor"],
                "kind": event["kind"],
                "visibility": list(event["visibility"]),
                "input": copy.deepcopy(event["input"]),
                "accepted": event["accepted"],
                "result_summary": copy.deepcopy(event["result_summary"]),
            }
            if viewer == event["actor"]:
                item["decision"] = copy.deepcopy(event["decision"])
            visible.append(item)
        return visible

    def act(
        self,
        action: Action,
        *,
        _recorded_judgments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(action, Action):
            raise TypeError("act 只接受 Action")
        before = copy.deepcopy(self._state)
        candidate = copy.deepcopy(self._state)
        visibility = self._visibility_for(action)
        try:
            if action.actor not in self._scenario.roles:
                raise _Refusal("unknown_actor", "行动者不在场景中", actor=action.actor)
            if action.kind not in ACTION_KINDS:
                raise _Refusal("unknown_action", "未知行动类型", kind=action.kind)
            if action.kind in PRIVATE_INTERACTIONS:
                self._consume_interaction(candidate)
            result = self._dispatch(candidate, action, _recorded_judgments)
            self._assert_invariants(candidate)
        except _Refusal as refusal:
            event = self._append_event(
                action=action,
                visibility=visibility,
                accepted=False,
                result={"refusal": refusal.reason},
                result_summary={"refusal": refusal.reason},
                state_diff=[],
            )
            return copy.deepcopy(event)

        self._state = candidate
        state_diff = _diff(before, candidate)
        summary = self._result_summary(action.kind, result)
        event = self._append_event(
            action=action,
            visibility=visibility,
            accepted=True,
            result=result,
            result_summary=summary,
            state_diff=state_diff,
        )
        return copy.deepcopy(event)

    def _consume_interaction(self, state: dict[str, Any]) -> None:
        day = str(state["day"])
        used = state["interaction_count_by_day"].get(day, 0)
        if used >= self._scenario.max_daily_interactions:
            raise _Refusal(
                "interaction_limit",
                "已达到当日私下交互上限",
                day=state["day"],
                limit=self._scenario.max_daily_interactions,
            )
        state["interaction_count_by_day"][day] = used + 1

    def _dispatch(
        self,
        state: dict[str, Any],
        action: Action,
        recorded: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if action.kind == "send_message":
            return self._send_message(state, action)
        if action.kind == "propose_commitment":
            return self._propose_commitment(state, action)
        if action.kind == "respond_commitment":
            return self._respond_commitment(state, action)
        if action.kind == "submit_work":
            return self._submit_work(state, action)
        if action.kind == "advance_day":
            return self._advance_day(state, action, recorded)
        raise AssertionError("unreachable")

    def _send_message(self, state: dict[str, Any], action: Action) -> dict[str, Any]:
        self._exact_keys(action.payload, {"to", "content"})
        target = action.payload["to"]
        content = action.payload["content"]
        self._require_other_role(action.actor, target)
        if not isinstance(content, str) or not content.strip():
            raise _Refusal("invalid_payload", "私信内容不能为空")
        return {"delivered": True, "to": target, "day_unchanged": state["day"]}

    def _propose_commitment(self, state: dict[str, Any], action: Action) -> dict[str, Any]:
        self._exact_keys(action.payload, {"to", "task_id", "amount", "terms"})
        role = self._scenario.roles[action.actor]
        if "propose_budget_commitment" not in role["permissions"]:
            raise _Refusal("permission_denied", "行动者无权承诺项目预算")
        target = action.payload["to"]
        self._require_other_role(action.actor, target)
        task_id = action.payload["task_id"]
        if task_id not in state["tasks"]:
            raise _Refusal("unknown_task", "任务不存在", task_id=task_id)
        task = state["tasks"][task_id]
        if task["status"] != "open":
            raise _Refusal("task_unavailable", "任务已被承诺或结算", status=task["status"])
        amount = action.payload["amount"]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise _Refusal("invalid_amount", "金额必须是正整数")
        if amount > state["budget"]["available"]:
            raise _Refusal(
                "insufficient_budget",
                "承诺金额超过可用预算",
                requested=amount,
                available=state["budget"]["available"],
            )
        terms = action.payload["terms"]
        if not isinstance(terms, str) or not terms.strip():
            raise _Refusal("invalid_payload", "承诺条款不能为空")
        availability = self._scenario.roles[target]["availability"].get(str(state["day"]), 0)
        if availability <= 0 or state["day"] > task["deadline_day"]:
            raise _Refusal("schedule_conflict", "接收方在截止日前没有可用档期")

        commitment_id = f"c{len(state['commitments']) + 1:04d}"
        state["commitments"][commitment_id] = {
            "id": commitment_id,
            "proposer": action.actor,
            "recipient": target,
            "task_id": task_id,
            "amount": amount,
            "terms": terms,
            "status": "proposed",
            "proposed_day": state["day"],
            "responded_day": None,
        }
        state["budget"]["available"] -= amount
        state["budget"]["reserved"] += amount
        return {"commitment_id": commitment_id, "status": "proposed"}

    def _respond_commitment(self, state: dict[str, Any], action: Action) -> dict[str, Any]:
        self._exact_keys(action.payload, {"commitment_id", "response"})
        commitment_id = action.payload["commitment_id"]
        commitment = state["commitments"].get(commitment_id)
        if commitment is None:
            raise _Refusal("unknown_commitment", "承诺不存在", commitment_id=commitment_id)
        if action.actor != commitment["recipient"]:
            raise _Refusal("permission_denied", "只有承诺接收方可以回应")
        if commitment["status"] != "proposed":
            raise _Refusal("already_resolved", "承诺已经回应", status=commitment["status"])
        response = action.payload["response"]
        if response not in ("accept", "reject"):
            raise _Refusal("invalid_response", "response 只能是 accept 或 reject")
        amount = commitment["amount"]
        commitment["responded_day"] = state["day"]
        state["budget"]["reserved"] -= amount
        if response == "reject":
            commitment["status"] = "rejected"
            state["budget"]["available"] += amount
        else:
            task = state["tasks"][commitment["task_id"]]
            if task["status"] != "open":
                raise _Refusal("task_unavailable", "任务不再可接受", status=task["status"])
            commitment["status"] = "accepted"
            state["budget"]["spent"] += amount
            task["status"] = "assigned"
            task["assignee"] = action.actor
            task["commitment_id"] = commitment_id
        return {"commitment_id": commitment_id, "status": commitment["status"]}

    def _submit_work(self, state: dict[str, Any], action: Action) -> dict[str, Any]:
        self._exact_keys(action.payload, {"task_id", "claimed_hours", "evidence"})
        task_id = action.payload["task_id"]
        task = state["tasks"].get(task_id)
        if task is None:
            raise _Refusal("unknown_task", "任务不存在", task_id=task_id)
        if task["status"] != "assigned" or task["assignee"] != action.actor:
            raise _Refusal("permission_denied", "只有已接受任务的执行者可以提交")
        if state["day"] > task["deadline_day"]:
            raise _Refusal("deadline_passed", "任务截止日期已过")
        hours = action.payload["claimed_hours"]
        if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0:
            raise _Refusal("invalid_hours", "claimed_hours 必须是正整数")
        day = str(state["day"])
        availability = self._scenario.roles[action.actor]["availability"].get(day, 0)
        used = state["hours_used"].get(action.actor, {}).get(day, 0)
        if used + hours > availability:
            raise _Refusal(
                "insufficient_time",
                "提交工时超过当日可用档期",
                claimed=hours,
                remaining=availability - used,
            )
        evidence = action.payload["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise _Refusal("missing_evidence", "至少需要一条结构化证据")
        evidence_ids: list[str] = []
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"id", "kind", "description"}:
                raise _Refusal("invalid_evidence", "证据字段必须为 id/kind/description")
            if any(not isinstance(item[key], str) or not item[key].strip() for key in item):
                raise _Refusal("invalid_evidence", "证据字段不能为空")
            evidence_ids.append(item["id"])
        if len(set(evidence_ids)) != len(evidence_ids):
            raise _Refusal("duplicate_evidence", "证据编号不可重复")

        submission_id = f"s{len(state['submissions']) + 1:04d}"
        submission = {
            "id": submission_id,
            "task_id": task_id,
            "actor": action.actor,
            "submitted_day": state["day"],
            "claimed_hours": hours,
            "evidence": copy.deepcopy(evidence),
            "status": "pending",
            "judge_verdict": None,
            "evaluation_error": None,
        }
        state["submissions"][submission_id] = submission
        state["hours_used"].setdefault(action.actor, {})[day] = used + hours
        task["status"] = "submitted"
        task["submission_id"] = submission_id
        return {"submission_id": submission_id, "status": "pending"}

    def _advance_day(
        self,
        state: dict[str, Any],
        action: Action,
        recorded: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        self._exact_keys(action.payload, set())
        if "advance_time" not in self._scenario.roles[action.actor]["permissions"]:
            raise _Refusal("permission_denied", "行动者无权推进世界时间")
        records: list[dict[str, Any]] = []
        pending = [
            submission
            for submission in state["submissions"].values()
            if submission["status"] == "pending"
        ]
        if recorded is not None and len(recorded) != len(pending):
            raise ReplayError("记录的 Judge 结果数量与待结算项不一致")

        for index, submission in enumerate(pending):
            request = self._judge_request(state, submission)
            if recorded is None:
                try:
                    verdict = validate_verdict(request, self._judge.evaluate(copy.deepcopy(request)))
                    record = {"status": "ok", "input": request, "output": verdict}
                except Exception as exc:  # Judge 是不可信边界，必须变成显式失败记录。
                    record = {
                        "status": "failed",
                        "input": request,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
            else:
                source = copy.deepcopy(recorded[index])
                if source.get("input") != request:
                    raise ReplayError("Judge 输入与记录不一致")
                if source.get("status") == "ok":
                    try:
                        verdict = validate_verdict(request, source.get("output"))
                    except JudgeOutputError as exc:
                        raise ReplayError(f"记录的 Judge 输出无效: {exc}") from exc
                    record = {"status": "ok", "input": request, "output": verdict}
                elif source.get("status") == "failed" and isinstance(source.get("error"), dict):
                    record = source
                else:
                    raise ReplayError("记录的 Judge 状态无效")

            records.append(record)
            task = state["tasks"][submission["task_id"]]
            if record["status"] == "failed":
                submission["status"] = "evaluation_failed"
                submission["evaluation_error"] = copy.deepcopy(record["error"])
                task["status"] = "evaluation_failed"
                continue

            verdict = record["output"]
            submission["status"] = "judged"
            submission["judge_verdict"] = copy.deepcopy(verdict)
            task["effective_hours"] = verdict["effective_hours"]
            task["quality"] = verdict["quality"]
            target_index = QUALITY_LEVELS.index(task["target_quality"])
            quality_index = QUALITY_LEVELS.index(verdict["quality"])
            task["status"] = (
                "completed"
                if verdict["effective_hours"] > 0 and quality_index >= target_index
                else "needs_revision"
            )

        old_day = state["day"]
        for task in state["tasks"].values():
            if task["deadline_day"] <= old_day and task["status"] in ("open", "assigned"):
                task["status"] = "missed_deadline"
        state["day"] += 1
        state["interaction_count_by_day"].setdefault(str(state["day"]), 0)
        return {"advanced_to_day": state["day"], "judge_records": records}

    def _judge_request(
        self, state: dict[str, Any], submission: dict[str, Any]
    ) -> dict[str, Any]:
        task = state["tasks"][submission["task_id"]]
        return {
            "submission_id": submission["id"],
            "task_id": task["id"],
            "claimed_hours": submission["claimed_hours"],
            "evidence": copy.deepcopy(submission["evidence"]),
            "rubric": {
                "quality_levels": list(QUALITY_LEVELS),
                "target_quality": task["target_quality"],
                "effective_hours_range": [0, submission["claimed_hours"]],
                "instruction": "仅评价给定证据；不得修改预算、权限、时间或世界状态。",
            },
        }

    def reset(self) -> dict[str, Any]:
        before = copy.deepcopy(self._state)
        self._state = json.loads(self._s0_json)
        self._assert_invariants(self._state)
        event = {
            "event_id": len(self._events) + 1,
            "sequence": len(self._events) + 1,
            "actor": AUDITOR,
            "kind": "reset",
            "visibility": ["*"],
            "input": {},
            "decision": {
                "intent": "恢复不可变初始状态 S0",
                "known_facts": ["实验时间线已结束"],
                "uncertainty": "无",
            },
            "accepted": True,
            "result": {"restored_s0": True, "s0_fingerprint": self.s0_fingerprint},
            "result_summary": {"restored_s0": True},
            "state_diff": _diff(before, self._state),
        }
        self._events.append(event)
        return copy.deepcopy(event)

    @classmethod
    def replay(cls, scenario: Scenario, events: Iterable[dict[str, Any]]) -> World:
        world = cls(scenario, _ReplayOnlyJudge())
        for expected_id, source in enumerate(events, start=1):
            expected = copy.deepcopy(source)
            if expected.get("event_id") != expected_id or expected.get("sequence") != expected_id:
                raise ReplayError("事件编号必须从 1 开始且连续")
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
                    raise ReplayError(f"事件行动信封无效: {exc}") from exc
                records = None
                if expected["kind"] == "advance_day" and expected.get("accepted"):
                    records = expected.get("result", {}).get("judge_records")
                    if not isinstance(records, list):
                        raise ReplayError("推进事件缺少 Judge 记录")
                actual = world.act(action, _recorded_judgments=records)
            if actual != expected:
                raise ReplayError(f"事件 {expected_id} 回放结果与日志不一致")
        return world

    def _append_event(
        self,
        *,
        action: Action,
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

    def _visibility_for(self, action: Action) -> list[str]:
        if action.kind == "advance_day":
            return ["*"]
        if action.kind in ("send_message", "propose_commitment"):
            target = action.payload.get("to")
            return list(dict.fromkeys([action.actor, target])) if isinstance(target, str) else [action.actor]
        if action.kind == "respond_commitment":
            commitment = self._state["commitments"].get(action.payload.get("commitment_id"))
            if commitment:
                return [commitment["proposer"], commitment["recipient"]]
        if action.kind == "submit_work":
            task = self._state["tasks"].get(action.payload.get("task_id"))
            if task and task.get("commitment_id"):
                commitment = self._state["commitments"][task["commitment_id"]]
                return [commitment["proposer"], commitment["recipient"]]
        return [action.actor]

    @staticmethod
    def _result_summary(kind: str, result: dict[str, Any]) -> dict[str, Any]:
        if kind == "advance_day":
            return {
                "advanced_to_day": result["advanced_to_day"],
                "settlements": [record["status"] for record in result["judge_records"]],
            }
        return copy.deepcopy(result)

    def _assert_invariants(self, state: dict[str, Any]) -> None:
        budget = state["budget"]
        if any(isinstance(budget[key], bool) or not isinstance(budget[key], int) for key in budget):
            raise AssertionError("预算必须使用整数")
        if min(budget["available"], budget["reserved"], budget["spent"]) < 0:
            raise AssertionError("预算账户不可为负")
        if budget["available"] + budget["reserved"] + budget["spent"] != budget["total"]:
            raise AssertionError("预算不守恒")
        if state["day"] < 1:
            raise AssertionError("模拟日不可小于 1")
        for commitment in state["commitments"].values():
            if commitment["proposer"] not in self._scenario.roles:
                raise AssertionError("承诺提出者不存在")
            if commitment["recipient"] not in self._scenario.roles:
                raise AssertionError("承诺接收者不存在")
            if commitment["status"] not in ("proposed", "accepted", "rejected"):
                raise AssertionError("承诺状态无效")
        for task in state["tasks"].values():
            if task["target_quality"] not in QUALITY_LEVELS:
                raise AssertionError("任务目标质量无效")

    @staticmethod
    def _exact_keys(payload: dict[str, Any], expected: set[str]) -> None:
        if set(payload) != expected:
            raise _Refusal(
                "invalid_payload",
                "payload 字段不符合行动契约",
                expected=sorted(expected),
                received=sorted(payload),
            )

    def _require_other_role(self, actor: str, target: object) -> None:
        if not isinstance(target, str) or target not in self._scenario.roles:
            raise _Refusal("unknown_recipient", "接收方不存在", recipient=target)
        if actor == target:
            raise _Refusal("invalid_recipient", "不能向自己发起该行动")
