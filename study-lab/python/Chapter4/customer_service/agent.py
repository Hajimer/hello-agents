"""学习用综合示例：Planner → Executor → ReAct / Reflection。"""
import json
import math

from customer_service import prompts

STAGES = ["understand", "collect", "decide", "review", "refund", "draft", "send"]
READ_TOOLS = {"GetOrder", "GetLogistics", "SearchPolicy"}


def dump(value):
    return json.dumps(value, ensure_ascii=False)


def ask(llm_client, prompt):
    """沿用 think(messages) 接口；只接受单个 JSON 对象。"""
    text = llm_client.think(messages=[{"role": "user", "content": prompt}])
    if not text:
        raise ValueError("LLM 未返回有效响应。")
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[len("```json"):-3].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("模型必须返回 JSON 对象。")
    return result


def validate_decision(decision):
    if decision.get("recommendation") not in {"approve", "reject", "need_info", "human_review"}:
        raise ValueError("无效的退款建议。")
    amount = decision.get("amount")
    confidence = decision.get("confidence")
    for value in (amount, confidence):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("金额和置信度必须是有限数字。")
    if amount < 0 or not 0 <= confidence <= 1:
        raise ValueError("金额或置信度范围错误。")
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        raise ValueError("缺少判断理由。")
    if not isinstance(decision.get("unresolved_issues"), list):
        raise ValueError("缺少争议列表。")
    if decision.get("policy_refs") != ["P1"]:
        raise ValueError("建议必须引用已查询的政策 P1。")
    return decision


def rule_issues(state):
    """Mock 自动审批仅覆盖未发货全额退款，其他场景转人工。"""
    evidence = state["evidence"]
    decision = state["decision"]
    issues = []
    if set(evidence) != READ_TOOLS:
        return ["必要证据不完整。"]
    order = evidence["GetOrder"]
    if order["status"] != "unshipped" or evidence["GetLogistics"]["status"] != "unshipped":
        issues.append("签收/配送状态需人工复核，不能自动批准或拒绝。")
    if decision["recommendation"] != "approve":
        issues.append("模型未给出可执行的自动批准建议。")
    if decision["amount"] != order["amount"]:
        issues.append("退款金额与订单全额不一致。")
    if decision["unresolved_issues"]:
        issues.append("模型报告存在未解决争议。")
    return issues


class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, request):
        output = ask(self.llm_client, prompts.PLANNER_PROMPT.format(request=request))
        steps = output.get("steps")
        if not isinstance(steps, list) or any(not isinstance(s, dict) for s in steps):
            raise ValueError("计划必须是步骤对象列表。")
        if [s.get("type") for s in steps] != STAGES:
            raise ValueError("计划必须包含按规定顺序排列的七个阶段。")
        if any(not isinstance(s.get("goal"), str) or not s["goal"].strip() for s in steps):
            raise ValueError("每个阶段必须具有具体目标。")
        return steps


class ReActAgent:
    def __init__(self, llm_client, tools, max_steps=6):
        self.llm_client = llm_client
        self.tools = tools
        self.max_steps = max_steps

    def run(self, state, goal):
        history = []
        for i in range(self.max_steps):
            print(f"--- 调查第 {i + 1} 步 ---")
            prompt = prompts.REACT_PROMPT.format(
                goal=goal, request=dump(state["request"]),
                evidence=dump(state["evidence"]), history=dump(history),
            )
            try:
                action = ask(self.llm_client, prompt)
                name = action.get("action")
                if name == "Finish":
                    if set(state["evidence"]) == READ_TOOLS:
                        return
                    raise ValueError("证据不足，必须完成三个查询。")
                if name not in READ_TOOLS or action.get("order_id") != state["order_id"]:
                    raise ValueError("工具未授权或订单号不匹配。")
                observation = self.tools.read(name, state["user_id"], state["order_id"])
                state["evidence"][name] = observation
                history.append({"action": action, "observation": observation})
            except (ValueError, TypeError, KeyError) as exc:
                history.append({"error": str(exc)})
            print(f"👀 {history[-1]}")
        raise ValueError("调查达到步数上限，证据未被确认完整。")


class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=2):
        self.llm_client = llm_client
        self.max_iterations = max_iterations

    def run(self, state):
        for i in range(self.max_iterations):
            print(f"--- 复核第 {i + 1} 轮 ---")
            state["rule_issues"] = rule_issues(state)
            feedback = ask(self.llm_client, prompts.REFLECT_PROMPT.format(case=dump(state)))
            state["reflection_records"].append(feedback)
            verdict = feedback.get("verdict")
            if verdict not in {"pass", "revise", "need_evidence", "human_review"}:
                raise ValueError("无效的复核结果。")
            if verdict in {"need_evidence", "human_review"}:
                return False
            if verdict == "pass":
                return not rule_issues(state)
            revised = ask(self.llm_client, prompts.REFINE_PROMPT.format(
                case=dump(state), feedback=dump(feedback),
            ))
            state["decision"] = validate_decision(revised)
        # 达到上限但未得到通过结论，不能按批准处理。
        return False


class Executor:
    def __init__(self, llm_client, tools):
        self.llm_client = llm_client
        self.tools = tools
        self.investigator = ReActAgent(llm_client, tools)
        self.reviewer = ReflectionAgent(llm_client)

    def execute(self, plan, state):
        for step in plan:
            kind = step["type"]
            print(f"\n--- {kind}: {step['goal']} ---")
            if kind == "understand":
                request = ask(self.llm_client, prompts.REQUEST_PROMPT.format(request=state["message"]))
                if not isinstance(request.get("reason"), str) or not request["reason"].strip():
                    raise ValueError("需要补充申请理由。")
                if request.get("order_id") != state["order_id"]:
                    raise ValueError("提取的订单号与当前案件不一致，请补充确认。")
                state["request"] = request
            elif kind == "collect":
                self.investigator.run(state, step["goal"])
            elif kind == "decide":
                state["decision"] = validate_decision(ask(
                    self.llm_client, prompts.DECISION_PROMPT.format(goal=step["goal"], case=dump(state)),
                ))
            elif kind == "review":
                decision = state["decision"]
                needs_review = rule_issues(state) or decision["confidence"] < 0.8
                if needs_review and not self.reviewer.run(state):
                    state["status"] = "human_review"
                    self.tools.create_ticket(state["user_id"], state["order_id"], "争议或证据不足")
                else:
                    state["status"] = "approved"
            elif kind == "refund":
                if state["status"] == "approved":
                    if rule_issues(state):
                        raise ValueError("执行前业务复核失败。")
                    state["refund_result"] = self.tools.submit_refund(
                        state["user_id"], state["order_id"], state["decision"]["amount"],
                    )
                    state["status"] = "mock_refunded"
            elif kind == "draft":
                draft = ask(self.llm_client, prompts.EMAIL_PROMPT.format(case=dump(state)))
                if any(not isinstance(draft.get(k), str) or not draft[k].strip() for k in ("subject", "body")):
                    raise ValueError("邮件主题或正文为空。")
                state["email_draft"] = draft
                # 模型草稿仅供演示查看；发送内容由可信模板构造，避免无依据承诺。
                if state["status"] == "mock_refunded":
                    body = f"您好，理解您的退款需求。模拟退款已完成，金额 {state['refund_result']['amount']} 元。"
                elif state["status"] == "human_review":
                    body = "您好，理解您的退款需求。申请已进入模拟人工审核，尚未批准或执行退款。"
                else:
                    raise ValueError("当前状态不允许发送通知。")
                state["email_message"] = {"subject": "退款申请处理通知（模拟）", "body": body}
            elif kind == "send":
                state["email_result"] = self.tools.send_email(
                    state["user_id"], state["order_id"], state["email_message"],
                )
            state["completed_steps"].append(kind)
        return state


class CustomerServiceAgent:
    def __init__(self, llm_client, tools):
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client, tools)
        self.tools = tools

    def run(self, message, user_id, order_id):
        # user_id 假定由登录系统提供；此示例由 demo 固定。
        state = {
            "message": message, "user_id": user_id, "order_id": order_id,
            "status": "processing", "evidence": {}, "reflection_records": [],
            "completed_steps": [],
        }
        try:
            self.tools.get_order(user_id, order_id)
            state["plan"] = self.planner.plan(message)
            self.executor.execute(state["plan"], state)
        except (ValueError, TypeError, KeyError) as exc:
            state["status"] = "stopped"
            state["error"] = str(exc)
            print(f"流程停止：{exc}")
        return state
