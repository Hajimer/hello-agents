"""纯内存 Mock 业务系统；不连接订单系统、不转账、不发送真实邮件。"""
from copy import deepcopy


class MockTools:
    def __init__(self):
        self.orders = {
            "O1001": {"user_id": "U001", "amount": 199, "status": "unshipped"},
            "O1002": {"user_id": "U001", "amount": 299, "status": "signed"},
            "O1003": {"user_id": "U002", "amount": 99, "status": "unshipped"},
        }
        self.users = {"U001": "user001@example.test"}
        self.refunds = {}
        self.outbox = {}
        self.tickets = {}

    def get_order(self, user_id, order_id):
        order = self.orders.get(order_id)
        if not order or order["user_id"] != user_id:
            raise ValueError("订单不存在或不属于当前用户。")
        return deepcopy(order)

    def read(self, name, user_id, order_id):
        order = self.get_order(user_id, order_id)
        if name == "GetOrder":
            return order
        if name == "GetLogistics":
            return {"status": order["status"], "source": "Mock物流系统"}
        if name == "SearchPolicy":
            return {"id": "P1", "version": "demo-v1", "text": (
                "未发货订单允许全额退款。已签收订单仅凭当前信息不能自动审批，需人工复核。"
            )}
        raise ValueError("不允许的查询工具。")

    def submit_refund(self, user_id, order_id, amount):
        order = self.get_order(user_id, order_id)
        # 后端独立检查，不依赖模型声称已获批准。
        if order["status"] != "unshipped" or amount != order["amount"]:
            raise ValueError("不满足自动全额退款条件。")
        key = (user_id, order_id)
        if key not in self.refunds:
            self.refunds[key] = {"status": "mock_refunded", "amount": amount}
        return deepcopy(self.refunds[key])

    def create_ticket(self, user_id, order_id, reason):
        key = (user_id, order_id)
        self.tickets.setdefault(key, {"status": "human_review", "reason": reason})
        return deepcopy(self.tickets[key])

    def send_email(self, user_id, order_id, message):
        # 收件人取自可信用户资料，不允许 LLM 指定。
        self.get_order(user_id, order_id)
        key = (user_id, order_id)
        self.outbox.setdefault(key, {
            "to": self.users[user_id], **message, "status": "mock_sent",
        })
        print(f"📧 Mock 邮件：{self.outbox[key]}")
        return deepcopy(self.outbox[key])
