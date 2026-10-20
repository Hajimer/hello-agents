# 退款客服综合示例

真实 LLM 使用上一级 `llm_client.py` 的 `HelloAgentsLLM.think(messages)`，
配置来自 `study-lab/python/.env`，默认模型为 `gpt-5.6-sol`。
订单、物流、政策、退款、人工工单和邮件均为内存 Mock，不发生真实业务操作。

从仓库根目录进入：

```bash
cd study-lab/python/Chapter4
python -m customer_service.demo --scenario normal
python -m customer_service.demo --scenario dispute
python -m customer_service.demo --scenario forbidden
```

使用已有环境中的 `openai`、`python-dotenv`，本示例不增加第三方依赖。
运行会调用真实 LLM 并消耗配额；无需 SerpApi。

## 阅读顺序

1. `prompts.py`：规划、信息提取、ReAct、决策、反思、修订及邮件 Prompt。
2. `tools.py`：固定数据、查询与业务校验、Mock 写入操作。
3. `agent.py`：Planner → Executor → ReActAgent / ReflectionAgent。
4. `demo.py`：加载配置，选择场景，组装运行。

## 预期场景

- normal：O1001 未发货，199 元，条件满足时自动模拟全额退款。
- dispute：O1002 已签收但用户称未收到，必须人工审核；模型即使高置信批准也不能越过业务规则。
- forbidden：O1003 属于其他用户，在模型调用前拒绝访问。

LLM 仍可能返回无效 JSON 或不合规计划；程序会停止或限次纠错，不保证每次都完成正常场景。

## 学习实现的边界

- 七个阶段由程序约束，Planner 负责生成具体目标；不支持自由重排或动态重规划。
- ReAct 只调用查询工具；退款和邮件由 Executor 控制。
- 置信度低于 0.8 或规则冲突时进入 Reflection，最多两轮。
- need_evidence 或仍有争议时转人工；没有自动追问与案件恢复功能。
- Memory 是案件字典和反思记录列表，没有数据库持久化。
- 仅支持未发货全额自动退款；其他建议均转人工，不实现自动拒绝、部分退款。
- LLM 邮件草稿保存在 email_draft；实际 Mock 发送使用确定性状态模板。
- 退款和邮件去重仅在同一个 MockTools 实例生命周期内有效；重新启动会重置数据。
- 用户身份由 demo 固定提供；没有真实登录系统、可靠消息队列或生产级审计。

已使用临时 Fake LLM 离线检查正常退款、同实例重复请求去重、争议拦截、反思修订、订单归属、计划约束和查询工具白名单。
未进行真实 LLM 联调，未生成练习测试文件。
