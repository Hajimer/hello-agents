from hello_agents import SimpleAgent, HelloAgentsLLM
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

llm = HelloAgentsLLM(provider="auto")

agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个有用的AI助手"
)

# 基础对话
response = agent.run("你好！请介绍一下自己")
print(response)

# 添加工具功能
from hello_agents.tools import CalculatorTool
calculator = CalculatorTool()

response = agent.run("请帮我计算 2 + 3 * 4")
print(response)

# 查看对话历史
print(f"历史消息数：{len(agent.get_history())}")