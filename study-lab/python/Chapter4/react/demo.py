"""4.2.3 ReAct 运行示例。请从 Chapter4 目录以模块方式运行。"""

from pathlib import Path

from dotenv import load_dotenv

from llm_client import HelloAgentsLLM
from tools.search_tool import search
from tools.calculate_tool import calculate
from tools.tool_executor import ToolExecutor
from react.agent import ReActAgent


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    llm_client = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        "Search",
        "一个网页搜索引擎。当你需要回答时事、事实或模型知识库中没有的信息时使用。",
        search,
    )

    # 习题3. 注册计算器工具
    calculate_description = "计算器工具 计算字符串表达式，支持整数、浮点数、+、-、*、/、括号和正负号。例如：(123 + 456) * 789 / 12。输入不包含 = 或 ?。"
    tool_executor.registerTool("Calculate", calculate_description, calculate)

    agent = ReActAgent(llm_client, tool_executor)
    # agent.run("华为最新的手机是哪一款？它的主要卖点是什么？")
    agent.run("计算 (123 + 456) × 789/ 12 = ? 的结果")
