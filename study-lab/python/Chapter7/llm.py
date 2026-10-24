"""第七章的 LLM 配置与客户端创建入口。

公司服务通过 study-lab/python/.env 中的 LLM_BASE_URL、LLM_API_KEY
和 LLM_MODEL_ID 配置，也可直接使用进程中已有的环境变量。
导入本模块不会加载配置或创建客户端。
"""

from pathlib import Path

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM


def create_llm() -> HelloAgentsLLM:
    """加载配置并创建客户端；已有环境变量优先于 .env，不发起模型请求。"""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)
    return HelloAgentsLLM(provider="auto")


__all__ = ["HelloAgentsLLM", "create_llm"]
