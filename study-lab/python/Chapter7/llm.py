"""第七章的 LLM 配置与客户端创建入口。

公司服务通过 study-lab/python/.env 中的 LLM_BASE_URL、LLM_API_KEY
和 LLM_MODEL_ID 配置，也可直接使用进程中已有的环境变量。
导入本模块不会加载配置或创建客户端。
"""

from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM as FrameworkHelloAgentsLLM
from hello_agents.core.exceptions import HelloAgentsException


class HelloAgentsLLM(FrameworkHelloAgentsLLM):
    """兼容流式响应中 choices 为空的数据块。"""

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """保留框架的流式接口，跳过没有文本选项的数据块。"""
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            try:
                print("✅ 大语言模型响应成功:")
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        print(content, end="", flush=True)
                        yield content
                print()
            finally:
                response.close()
        except Exception as e:
            raise HelloAgentsException(f"LLM调用失败: {e}") from e


def create_llm() -> HelloAgentsLLM:
    """加载配置并创建客户端；已有环境变量优先于 .env，不发起模型请求。"""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)
    return HelloAgentsLLM(provider="auto")


__all__ = ["HelloAgentsLLM", "create_llm"]
