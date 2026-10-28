"""第七章的 LLM 配置与客户端创建入口。

公司服务通过 study-lab/python/.env 中的 LLM_BASE_URL、LLM_API_KEY
和 LLM_MODEL_ID 配置，也可直接使用进程中已有的环境变量。
导入本模块不会加载配置或创建客户端。
"""

import os
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM as FrameworkHelloAgentsLLM
from hello_agents.core.exceptions import HelloAgentsException


class HelloAgentsLLM(FrameworkHelloAgentsLLM):
    """扩展公司 OpenAI 兼容网关，并兼容空 choices 流式数据块。"""

    # 已知的请求 ID，供调用方参考；不是限制模型范围的白名单。
    COMPANY_MODEL_IDS = (
        "gpt-5.6-sol",
        "Opus 5.5",
        "Gemini 4 Argon",
        "DeepSeek-V4-Pro",
        "Grok-4.7",
    )

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs,
    ):
        """显式参数优先于环境变量；自动模式优先使用完整的通用网关配置。

        本项目以 LLM_BASE_URL + LLM_API_KEY 表示公司网关配置，
        不根据模型名称推断官方供应商。直接构造时不加载 .env。
        """
        resolved_provider = provider
        if provider in (None, "auto"):
            gateway_url = base_url or os.getenv("LLM_BASE_URL")
            gateway_key = api_key or os.getenv("LLM_API_KEY")
            resolved_provider = "company" if gateway_url and gateway_key else None

        if resolved_provider == "company":
            model = model or os.getenv("LLM_MODEL_ID")
            if not model:
                raise HelloAgentsException("公司网关需要指定 model 或设置 LLM_MODEL_ID。")

        # None 才会触发当前安装版本父类的自动检测。
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider=resolved_provider,
            **kwargs,
        )

    def _resolve_credentials(
        self, api_key: Optional[str], base_url: Optional[str]
    ) -> tuple[str, str]:
        """公司网关仅使用通用配置，避免误用其他供应商的密钥或地址。"""
        if self.provider == "company":
            resolved_key = api_key or os.getenv("LLM_API_KEY")
            resolved_url = base_url or os.getenv("LLM_BASE_URL")
            if not resolved_key or not resolved_url:
                raise HelloAgentsException(
                    "公司网关需要 api_key / LLM_API_KEY 和 base_url / LLM_BASE_URL。"
                )
            return resolved_key, resolved_url
        return super()._resolve_credentials(api_key, base_url)

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


def create_llm(model: Optional[str] = None) -> HelloAgentsLLM:
    """加载配置并创建客户端；model 参数优先，已有环境变量优先于 .env。

    仅创建客户端，不发起模型请求。
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)
    return HelloAgentsLLM(model=model, provider="auto")


__all__ = ["HelloAgentsLLM", "create_llm"]
