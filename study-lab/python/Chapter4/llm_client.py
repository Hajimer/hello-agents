"""第四章共用的 LLM 调用层；密钥通过 LLM_API_KEY 环境变量提供。"""

import os

import requests


class HelloAgentsLLM:
    """调用公司的 Chat Completions 接口，返回助手回复的文本。"""

    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        api_key: str | None = None,
        base_url: str = "https://oneapi-comate.baidu-int.com/v1",
        timeout: float = 60,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not self.api_key or not self.api_key.strip():
            raise ValueError("请先设置 LLM_API_KEY 环境变量，或传入 api_key。")
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0。")

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        """发送一次请求；失败时抛出异常，由调用者决定如何处理。"""
        # 默认仅发送公司示例中已有的字段，避免假定网关支持额外参数。
        payload = {"model": self.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature

        with requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=payload,
            timeout=self.timeout,
        ) as response:
            # 先检查 HTTP 状态，避免将认证失败等错误误当成正常模型回复。
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError("LLM 接口未返回有效的 JSON。") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "LLM 响应缺少 choices[0].message.content，请检查接口和模型配置。"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM 未返回有效的文本回复。")
        return content


if __name__ == "__main__":
    llm = HelloAgentsLLM(api_key='sk-wARsv1CCVfHnqxm581B5339339654c26B3E991AfEe8cC4Ba')
    print(llm.think([{"role": "user", "content": "你好，请用一句话介绍自己。"}]))
