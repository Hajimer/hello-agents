import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# 加载当前环境中的 .env 文件（如果存在）
load_dotenv()


class HelloAgentsLLM:
    """第四章使用的基础 LLM 客户端。"""

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
    ):
        # 调用者传入的参数优先；没有传入时再读取环境变量。
        self.model = model or os.getenv("LLM_MODEL_ID", "gpt-5.6-sol")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv(
            "LLM_BASE_URL",
            "https://oneapi-comate.baidu-int.com/v1",
        )
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError(
                "模型 ID、API Key 和服务地址必须被提供或在环境变量中定义。"
            )

        self.client = OpenAI(
            api_key=apiKey,
            base_url=baseUrl,
            timeout=timeout,
        )

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
    ) -> str:
        """调用模型，并返回完整的文本回复。"""
        print(f"🧠 正在调用 {self.model} 模型...")
        for message in messages:
            print(message)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("✅ 大语言模型响应成功:")
            collected_content = []

            for chunk in response:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)

            print()
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用 LLM API 时发生错误：{e}")
            return None


if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好，请用一句话介绍自己。"},
        ]

        print("--- 调用 LLM ---")
        responseText = llmClient.think(exampleMessages)

        if responseText:
            print("\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
