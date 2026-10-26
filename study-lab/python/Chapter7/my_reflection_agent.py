from hello_agents import ReflectionAgent, HelloAgentsLLM, Config, Message, ToolRegistry
from typing import Optional, List, Tuple


class MyReflectionAgent(ReflectionAgent):
    """基于框架 ReflectionAgent 的可配置反思智能体。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[dict] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            max_iterations=max_iterations,
            custom_prompts=custom_prompts,
        )

    def run(self, input_text: str, **kwargs) -> str:
        """执行初始回答、反思和修订循环，返回最后一次结果。"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 每次 run 都使用新的短期反思轨迹。
        self.memory = type(self.memory)()

        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        for iteration in range(self.max_iterations):
            print(f"\n--- 第 {iteration + 1}/{self.max_iterations} 轮迭代 ---")

            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result,
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                break

            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback,
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution() or ""
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))
        return final_result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """将单个提示词转换为框架统一的消息并调用 LLM。"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
