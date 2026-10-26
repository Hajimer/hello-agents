from typing import Optional
import ast
import re
from hello_agents import Config, HelloAgentsLLM, PlanAndSolveAgent, Message
from prompt import DEFAULT_PLANNER_PROMPT, DEFAULT_EXECUTOR_PROMPT


class MyPlanAndSolveAgent(PlanAndSolveAgent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[dict[str, str]] = None,
    ):
        """初始化父类状态，选择默认或自定义 planner / executor 提示词。"""
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config
        )
        prompts = custom_prompts or {}
        self.planner_prompt = prompts.get(
            "planner",
            DEFAULT_PLANNER_PROMPT
        )
        self.executor_prompt = prompts.get(
            "executor",
            DEFAULT_EXECUTOR_PROMPT
        )

    def run(self, input_text: str, **kwargs) -> str:
        """先规划，再逐步执行，最后保存输入和最终结果。"""
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        plan = self._plan(input_text, **kwargs)

        if not plan:
            final_answer = "无法生成有效计划，任务已终止。"
        else:
            final_answer = self._execute(input_text, plan, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    def _plan(self, question: str, **kwargs) -> list[str]:
        """对应第四章 Planner.plan：构造提示词、请求模型并解析计划。"""
        prompt = self.planner_prompt.format(question=question)

        print("\n--- 正在生成计划 ---")
        response_text = self._get_llm_response(prompt, **kwargs)
        print(f"模型返回的计划:\n{response_text}")

        return self._parse_plan(response_text)

    def _parse_plan(self, response_text: str) -> list[str]:
        """解析模型输出，校验是否为非空步骤字符串列表，处理无效输出。"""
        plan_text = response_text.strip()

        # 默认 Prompt 要求的格式：```python ... ```
        match = re.search(
            r"```(?:python)?\s*(.*?)```",
            plan_text,
            re.DOTALL,
        )
        if match:
            plan_text = match.group(1).strip()
        elif plan_text.startswith("```python\n"):
            # 兼容当前数学 Prompt 示例中的 python 换行前缀。
            plan_text = plan_text[len("python\n"):].strip()

        try:
            plan = ast.literal_eval(plan_text)
        except (ValueError, SyntaxError) as e:
            print(f"❌ 计划解析失败: {e}")
            return []

        if not isinstance(plan, list) or not plan:
            print("❌ 计划必须是非空列表。")
            return []

        if not all(
            isinstance(step, str) and step.strip()
            for step in plan
        ):
            print("❌ 每个计划步骤必须是非空字符串。")
            return []

        return [step.strip() for step in plan]

    def _execute(self, question: str, plan: list[str], **kwargs) -> str:
        """对应第四章 Executor.execute：依次执行计划，将前面的结果传递给后面的步骤。"""
        history = []
        final_answer = ""

        for index, step in enumerate(plan, start=1):
            print(f"\n--- 执行步骤 {index}/{len(plan)}: {step} ---")

            prompt = self.executor_prompt.format(
                question=question,
                plan=plan,
                history="\n\n".join(history) if history else "无",
                current_step=step,
            )

            result = self._get_llm_response(prompt, **kwargs)

            if not result.strip():
                raise ValueError(
                    f"第 {index} 步未获得有效结果，停止执行。"
                )

            history.append(
                f"步骤 {index}: {step}\n结果: {result}"
            )
            final_answer = result
            print(f"✅ 本步结果: {result}")

        return final_answer
    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """组织消息并通过统一客户端获取完整回复。"""
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        return self.llm.invoke(messages, **kwargs) or ""
