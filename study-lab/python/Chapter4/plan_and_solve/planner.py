import ast

from plan_and_solve.prompt import PLANNER_PROMPT_TEMPLATE

class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question:str) -> list[str]:
        """
        Generate an action plan based on user question.
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        # To generate a plan, we build a simple message list
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        # Use streaming output to get the complete plan
        response_text = self.llm_client.think(messages=messages) or ""

        print(f"✅ 计划已生成:\n{response_text}")

        # Parse the list string output by LLM
        try:
            # 按 Prompt 约定提取 ```python 与 ``` 之间的列表文本，并去掉两端空白。
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 将列表字符串解析为 Python 对象，不执行任意 Python 代码。
            plan = ast.literal_eval(plan_str)
            # 确认结果是列表。
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []
