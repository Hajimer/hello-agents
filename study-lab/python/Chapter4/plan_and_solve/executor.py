from plan_and_solve.prompt import EXECUTOR_PROMPT_TEMPLATE

class Executor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        Execute step by step according to the plan and solve the problem.
        """
        history = ""  # String to store historical steps and results

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )

            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.think(messages=messages) or ""

            # Update history for the next step
            history += f"步骤 {i+1}: {step}\n结果：{response_text}\n\n"

            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        # After the loop ends, the last step's response is the final answer
        final_answer = response_text
        return final_answer