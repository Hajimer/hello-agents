from plan_and_solve.executor import Executor
from plan_and_solve.planner import Planner


class PlanAndSolveAgent:
    def __init__(self, llm_client):
        """
        Initialize the agent and create planner and executor instances.
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question:str):
        """
        Run the agent's complete process: plan first, then execute.
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1. Call planner to generate plan
        plan = self.planner.plan(question)

        # Check if plan was successfully generated
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 2. Call executor to execute plan
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")