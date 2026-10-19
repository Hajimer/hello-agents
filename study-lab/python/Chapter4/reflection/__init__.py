from llm_client import HelloAgentsLLM
from reflection.reflection_agent import ReflectionAgent

if __name__ == '__main__':
    try:
        llm_client = HelloAgentsLLM()
        agent = ReflectionAgent(llm_client)
        task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
        agent.run(task=task)
    except ValueError as e:
        print(e)