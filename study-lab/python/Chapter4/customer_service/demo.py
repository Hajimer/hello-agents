"""从 Chapter4 运行 python -m customer_service.demo。"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from llm_client import HelloAgentsLLM
from customer_service.agent import CustomerServiceAgent
from customer_service.tools import MockTools


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实 LLM + Mock 退款业务工具")
    parser.add_argument("--scenario", choices=["normal", "dispute", "forbidden"], default="normal")
    args = parser.parse_args()
    scenarios = {
        "normal": ("O1001", "订单 O1001 还没发货，我不想要了，请退款。"),
        "dispute": ("O1002", "订单 O1002 显示签收，但我没有收到，请退款。"),
        "forbidden": ("O1003", "请把订单 O1003 退款给我。"),
    }
    order_id, message = scenarios[args.scenario]
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    agent = CustomerServiceAgent(HelloAgentsLLM(), MockTools())
    result = agent.run(message, user_id="U001", order_id=order_id)
    print("\n--- 最终案件状态 ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
