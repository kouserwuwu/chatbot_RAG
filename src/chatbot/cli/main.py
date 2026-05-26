from chatbot.core.agent import Agent
from chatbot.core.llm_client import create_default_client
from chatbot.core.tools import ToolExecutor
from chatbot.logging_config import setup_logging, get_logger
from chatbot.config.settings import settings


def main():
    setup_logging(level=settings.LOG_LEVEL)
    logger = get_logger("cli")

    # CLI 单用户场景，使用固定 user_id
    user_id = "cli_user"
    tool_executor = ToolExecutor(user_id=user_id)

    agent = Agent(
        llm_client=create_default_client(),
        tool_registry=tool_executor.get_registry(),
        max_iterations=settings.MAX_ITERATIONS,
    )

    print("=" * 40)
    print("AI Chatbot v2.0 (Engineered)")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        if not user_input:
            continue

        logger.debug("User input: %s", user_input)
        print("AI 正在思考...")

        result = agent.run(user_input)

        for step in result.steps:
            if step.tool_name:
                print(f"  [工具] {step.tool_name}({step.tool_args}) -> {step.observation[:80]}...")

        print(f"\nAI: {result.answer}")


if __name__ == "__main__":
    main()
