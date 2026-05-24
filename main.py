from core.llm_client import llm_client
from core.tools import TOOL_MAP

import re
import ast

def main():
    print("========================================")
    print("🤖 欢迎来到 AI Chatbot v6.0 (长期记忆版)")
    print("输入 'quit' 或 'exit' 可以退出对话")
    print("========================================\n")

    while True:
        # 1. 获取用户输入
        user_input = input("👤 你: ").strip()

        # 2. 检查是否需要退出
        if user_input.lower() in ['quit', 'exit']:
            print("\n👋 再见！祝你 Agent 学习之旅愉快！")
            break

        if not user_input:
            continue

        # 3. 执行循环 (Thought -> Action -> Observation -> Final Response)
        current_input = user_input
        max_iterations = 3  # 防止 AI 陷入死循环
        iteration = 0

        while iteration < max_iterations:
            print("🤖 AI 正在思考...", end="\r")
            print(f"--- 内存快照 (当前历史记录条数): {len(llm_client.history)} ---")
            response = llm_client.get_response(current_input)

            # 检查 AI 是否请求调用工具 (匹配 Action: tool_name(args))
            match = re.search(r"Action:\s*(\w+)\((.*)\)", response)

            if match:
                tool_name = match.group(1)
                args_str = match.group(2).strip()

                if tool_name in TOOL_MAP:
                    print(f"🛠️  AI 决定调用工具: {tool_name}({args_str})", end="\r")
                    # 执行工具
                    tool_func = TOOL_MAP[tool_name]

                    try:
                        if args_str:
                            # 使用 ast.literal_eval 将字符串形式的参数转化为 Python 对象（元组/列表）
                            # 比如 " '姓名', '小明' " -> ('姓名', '小明')
                            parsed_args = ast.literal_eval(f"({args_str})")
                            if isinstance(parsed_args, tuple):
                                observation = tool_func(*parsed_args)
                            else:
                                observation = tool_func(parsed_args)
                        else:
                            observation = tool_func()
                    except Exception as e:
                        observation = f"工具调用参数解析失败: {str(e)}"

                    # 将工具结果喂回给 AI (作为新的输入)
                    current_input = f"【工具执行结果】：\n{observation}\n\n请基于以上结果给出最终回答。"
                    iteration += 1
                    continue # 再次进入 LLM 循环，生成最终回答
                else:
                    # 如果工具不存在，告知 AI 错误
                    current_input = f"错误：工具 {tool_name} 不存在。请尝试使用可用工具或直接回答。"
                    iteration += 1
                    continue
            else:
                # 没有调用工具，直接返回最终结果
                print(f"🤖 AI: {response}\n")
                break

        if iteration >= max_iterations:
            print(f"🤖 AI: (思考次数过多，无法给出结论) {response}\n")

if __name__ == "__main__":
    main()
