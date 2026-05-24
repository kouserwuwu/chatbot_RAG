from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import ast

from core.llm_client import llm_client
from core.tools import TOOL_MAP

app = FastAPI(title="AI Agent v7.0 API Server")

# 允许跨域访问，方便前端调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    thought_process: list[str]  # 记录思考过程，方便前端展示

@app.get("/status")
async def get_status():
    """
    返回当前的内存压缩状态
    """
    return {
        "l0_count": len(llm_client.history),
        "l1_count": len(llm_client.l1_summaries),
        "l2_summary": llm_client.l2_summary,
        "max_l0": llm_client.max_history,
        "max_l1": llm_client.max_l1
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    核心聊天接口，实现了 v7.0 的 ReAct 循环
    """
    user_input = request.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    current_input = user_input
    thought_process = []
    max_iterations = 3
    iteration = 0

    while iteration < max_iterations:
        # 1. 获取 LLM 响应 (内部已包含分级记忆管理)
        response = llm_client.get_response(current_input)
        thought_process.append(response)

        # 2. 检查是否请求调用工具 (Action: tool_name(args))
        match = re.search(r"Action:\s*(\w+)\((.*)\)", response)

        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            if tool_name in TOOL_MAP:
                tool_func = TOOL_MAP[tool_name]
                try:
                    if args_str:
                        # 保持 v7.0 的 ast.literal_eval 解析逻辑
                        parsed_args = ast.literal_eval(f"({args_str})")
                        if isinstance(parsed_args, tuple):
                            observation = tool_func(*parsed_args)
                        else:
                            observation = tool_func(parsed_args)
                    else:
                        observation = tool_func()
                except Exception as e:
                    observation = f"工具调用参数解析失败: {str(e)}"

                # 将工具结果喂回 LLM
                current_input = f"【工具执行结果】：\n{observation}\n\n请基于以上结果给出最终回答。"
                thought_process.append(f"🛠️ 调用工具 {tool_name} -> 结果: {observation}")
                iteration += 1
                continue
            else:
                current_input = f"错误：工具 {tool_name} 不存在。请尝试使用可用工具或直接回答。"
                thought_process.append(f"❌ 工具 {tool_name} 未找到")
                iteration += 1
                continue
        else:
            # 没有工具调用，返回最终答案
            return ChatResponse(answer=response, thought_process=thought_process)

    return ChatResponse(
        answer=f"(思考次数过多，无法给出结论) {response}",
        thought_process=thought_process
    )

@app.post("/clear")
async def clear_memory():
    """
    清空所有层级记忆
    """
    result = llm_client.clear_memory()
    return {"status": "success", "message": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
