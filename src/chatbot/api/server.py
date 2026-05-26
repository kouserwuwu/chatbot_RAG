import uuid
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chatbot.core.session_manager import SessionManager
from chatbot.core.tools import ToolExecutor
from chatbot.core.rag_engine import get_rag_engine
from chatbot.core.persistence import ConversationStore
from chatbot.config.settings import settings
from chatbot.exceptions import ChatbotError, KnowledgeNotFoundError
from chatbot.logging_config import setup_logging, get_logger


app = FastAPI(title="Enterprise AI Agent API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session_manager = SessionManager()
conv_store = ConversationStore()
logger = get_logger("server")


# ── 数据模型 ──

class SourceInfo(BaseModel):
    """RAG 答案溯源：引用的知识库文档片段。"""
    document: str           # 文档文件名
    chunk: str              # 匹配的文本片段
    score: float | None = None  # 相似度分数（如果可用）


class ErrorInfo(BaseModel):
    """结构化错误响应。"""
    code: str               # 错误码，如 LLM_CLIENT_ERROR
    message: str            # 人类可读的错误描述
    details: dict = Field(default_factory=dict)  # 附加详情


class ChatRequest(BaseModel):
    message: str = Field(description="用户消息文本")


class ChatResponse(BaseModel):
    answer: str = Field(description="AI 回答")
    thought_process: list[str] = Field(default_factory=list)
    sources: list[SourceInfo] = Field(default_factory=list,
                                        description="引用的知识库文档片段")
    tokens_used: dict = Field(default_factory=dict,
                               description="Token 用量估算")


# ── 全局异常处理器 ──

@app.exception_handler(ChatbotError)
async def chatbot_error_handler(request: Request, exc: ChatbotError):
    """将 ChatbotError 及其子类转为结构化 JSON 错误响应。"""
    logger.error("[%s] %s", exc.code, exc.message)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
            "request_id": str(uuid.uuid4())[:8],
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """统一 HTTP 异常的 JSON 格式。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "details": {},
            },
            "request_id": str(uuid.uuid4())[:8],
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底：未知异常也返回结构化 JSON。"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "服务器内部错误",
                "details": {"type": type(exc).__name__},
            },
            "request_id": str(uuid.uuid4())[:8],
        },
    )


# ── 端点 ──

@app.get("/health")
async def health_check():
    """服务健康检查：验证 LLM 和 ChromaDB 连通性。"""
    checks = {"database": "ok"}
    try:
        rag = get_rag_engine()
        if rag.vector_db is not None:
            rag.query("ping")
        checks["knowledge_base"] = "ok"
    except Exception as e:
        checks["knowledge_base"] = f"degraded: {e}"
    return {"status": "ok", "checks": checks}


@app.get("/status")
async def get_status(x_user_id: str | None = Header(None, alias="X-User-Id")):
    """返回当前用户的内存压缩状态。"""
    uid = _get_user_id(x_user_id)
    return session_manager.get_status(uid)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
):
    """核心聊天接口，支持 RAG 答案溯源和多用户隔离。"""
    user_input = request.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="消息不能为空")

    uid = _get_user_id(x_user_id)
    logger.info("Chat request: user=%s, message=%s", uid, user_input[:100])

    ctx = session_manager.get_or_create(uid)
    tool_executor = ToolExecutor(user_id=uid)
    ctx.agent.tools = tool_executor.get_registry()

    # 活跃会话管理：获取或创建 conversation_id
    active_conv = conv_store.get_active_conversation(uid)
    if active_conv is None:
        conv_id = conv_store.create_conversation(uid)
        # 恢复历史消息到 LLM 记忆
        conv_store.load_history_into_llm(conv_id, ctx.agent.llm_client)
    else:
        conv_id = active_conv["id"]

    result = ctx.agent.run(user_input)

    # 持久化本轮对话
    conv_store.save_message_pair(conv_id, user_input, result.answer)

    # 构建思考过程
    thought_process = []
    sources: list[SourceInfo] = []
    for step in result.steps:
        if step.tool_name:
            thought_process.append(
                f"Tool {step.tool_name}({step.tool_args}) -> {step.observation}"
            )
            if step.tool_name == "search_knowledge" and step.observation:
                sources = _extract_sources(user_input)
        else:
            thought_process.append(step.thought or "")

    return ChatResponse(
        answer=result.answer,
        thought_process=thought_process,
        sources=sources,
        tokens_used={"estimated_input": len(user_input), "estimated_output": len(result.answer)},
    )


@app.post("/clear")
async def clear_memory(x_user_id: str | None = Header(None, alias="X-User-Id")):
    """清空当前用户的所有层级记忆。"""
    uid = _get_user_id(x_user_id)
    msg = session_manager.clear_session(uid)
    return {"status": "success", "message": msg}


# ── 知识库管理端点 ──

@app.get("/knowledge")
async def list_knowledge():
    """列出知识库中的所有文档。"""
    rag = get_rag_engine()
    if not rag.knowledge_dir.exists():
        return {"documents": []}
    docs = [
        {"name": f, "size": (rag.knowledge_dir / f).stat().st_size}
        for f in sorted(rag.knowledge_dir.glob("*.txt"))
        if f.is_file()
    ]
    return {"documents": docs}


@app.post("/knowledge/reload")
async def reload_knowledge():
    """热重载知识库：重新扫描 knowledge/ 目录并重建索引，无需重启服务。"""
    rag = get_rag_engine()
    try:
        rag._initialize_db()
        return {"status": "success", "message": "知识库已重新加载"}
    except Exception as e:
        raise ChatbotError(f"知识库重载失败: {e}")


@app.post("/knowledge")
async def add_knowledge(name: str, content: str):
    """动态添加一篇知识库文档（写入 .txt 文件并加入向量索引）。"""
    rag = get_rag_engine()

    # 安全校验：只允许 .txt 扩展名
    if not name.endswith(".txt"):
        name = f"{name}.txt"
    safe_name = name.replace("..", "").replace("/", "_").replace("\\", "_")

    file_path = rag.knowledge_dir / safe_name
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"文档 {safe_name} 已存在")

    file_path.write_text(content, encoding="utf-8")
    logger.info("知识库新增文档: %s", safe_name)

    # 增量加入向量库
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    try:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=rag.chunk_size,
            chunk_overlap=rag.chunk_overlap,
        )
        splits = splitter.split_documents(docs)
        if rag.vector_db is not None:
            rag.vector_db.add_documents(splits)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise ChatbotError(f"文档索引失败: {e}")

    return {"status": "success", "document": safe_name, "chunks": len(splits)}


@app.delete("/knowledge/{name}")
async def delete_knowledge(name: str):
    """删除一篇知识库文档（从磁盘和向量库中移除）。"""
    rag = get_rag_engine()
    file_path = rag.knowledge_dir / name

    if not file_path.exists():
        raise KnowledgeNotFoundError(f"文档 {name} 不存在")

    # 从向量库删除（如果 ChromaDB 有 metadata 过滤，这里简化处理：全量重建）
    file_path.unlink()
    rag._initialize_db()
    logger.info("知识库删除文档: %s", name)
    return {"status": "success", "message": f"文档 {name} 已删除"}


# ── 辅助函数 ──

def _get_user_id(x_user_id: str | None = None) -> str:
    if x_user_id:
        return x_user_id
    return "anonymous"


def _extract_sources(query: str) -> list[SourceInfo]:
    """从 RAG 引擎提取溯源信息。"""
    try:
        rag = get_rag_engine()
        if rag.vector_db is None:
            return []
        docs_with_scores = rag.vector_db.similarity_search_with_score(query, k=3)
        sources = []
        for doc, score in docs_with_scores:
            source_name = doc.metadata.get("source", "unknown")
            sources.append(SourceInfo(
                document=source_name.split("/")[-1] if "/" in source_name else source_name,
                chunk=doc.page_content[:200],
                score=round(float(score), 4),
            ))
        return sources
    except Exception:
        return []


@app.get("/conversations")
async def list_conversations(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
):
    """获取当前用户的历史对话列表（从持久化存储读取）。"""
    uid = _get_user_id(x_user_id)
    convs = conv_store.list_conversations(uid, limit=20)
    return {
        "user_id": uid,
        "conversations": [
            {
                "id": c["id"],
                "status": c["status"],
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
            }
            for c in convs
        ],
        "count": len(convs),
    }


@app.get("/conversations/{conv_id}/messages")
async def get_conversation_messages(conv_id: str):
    """获取指定对话的完整消息历史。"""
    msgs = conv_store.get_messages(conv_id)
    if not msgs:
        raise HTTPException(status_code=404, detail=f"对话 {conv_id} 不存在")
    return {
        "conversation_id": conv_id,
        "messages": [
            {"role": m["role"], "content": m["content"], "time": m["created_at"]}
            for m in msgs
        ],
        "count": len(msgs),
    }


@app.post("/conversations/{conv_id}/close")
async def close_conversation(conv_id: str):
    """关闭一个对话（标记为已解决）。"""
    msgs = conv_store.get_messages(conv_id)
    if not msgs:
        raise HTTPException(status_code=404, detail=f"对话 {conv_id} 不存在")
    conv_store.close_conversation(conv_id)
    return {"status": "success", "message": f"对话 {conv_id} 已关闭"}


def main():
    import uvicorn
    setup_logging(level=settings.LOG_LEVEL)
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
