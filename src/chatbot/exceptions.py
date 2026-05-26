class ChatbotError(Exception):
    """所有聊天机器人异常的基类。"""

    code: str = "UNKNOWN_ERROR"

    def __init__(self, message: str = "", details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(ChatbotError):
    """配置无效时抛出。"""
    code = "CONFIGURATION_ERROR"


class LLMClientError(ChatbotError):
    """LLM API 调用失败时抛出。"""
    code = "LLM_CLIENT_ERROR"


class RAGEngineError(ChatbotError):
    """RAG 引擎初始化或查询失败时抛出。"""
    code = "RAG_ENGINE_ERROR"


class MemoryStoreError(ChatbotError):
    """向量记忆存储读写失败时抛出。"""
    code = "MEMORY_STORE_ERROR"


class ToolExecutionError(ChatbotError):
    """工具函数运行时出错时抛出。"""
    code = "TOOL_EXECUTION_ERROR"


class AgentLoopError(ChatbotError):
    """ReAct 循环达到最大迭代次数或遇到不可恢复状态时抛出。"""
    code = "AGENT_LOOP_ERROR"


class KnowledgeNotFoundError(ChatbotError):
    """知识库文档未找到时抛出。"""
    code = "KNOWLEDGE_NOT_FOUND"
