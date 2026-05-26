"""用户上下文数据类，封装一个用户的所有运行时状态。"""

from dataclasses import dataclass, field

from chatbot.core.llm_client import LLMClient


@dataclass
class UserContext:
    """一个用户的完整会话上下文。"""

    user_id: str
    agent: object | None = None
    llm_client: LLMClient | None = None
    memory_store: object | None = None

    def cleanup(self):
        """清理该用户的资源。"""
        if self.llm_client is not None:
            self.llm_client.clear_memory()
        self.agent = None
        self.llm_client = None
        self.memory_store = None
