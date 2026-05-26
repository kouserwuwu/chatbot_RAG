"""会话管理器，管理多用户会话生命周期，线程安全。"""

import time
import threading

from chatbot.core.agent import Agent
from chatbot.core.llm_client import LLMClient
from chatbot.core.user_context import UserContext
from chatbot.config.settings import settings
from chatbot.logging_config import get_logger

logger = get_logger("session_manager")


class SessionManager:
    """
    管理用户会话的创建、获取和清理。

    每个用户拥有独立的 Agent + LLMClient，确保：
    - 对话历史互不污染
    - 分级记忆（L0/L1/L2）完全隔离

    用法:
        manager = SessionManager()
        ctx = manager.get_or_create("user_123")
        result = ctx.agent.run("你好")
    """

    def __init__(
        self,
        session_ttl: float | None = None,
    ):
        """
        Args:
            session_ttl: 会话过期时间（秒）。None 表示永不过期。
        """
        self._sessions: dict[str, UserContext] = {}
        self._lock = threading.Lock()
        self._session_ttl = session_ttl
        self._last_access: dict[str, float] = {}

    def get_or_create(self, user_id: str) -> UserContext:
        """获取或创建用户会话。"""
        # 快速路径：无锁读取
        if user_id in self._sessions:
            self._last_access[user_id] = time.time()
            return self._sessions[user_id]

        # 慢速路径：加锁创建
        with self._lock:
            # 双重检查（可能另一个线程已经创建了）
            if user_id in self._sessions:
                self._last_access[user_id] = time.time()
                return self._sessions[user_id]

            logger.info("创建新会话: user_id=%s", user_id)

            llm_client = LLMClient()
            agent = Agent(llm_client=llm_client, max_iterations=settings.MAX_ITERATIONS)

            ctx = UserContext(
                user_id=user_id,
                agent=agent,
                llm_client=llm_client,
            )
            self._sessions[user_id] = ctx
            self._last_access[user_id] = time.time()

        return ctx

    def clear_session(self, user_id: str) -> str:
        """清空并移除指定用户的会话。"""
        with self._lock:
            ctx = self._sessions.pop(user_id, None)
            self._last_access.pop(user_id, None)

        if ctx is not None:
            ctx.cleanup()
            logger.info("会话已清除: user_id=%s", user_id)
            return f"用户 {user_id} 的所有记忆已清空"
        return f"用户 {user_id} 没有活动会话"

    def get_status(self, user_id: str) -> dict:
        """获取指定用户的内存状态。"""
        ctx = self._sessions.get(user_id)
        if ctx is None or ctx.llm_client is None:
            return {
                "l0_count": 0,
                "l1_count": 0,
                "l2_summary": "",
                "max_l0": settings.MAX_HISTORY,
                "max_l1": settings.MAX_L1_SUMMARIES,
            }

        llm = ctx.llm_client
        return {
            "l0_count": len(llm.history),
            "l1_count": len(llm.l1_summaries),
            "l2_summary": llm.l2_summary,
            "max_l0": llm.max_history,
            "max_l1": llm.max_l1,
        }

    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量。"""
        if self._session_ttl is None:
            return 0

        now = time.time()
        expired: list[str] = []
        with self._lock:
            for user_id, last in list(self._last_access.items()):
                if now - last > self._session_ttl:
                    expired.append(user_id)

        count = 0
        for user_id in expired:
            logger.info("清理过期会话: user_id=%s", user_id)
            self.clear_session(user_id)
            count += 1

        return count

    @property
    def active_count(self) -> int:
        """当前活跃会话数量。"""
        return len(self._sessions)


# 全局单例（服务端共享）
session_manager = SessionManager()
