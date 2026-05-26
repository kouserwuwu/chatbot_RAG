import datetime
import math

from chatbot.core.rag_engine import get_rag_engine
from chatbot.core.memory_store import VectorMemoryStore
from chatbot.exceptions import ToolExecutionError
from chatbot.logging_config import get_logger

logger = get_logger("tools")

# ── 工具函数（纯函数，无用户状态） ──


def calculate(expression: str) -> str:
    """
    执行安全数学计算。
    输入应为有效的 Python 数学表达式，例如 '2**10' 或 'math.sqrt(16)'。
    """
    try:
        result = eval(expression, {"__builtins__": None}, {"math": math})
        return f"计算结果: {result}"
    except Exception as e:
        raise ToolExecutionError(f"计算出错: {e}")


def get_current_time() -> str:
    """获取当前系统时间。"""
    now = datetime.datetime.now()
    return f"当前时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def search_knowledge(query: str) -> str:
    """检索本地知识库（全局共享，不需要用户隔离）。"""
    logger.info("RAG 检索: %s", query)
    result = get_rag_engine().query(query)
    return result if result else "知识库中没有找到相关信息。"


# ── 工具注册表（纯函数，不绑定用户状态） ──

TOOL_REGISTRY: dict[str, callable] = {}


def register_tool(name: str):
    """装饰器：将函数注册到 TOOL_REGISTRY。"""
    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func
    return decorator


# 批量注册纯函数工具
for _name, _func in [
    ("calculate", calculate),
    ("get_current_time", get_current_time),
    ("search_knowledge", search_knowledge),
]:
    register_tool(_name)(_func)


# ── 工具执行器（绑定用户状态） ──

class ToolExecutor:
    """
    工具执行器，绑定用户特定的 MemoryStore。

    每个用户会话应创建一个独立的 ToolExecutor 实例，
    确保 save_memory / recall_memory / list_all_memories 操作
    隔离到用户专属的 ChromaDB collection。

    用法:
        executor = ToolExecutor(user_id="alice")
        result = executor.call("save_memory", "我喜欢爬山")
    """

    def __init__(self, user_id: str | None = None):
        self.user_id = user_id
        self._memory_store: VectorMemoryStore | None = None

    def _get_memory_store(self) -> VectorMemoryStore:
        """延迟创建用户隔离的 MemoryStore。"""
        if self._memory_store is None:
            self._memory_store = VectorMemoryStore(user_id=self.user_id)
        return self._memory_store

    def save_memory(self, content: str) -> str:
        """保存一条长期记忆到用户的专属 collection。"""
        logger.info("存储语义记忆 [%s]: %s", self.user_id or "global", content)
        return self._get_memory_store().save(content)

    def recall_memory(self, query: str) -> str:
        """从用户的专属 collection 检索记忆。"""
        logger.info("回忆语义记忆 [%s]: %s", self.user_id or "global", query)
        return self._get_memory_store().recall(query)

    def list_all_memories(self) -> str:
        """列出用户的所有记忆。"""
        logger.info("列出所有记忆 [%s]", self.user_id or "global")
        return self._get_memory_store().list_all()

    def get_registry(self) -> dict[str, callable]:
        """
        返回完整的工具注册表（包含纯函数 + 用户绑定的方法）。

        这个返回值可以直接传入 Agent(tool_registry=...)。
        """
        return {
            "calculate": calculate,
            "get_current_time": get_current_time,
            "search_knowledge": search_knowledge,
            "save_memory": self.save_memory,
            "recall_memory": self.recall_memory,
            "list_all_memories": self.list_all_memories,
        }


# ── 向后兼容 ──
TOOL_MAP = TOOL_REGISTRY
