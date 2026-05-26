import gc
import pytest
from pathlib import Path
import tempfile


class FakeLLMClient:
    """返回预制响应，不调用 Ollama。"""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["这是一条不需要工具的回复。"]
        self.call_count = 0
        self.history: list[dict] = []
        self.l1_summaries: list[str] = []
        self.l2_summary: str = ""
        self.max_history = 10
        self.max_l1 = 10
        self.received_messages: list[list[dict]] = []

    def get_response(self, user_input: str) -> str:
        idx = min(self.call_count, len(self.responses) - 1)
        response = self.responses[idx]
        self.call_count += 1
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        return response

    def clear_memory(self):
        self.history = []
        self.l1_summaries = []
        self.l2_summary = ""


class _ChromaCleanup:
    """跟踪 ChromaDB 实例，在测试结束后清理。"""
    _instances: list = []

    @classmethod
    def register(cls, instance):
        cls._instances.append(instance)

    @classmethod
    def cleanup(cls):
        for inst in cls._instances:
            try:
                if hasattr(inst, 'vector_db') and inst.vector_db is not None:
                    if hasattr(inst.vector_db, '_client'):
                        del inst.vector_db
            except Exception:
                pass
        cls._instances.clear()
        gc.collect()


@pytest.fixture(autouse=True)
def chroma_cleanup():
    """每个测试后清理 ChromaDB 连接。"""
    yield
    _ChromaCleanup.cleanup()


@pytest.fixture
def fake_llm():
    return FakeLLMClient()


@pytest.fixture
def temp_knowledge_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("这是测试文档内容。小明喜欢编程和爬山。", encoding="utf-8")
        yield Path(tmpdir)


@pytest.fixture
def temp_chroma_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)
