import pytest
from chatbot.core.tools import calculate, get_current_time, search_knowledge
from chatbot.core.tools import ToolExecutor, TOOL_REGISTRY
from chatbot.exceptions import ToolExecutionError


class TestPureFunctions:
    """纯函数工具测试。"""

    def test_calculate_simple(self):
        result = calculate("2**10")
        assert "1024" in result

    def test_calculate_math_sqrt(self):
        result = calculate("math.sqrt(16)")
        assert "4" in result or "4.0" in result

    def test_calculate_invalid(self):
        with pytest.raises(ToolExecutionError):
            calculate("invalid$$$$")

    def test_get_current_time(self):
        result = get_current_time()
        assert "当前时间是:" in result
        assert ":" in result

    def test_tool_registry_has_core_tools(self):
        assert "calculate" in TOOL_REGISTRY
        assert "get_current_time" in TOOL_REGISTRY
        assert "search_knowledge" in TOOL_REGISTRY


class TestToolExecutor:
    """ToolExecutor 用户隔离测试。"""

    def test_executor_registry_has_all_tools(self):
        executor = ToolExecutor(user_id="test_user")
        registry = executor.get_registry()
        assert "calculate" in registry
        assert "get_current_time" in registry
        assert "search_knowledge" in registry
        assert "save_memory" in registry
        assert "recall_memory" in registry
        assert "list_all_memories" in registry

    def test_executor_save_and_recall(self, temp_chroma_dir):
        """用户隔离：保存和召回在同一用户的 executor 中生效。"""
        # 注意：ToolExecutor 内部创建 VectorMemoryStore(user_id=...)
        # 需要覆盖 persist_directory
        from chatbot.core.memory_store import VectorMemoryStore

        original_init = VectorMemoryStore.__init__

        def patched_init(self, user_id=None, persist_directory=None, embedding_model=None):
            original_init(
                self,
                user_id=user_id,
                persist_directory=temp_chroma_dir / "chroma",
                embedding_model=embedding_model,
            )

        import chatbot.core.tools as tools_module
        import chatbot.core.memory_store as memory_module
        old_init = memory_module.VectorMemoryStore.__init__
        memory_module.VectorMemoryStore.__init__ = patched_init

        try:
            executor = ToolExecutor(user_id="alice")
            registry = executor.get_registry()
            registry["save_memory"]("Alice 喜欢爬山")
            result = registry["recall_memory"]("户外活动")
            assert "爬山" in result
        finally:
            memory_module.VectorMemoryStore.__init__ = old_init

    def test_user_isolation(self, temp_chroma_dir):
        """两个用户的记忆互不干扰。"""
        from chatbot.core.memory_store import VectorMemoryStore

        original_init = VectorMemoryStore.__init__

        def make_patched(user_id):
            def patched_init(self, user_id=user_id, persist_directory=None, embedding_model=None):
                original_init(
                    self,
                    user_id=user_id,
                    persist_directory=temp_chroma_dir / "chroma",
                    embedding_model=embedding_model,
                )
            return patched_init

        import chatbot.core.memory_store as memory_module
        old_init = memory_module.VectorMemoryStore.__init__

        try:
            # Alice 存记忆
            memory_module.VectorMemoryStore.__init__ = make_patched("alice")
            executor_a = ToolExecutor(user_id="alice")
            reg_a = executor_a.get_registry()
            reg_a["save_memory"]("Alice 的秘密")

            # Bob 召回
            memory_module.VectorMemoryStore.__init__ = make_patched("bob")
            executor_b = ToolExecutor(user_id="bob")
            reg_b = executor_b.get_registry()
            result = reg_b["recall_memory"]("秘密")
            # Bob 不应该看到 Alice 的记忆
            assert "Alice" not in result or "没有找到" in result
        finally:
            memory_module.VectorMemoryStore.__init__ = old_init
