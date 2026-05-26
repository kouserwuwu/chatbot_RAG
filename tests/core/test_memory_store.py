import pytest
from chatbot.core.memory_store import VectorMemoryStore
from chatbot.exceptions import MemoryStoreError


def test_save_and_retrieve(temp_chroma_dir):
    """存一个记忆，检索出来。"""
    store = VectorMemoryStore(persist_directory=temp_chroma_dir / "chroma")
    store.save("用户喜欢爬山")
    result = store.recall("户外活动")
    assert "爬山" in result


def test_save_empty_content(temp_chroma_dir):
    """空内容应抛出异常。"""
    store = VectorMemoryStore(persist_directory=temp_chroma_dir / "chroma")
    with pytest.raises(MemoryStoreError):
        store.save("")


def test_save_whitespace_only(temp_chroma_dir):
    """纯空白也应抛出异常。"""
    store = VectorMemoryStore(persist_directory=temp_chroma_dir / "chroma")
    with pytest.raises(MemoryStoreError):
        store.save("   ")


def test_list_all_empty(temp_chroma_dir):
    """新存储 list_all 返回提示。"""
    store = VectorMemoryStore(persist_directory=temp_chroma_dir / "chroma")
    result = store.list_all()
    assert "没有记录" in result


def test_list_all_after_save(temp_chroma_dir):
    """存储后有内容时列出。"""
    store = VectorMemoryStore(persist_directory=temp_chroma_dir / "chroma")
    store.save("测试记忆1")
    result = store.list_all()
    assert "测试记忆1" in result


def test_recall_empty_query(temp_chroma_dir):
    """空查询返回提示。"""
    store = VectorMemoryStore(persist_directory=temp_chroma_dir / "chroma")
    result = store.recall("")
    assert result == "查询词为空"


def test_count(temp_chroma_dir):
    """count 属性返回记忆条数。"""
    store = VectorMemoryStore(persist_directory=temp_chroma_dir / "chroma")
    assert store.count == 0
    store.save("记忆A")
    assert store.count == 1
