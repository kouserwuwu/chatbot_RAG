import pytest
from chatbot.core.rag_engine import RAGEngine


def test_rag_initialize_with_empty_dir(temp_chroma_dir):
    """空知识库目录时不崩溃，vector_db 为 None。"""
    engine = RAGEngine(
        knowledge_dir=temp_chroma_dir,
        persist_directory=temp_chroma_dir / "chroma",
    )
    assert engine.vector_db is None


def test_rag_query_without_db(temp_chroma_dir):
    """vector_db 为 None 时 query 返回空字符串。"""
    engine = RAGEngine(
        knowledge_dir=temp_chroma_dir,
        persist_directory=temp_chroma_dir / "chroma",
    )
    result = engine.query("测试查询")
    assert result == ""


def test_rag_initialize_with_files(temp_knowledge_dir, temp_chroma_dir):
    """有 .txt 文件时应成功建立索引。"""
    engine = RAGEngine(
        knowledge_dir=temp_knowledge_dir,
        persist_directory=temp_chroma_dir / "chroma",
    )
    assert engine.vector_db is not None


def test_rag_query_returns_results(temp_knowledge_dir, temp_chroma_dir):
    """应能检索到相关文档片段。"""
    engine = RAGEngine(
        knowledge_dir=temp_knowledge_dir,
        persist_directory=temp_chroma_dir / "chroma",
    )
    result = engine.query("小明喜欢什么")
    assert result != ""
    assert "小明" in result or "爬山" in result
