import pytest
from unittest.mock import MagicMock, patch
from chatbot.core.llm_client import LLMClient


@pytest.fixture
def mock_openai_response():
    """模拟 OpenAI API 响应。"""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = "这是一条测试回复。"
    return mock


def test_get_response_adds_to_history(mock_openai_response):
    """get_response 应把问答对添加到 history。"""
    with patch.object(LLMClient, "__init__", lambda self: None):
        client = LLMClient.__new__(LLMClient)
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_openai_response
        client.system_prompt = "test prompt"
        client.history = []
        client.l1_summaries = []
        client.l2_summary = ""
        client.max_history = 10
        client.max_l1 = 10
        client.summary_temperature = 0.3

        response = client.get_response("你好")

        assert response == "这是一条测试回复。"
        assert len(client.history) == 2
        assert client.history[0] == {"role": "user", "content": "你好"}
        assert client.history[1] == {"role": "assistant", "content": "这是一条测试回复。"}


def test_clear_memory():
    """clear_memory 应重置所有层级。"""
    import logging
    with patch.object(LLMClient, "__init__", lambda self: None):
        client = LLMClient.__new__(LLMClient)
        client.logger = logging.getLogger("test")
        client.history = [{"role": "user", "content": "test"}]
        client.l1_summaries = ["摘要1摘要2"]
        client.l2_summary = "全局共识"

        result = client.clear_memory()

        assert client.history == []
        assert client.l1_summaries == []
        assert client.l2_summary == ""
        assert "已彻底清空" in result
