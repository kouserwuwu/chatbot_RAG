"""会话持久化 SQLite 测试。"""
import pytest
import tempfile
from pathlib import Path
from chatbot.core.persistence import ConversationStore


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = ConversationStore(db_path=Path(tmpdir) / "test.db")
        yield store


def test_create_conversation(db):
    conv_id = db.create_conversation("user1")
    assert conv_id is not None
    assert len(conv_id) == 12


def test_get_active_conversation(db):
    conv_id = db.create_conversation("user1")
    active = db.get_active_conversation("user1")
    assert active is not None
    assert active["id"] == conv_id


def test_list_conversations(db):
    db.create_conversation("user1")
    db.create_conversation("user1")
    convs = db.list_conversations("user1")
    assert len(convs) == 2


def test_save_and_get_messages(db):
    conv_id = db.create_conversation("user1")
    db.save_message_pair(conv_id, "你好", "你好！有什么可以帮助你的？")
    msgs = db.get_messages(conv_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_close_conversation(db):
    conv_id = db.create_conversation("user1")
    db.close_conversation(conv_id)
    active = db.get_active_conversation("user1")
    assert active is None


def test_delete_conversation(db):
    conv_id = db.create_conversation("user1")
    db.save_message_pair(conv_id, "test", "response")
    db.delete_conversation(conv_id)
    msgs = db.get_messages(conv_id)
    assert len(msgs) == 0


def test_load_history_into_llm(db):
    """从数据库恢复历史到 LLMClient。"""
    class FakeLLM:
        def __init__(self):
            self.history = []

    conv_id = db.create_conversation("user1")
    db.save_message_pair(conv_id, "问题1", "回答1")
    db.save_message_pair(conv_id, "问题2", "回答2")

    llm = FakeLLM()
    db.load_history_into_llm(conv_id, llm, limit=10)
    assert len(llm.history) == 4  # 2 user + 2 assistant


def test_user_isolation(db):
    """不同用户的对话互不干扰。"""
    # user1 的对话
    c1 = db.create_conversation("user1")
    db.save_message_pair(c1, "user1的消息", "回复")

    # user2 的对话
    c2 = db.create_conversation("user2")
    db.save_message_pair(c2, "user2的消息", "回复")

    # user1 看不到 user2 的对话
    convs1 = db.list_conversations("user1")
    convs2 = db.list_conversations("user2")
    assert len(convs1) == 1
    assert len(convs2) == 1
    assert convs1[0]["id"] == c1
    assert convs2[0]["id"] == c2
