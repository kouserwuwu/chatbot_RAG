import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from chatbot.api.server import app, session_manager, conv_store
    # 清空所有会话和数据库，确保测试隔离
    for uid in list(session_manager._sessions.keys()):
        session_manager.clear_session(uid)
    # 清理测试数据库
    import sqlite3
    conn = sqlite3.connect(str(conv_store.db_path))
    conn.execute("DELETE FROM messages")
    conn.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()
    return TestClient(app)


# ── 原有测试 ──

def test_status_endpoint(client):
    """GET /status 返回内存状态。"""
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "l0_count" in data


def test_status_endpoint_with_user_id(client):
    """带 X-User-Id 头请求状态。"""
    resp = client.get("/status", headers={"X-User-Id": "alice"})
    assert resp.status_code == 200
    assert resp.json()["l0_count"] == 0


def test_chat_endpoint(client):
    """POST /chat 返回有效响应（含 sources 和 tokens_used）。"""
    resp = client.post("/chat", json={"message": "你好"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "thought_process" in data
    assert "sources" in data
    assert "tokens_used" in data


def test_chat_with_user_id(client):
    """带 X-User-Id 头的聊天请求。"""
    resp = client.post(
        "/chat",
        json={"message": "你好"},
        headers={"X-User-Id": "bob"},
    )
    assert resp.status_code == 200
    assert "answer" in resp.json()


def test_chat_empty_message(client):
    """空消息返回 400 结构化错误。"""
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_ERROR"


def test_clear_endpoint(client):
    """POST /clear 成功。"""
    resp = client.post("/clear", headers={"X-User-Id": "test_clear"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


# ── 新增测试：健康检查 ──

def test_health_endpoint(client):
    """GET /health 返回服务状态。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "checks" in data


# ── 新增测试：会话持久化 ──

def test_conversations_list(client):
    """GET /conversations 返回对话列表（初始为空）。"""
    resp = client.get("/conversations", headers={"X-User-Id": "test_conv"})
    assert resp.status_code == 200
    data = resp.json()
    assert "conversations" in data
    assert "count" in data


def test_chat_creates_conversation(client):
    """聊天后自动创建持久化对话记录。"""
    resp = client.post(
        "/chat",
        json={"message": "你好"},
        headers={"X-User-Id": "test_persist"},
    )
    assert resp.status_code == 200

    # 验证对话已持久化
    resp2 = client.get("/conversations", headers={"X-User-Id": "test_persist"})
    data = resp2.json()
    assert data["count"] >= 1


def test_conversation_messages_endpoint(client):
    """GET /conversations/{id}/messages 返回消息历史。"""
    # 先发一条消息创建对话
    resp = client.post(
        "/chat",
        json={"message": "你好"},
        headers={"X-User-Id": "test_msgs"},
    )
    assert resp.status_code == 200

    # 获取对话列表
    resp2 = client.get("/conversations", headers={"X-User-Id": "test_msgs"})
    convs = resp2.json()["conversations"]

    if convs:
        conv_id = convs[0]["id"]
        resp3 = client.get(f"/conversations/{conv_id}/messages")
        assert resp3.status_code == 200
        msgs = resp3.json()["messages"]
        assert len(msgs) >= 2  # user + assistant


# ── 新增测试：知识库管理 ──

def test_list_knowledge(client):
    """GET /knowledge 返回文档列表。"""
    resp = client.get("/knowledge")
    assert resp.status_code == 200
    assert "documents" in resp.json()


def test_add_and_delete_knowledge(client):
    """POST + DELETE /knowledge 动态增删文档。"""
    # 添加
    resp = client.post(
        "/knowledge?name=test_doc&content=测试文档内容",
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # 确认加入了列表
    resp2 = client.get("/knowledge")
    docs = resp2.json()["documents"]
    assert any("test_doc" in d["name"] for d in docs)

    # 删除
    resp3 = client.delete("/knowledge/test_doc.txt")
    assert resp3.status_code == 200

    # 确认删除
    resp4 = client.get("/knowledge")
    docs2 = resp4.json()["documents"]
    assert not any("test_doc" in d["name"] for d in docs2)


def test_reload_knowledge(client):
    """POST /knowledge/reload 热重载知识库。"""
    resp = client.post("/knowledge/reload")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
