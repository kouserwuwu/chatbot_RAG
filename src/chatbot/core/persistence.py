"""会话持久化层：SQLite 存储对话历史，服务重启后可恢复。"""

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock

from chatbot.config.settings import settings
from chatbot.logging_config import get_logger

logger = get_logger("persistence")


class ConversationStore:
    """
    SQLite 持久化存储，管理对话历史的 CRUD。

    表结构:
      conversations: conversation_id, user_id, created_at, updated_at, status
      messages: id, conversation_id, role, content, created_at
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id, id);
                CREATE INDEX IF NOT EXISTS idx_conversations_user
                    ON conversations(user_id, updated_at DESC);
            """)
            conn.commit()
            conn.close()
            logger.info("数据库已就绪: %s", self.db_path)

    # ── 会话操作 ──

    def create_conversation(self, user_id: str, conv_id: str | None = None) -> str:
        import uuid
        conv_id = conv_id or str(uuid.uuid4())[:12]
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO conversations(id, user_id, created_at, updated_at) VALUES(?,?,?,?)",
                (conv_id, user_id, now, now),
            )
            conn.commit()
            conn.close()
        return conv_id

    def get_active_conversation(self, user_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM conversations WHERE user_id=? AND status='active' ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_conversations(self, user_id: str, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def close_conversation(self, conv_id: str):
        self._update_status(conv_id, "closed")

    def _update_status(self, conv_id: str, status: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE conversations SET status=?, updated_at=? WHERE id=?",
                (status, time.time(), conv_id),
            )
            conn.commit()
            conn.close()

    # ── 消息操作 ──

    def save_message(self, conv_id: str, role: str, content: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
                (conv_id, role, content, time.time()),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (time.time(), conv_id),
            )
            conn.commit()
            conn.close()

    def save_message_pair(self, conv_id: str, user_msg: str, assistant_msg: str):
        """同时保存用户消息和 AI 回复。"""
        with self._lock:
            conn = self._get_conn()
            now = time.time()
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
                (conv_id, "user", user_msg, now),
            )
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
                (conv_id, "assistant", assistant_msg, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conv_id),
            )
            conn.commit()
            conn.close()

    def get_messages(self, conv_id: str, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id ASC LIMIT ?",
            (conv_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_recent_messages(self, conv_id: str, limit: int = 10) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
            (conv_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows][::-1]

    def load_history_into_llm(self, conv_id: str, llm_client, limit: int = 10):
        """从数据库恢复对话历史到 LLMClient 的 L0 记忆。"""
        msgs = self.get_recent_messages(conv_id, limit)
        for msg in msgs:
            llm_client.history.append({
                "role": msg["role"],
                "content": msg["content"],
            })
        logger.info("从数据库恢复 %d 条历史消息到会话 %s", len(msgs), conv_id)

    def delete_conversation(self, conv_id: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
            conn.commit()
            conn.close()


# 全局单例
conv_store = ConversationStore()
