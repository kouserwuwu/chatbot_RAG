from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from chatbot.config.settings import settings
from chatbot.exceptions import MemoryStoreError
from chatbot.logging_config import get_logger


class VectorMemoryStore:
    """
    长期语义记忆存储（支持用户隔离）。

    每个用户拥有独立的 ChromaDB collection: user_memory_{user_id}，
    确保用户 A 的长期记忆不会被用户 B 检索到。

    用法:
        # 用户隔离模式
        store = VectorMemoryStore(user_id="alice")
        store.save("我喜欢爬山")
        store.recall("户外活动")  # 只检索 alice 的记忆

        # 全局模式（无 user_id，使用默认 collection）
        store = VectorMemoryStore()
    """

    def __init__(
        self,
        user_id: str | None = None,
        persist_directory: Path | None = None,
        embedding_model: str | None = None,
    ):
        self.user_id = user_id
        self.persist_directory = persist_directory or settings.CHROMA_DIR
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.logger = get_logger("memory_store")

        # Collection 命名：按用户隔离
        if user_id:
            collection_name = f"user_memory_{user_id}"
        else:
            collection_name = "user_memory"

        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.vector_db = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            self.logger.info("长期记忆向量库已就绪 (Collection: %s)", collection_name)
        except Exception as e:
            raise MemoryStoreError(f"初始化记忆存储失败: {e}") from e

    def save(self, content: str) -> str:
        """将一条记忆片段存入向量库。"""
        if not content or not content.strip():
            raise MemoryStoreError("记忆内容为空，无法保存")

        try:
            doc = Document(page_content=content.strip())
            self.vector_db.add_documents([doc])
            return f"已成功记住: {content.strip()}"
        except Exception as e:
            raise MemoryStoreError(f"保存记忆失败: {e}") from e

    def recall(self, query: str, k: int | None = None) -> str:
        """通过语义搜索找回最相关的记忆片段。"""
        if not query or not query.strip():
            return "查询词为空"

        k = k or settings.MEMORY_TOP_K

        try:
            docs = self.vector_db.similarity_search(query, k=k)
        except Exception as e:
            self.logger.error("记忆检索失败: %s", e)
            return "记忆检索失败"

        results = "\n".join([doc.page_content for doc in docs])
        return results if results else "没有找到相关的长期记忆"

    def list_all(self) -> str:
        """获取所有已存储的记忆片段。"""
        try:
            all_docs = self.vector_db.get()
            contents = all_docs.get("documents", [])
        except Exception as e:
            self.logger.error("列出记忆失败: %s", e)
            return "获取记忆列表失败"

        if not contents:
            return "目前还没有记录任何长期记忆"

        return "\n".join(contents)

    def delete(self, content: str) -> str:
        """删除指定内容的记忆。"""
        if not content or not content.strip():
            return "删除内容为空"

        try:
            all_docs = self.vector_db.get()
            ids = all_docs.get("ids", [])
            documents = all_docs.get("documents", [])

            to_delete = [
                id_ for id_, doc in zip(ids, documents)
                if doc.strip() == content.strip()
            ]
            if to_delete:
                self.vector_db.delete(ids=to_delete)
                return f"已删除 {len(to_delete)} 条记忆"
            return "未找到匹配的记忆"
        except Exception as e:
            self.logger.error("删除记忆失败: %s", e)
            return f"删除失败: {e}"

    @property
    def count(self) -> int:
        """当前存储的记忆条数。"""
        try:
            return len(self.vector_db.get().get("documents", []))
        except Exception:
            return 0


# 向后兼容：无用户隔离的全局单例
_memory_store: VectorMemoryStore | None = None


def get_memory_store() -> VectorMemoryStore:
    """获取全局默认记忆存储（向后兼容 CLI 等单用户场景）。"""
    global _memory_store
    if _memory_store is None:
        _memory_store = VectorMemoryStore()
    return _memory_store
