import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from chatbot.config.settings import settings
from chatbot.exceptions import RAGEngineError
from chatbot.logging_config import get_logger


class RAGEngine:
    """
    本地知识库检索引擎。

    负责加载 knowledge/ 目录下的 .txt 文件，切片后存入 ChromaDB 向量数据库，
    提供语义相似度搜索接口。
    """

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        persist_directory: Path | None = None,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.knowledge_dir = knowledge_dir or settings.KNOWLEDGE_DIR
        self.persist_directory = persist_directory or settings.CHROMA_DIR
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self.logger = get_logger("rag_engine")

        # 初始化嵌入模型
        try:
            self.logger.info("Loading embedding model: %s", self.embedding_model)
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        except Exception as e:
            raise RAGEngineError(f"加载嵌入模型失败: {e}") from e

        # 初始化向量数据库
        self.vector_db = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """扫描知识库目录，加载文档并建立索引。重载时会先清空旧索引。"""
        if not self.knowledge_dir.exists():
            self.logger.warning("知识库目录不存在: %s，将被创建", self.knowledge_dir)
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in os.listdir(self.knowledge_dir) if f.endswith(".txt")]
        if not files:
            self.logger.warning("知识库中没有找到 .txt 文件")
            return

        all_docs = []
        # 尝试多种编码顺序
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        for file in files:
            file_path = self.knowledge_dir / file
            loaded = False
            for enc in encodings:
                try:
                    loader = TextLoader(str(file_path), encoding=enc)
                    docs = loader.load()
                    loaded = True
                    break
                except (UnicodeDecodeError, RuntimeError):
                    # RuntimeError: TextLoader 在编码错误时也抛 RuntimeError
                    continue
                except Exception as e:
                    self.logger.error("加载文档失败: %s, 错误: %s", file_path, e)
                    break
            if not loaded:
                self.logger.warning("跳过 %s：所有编码均无法解码", file)
                continue

            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            splits = text_splitter.split_documents(docs)
            all_docs.extend(splits)

        if not all_docs:
            self.logger.warning("没有成功加载任何文档片段")
            return

        try:
            # 重建索引前先删除旧 collection（避免历史和当前数据混合）
            if self.vector_db is not None:
                try:
                    self.vector_db.delete_collection()
                except Exception:
                    pass

            self.vector_db = Chroma.from_documents(
                documents=all_docs,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            self.logger.info("知识库索引构建完成，共 %d 个文档片段", len(all_docs))
        except Exception as e:
            raise RAGEngineError(f"创建向量数据库失败: {e}") from e

    def query(self, user_query: str, k: int | None = None) -> str:
        """
        语义检索：在知识库中寻找与用户问题最相关的片段。

        Args:
            user_query: 用户查询文本。
            k: 返回的结果数量，默认使用 settings.RAG_TOP_K。

        Returns:
            拼接后的检索文本；若数据库未初始化则返回空字符串。
        """
        if self.vector_db is None:
            return ""

        k = k or settings.RAG_TOP_K

        try:
            docs = self.vector_db.similarity_search(user_query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            self.logger.error("知识库检索失败: %s", e)
            return ""


# 全局单例（延迟加载，避免导入时崩溃）
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


# 兼容旧代码
rag_engine = None  # type: ignore  # 通过 get_rag_engine() 获取
