from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

class VectorMemoryStore:
    def __init__(self, persist_directory="D:/test/chatbot_v1/chroma_db"):
        self.persist_directory = persist_directory

        # 1. 初始化 Embedding 模型 (复用 v3.0 的模型)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. 创建/加载一个专门用于用户长期记忆的 Collection
        # 我们将知识库(knowledge)和用户记忆(user_memory)分开存储，避免干扰
        self.vector_db = Chroma(
            collection_name="user_memory",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("✅ 长期记忆向量库已就绪 (Collection: user_memory)")

    def save(self, content: str):
        """
        将一条记忆片段存入向量库。
        """
        if not content:
            return "记忆内容为空，无法保存。"

        # 将文字转化为 Document 对象
        doc = Document(page_content=content)
        self.vector_db.add_documents([doc])
        return f"已成功记住: {content}"

    def recall(self, query: str, k=3):
        """
        通过语义搜索找回最相关的记忆片段。
        """
        if not query:
            return "查询词为空。"

        # 执行相似度搜索
        docs = self.vector_db.similarity_search(query, k=k)

        # 拼接结果
        results = "\n".join([doc.page_content for doc in docs])
        return results if results else "没有找到相关的长期记忆。"

    def list_all(self):
        """
        获取所有存储的记忆片段。
        """
        # ChromaDB 的 get() 方法可以获取所有文档
        all_docs = self.vector_db.get()
        contents = all_docs.get('documents', [])

        if not contents:
            return "目前还没有记录任何长期记忆。"

        return "\n".join(contents)

# 实例化向量记忆存储
memory_store = VectorMemoryStore()
