import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGEngine:
    def __init__(self, knowledge_dir="D:/test/chatbot_v1/knowledge"):
        self.knowledge_dir = knowledge_dir
        self.persist_directory = "D:/test/chatbot_v1/chroma_db"

        # 1. 初始化 Embedding 模型 (本地运行)
        # 使用一个小巧且强大的模型将文字转化为向量
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. 初始化向量数据库
        self.vector_db = None
        self._initialize_db()

    def _initialize_db(self):
        """
        扫描知识库目录，加载文档并建立索引
        """
        # 检查是否有 .txt 文件
        files = [f for f in os.listdir(self.knowledge_dir) if f.endswith('.txt')]
        if not files:
            print("⚠️ 知识库文件夹中没有找到 .txt 文件。")
            return

        all_docs = []
        for file in files:
            file_path = os.path.join(self.knowledge_dir, file)
            # 加载文档
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()

            # 切片：将长文档切成小块，方便 AI 精准检索
            # chunk_size 是每个块的大小，chunk_overlap 是块之间重叠的部分（防止语义断层）
            text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            all_docs.extend(splits)

        # 将切片后的文档存入 Chroma 向量数据库
        self.vector_db = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ 知识库索引构建完成，共加载 {len(all_docs)} 个文档片段。")

    def query(self, user_query: str, k=3):
        """
        语义检索：在知识库中寻找与用户问题最相关的 k 个片段
        """
        if not self.vector_db:
            return ""

        # 执行相似度搜索 (Similarity Search)
        # 这一步会将 user_query 转化为向量，并计算与库中向量的余弦距离
        docs = self.vector_db.similarity_search(user_query, k=k)

        # 将检索到的片段拼接成一个字符串
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

# 实例化 RAG 引擎
rag_engine = RAGEngine()
