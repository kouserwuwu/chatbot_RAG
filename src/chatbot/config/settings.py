import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── 项目根目录（src/chatbot/config/settings.py → 上溯3层到项目根）──
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
    SRC_DIR = ROOT_DIR / "src"

    # ── 数据目录 ──
    DATA_DIR = Path(os.getenv("CHATBOT_DATA_DIR", ROOT_DIR / "data"))
    KNOWLEDGE_DIR = Path(os.getenv("CHATBOT_KNOWLEDGE_DIR", ROOT_DIR / "knowledge"))
    CHROMA_DIR = Path(os.getenv("CHATBOT_CHROMA_DIR", DATA_DIR / "chroma_db"))
    LOG_DIR = ROOT_DIR / "logs"

    # ── Ollama API ──
    API_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", "0.7"))

    # ── 嵌入模型 ──
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── 分级记忆参数 ──
    MAX_HISTORY = int(os.getenv("CHATBOT_MAX_HISTORY", "10"))
    MAX_L1_SUMMARIES = int(os.getenv("CHATBOT_MAX_L1", "10"))

    # ── RAG 参数 ──
    RAG_CHUNK_SIZE = int(os.getenv("CHATBOT_CHUNK_SIZE", "300"))
    RAG_CHUNK_OVERLAP = int(os.getenv("CHATBOT_CHUNK_OVERLAP", "50"))
    RAG_TOP_K = int(os.getenv("CHATBOT_RAG_TOP_K", "3"))

    # ── Agent 参数 ──
    MAX_ITERATIONS = int(os.getenv("CHATBOT_MAX_ITERATIONS", "3"))

    # ── 长期记忆参数 ──
    MEMORY_TOP_K = int(os.getenv("CHATBOT_MEMORY_TOP_K", "3"))

    # ── 摘要生成 ──
    SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.3"))

    # ── 日志 ──
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # ── 系统提示词 ──
    SYSTEM_PROMPT_PATH = Path(os.getenv(
        "CHATBOT_SYSTEM_PROMPT_PATH",
        ROOT_DIR / "prompts" / "system_prompt.txt",
    ))

    # ── 数据库 ──
    DATABASE_PATH = Path(os.getenv(
        "CHATBOT_DATABASE_PATH",
        DATA_DIR / "chatbot.db",
    ))


settings = Settings()
