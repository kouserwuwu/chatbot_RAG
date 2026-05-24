import os
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

class Settings:
    # Ollama API 基础路径 (兼容 OpenAI 格式)
    API_BASE_URL = "http://localhost:11434/v1"

    # 从系统环境变量读取 API Key
    # 即使 Ollama 本地运行通常不需要 Key，但为了以后扩展到云端，我们保留这个逻辑
    API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

    # 指定使用的模型名称
    MODEL_NAME = "gemma4:31b-cloud"

    # 默认温度 (控制随机性：0 为最严谨，1 为最随机)
    TEMPERATURE = 0.7

# 实例化配置对象方便其他模块直接导入
settings = Settings()
