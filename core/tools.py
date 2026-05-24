import datetime
import math
from core.rag_engine import rag_engine
from core.memory_store import memory_store

def calculate(expression: str):
    """
    执行数学计算。
    输入应为有效的 Python 数学表达式，例如 '2**10' 或 'math.sqrt(16)'。
    """
    try:
        # 允许使用 math 库中的函数
        result = eval(expression, {"__builtins__": None}, {"math": math})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算出错: {str(e)}"

def get_current_time():
    """
    获取当前系统时间。
    不需要输入参数。
    """
    now = datetime.datetime.now()
    return f"当前时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def search_knowledge(query: str):
    """
    检索本地知识库。
    当你需要查询关于用户私有信息、特定文档内容或之前定义的知识时使用。
    输入应该是检索关键词或具体问题。
    """
    print(f"🔍 [Tool-RAG] 正在检索知识库: {query}...")
    result = rag_engine.query(query)
    return result if result else "知识库中没有找到相关信息。"

def save_memory(content: str):
    """
    保存一条长期记忆。
    当你意识到用户提供了关于自己的个人偏好、重要事实或关键信息时使用。
    输入应该是要记住的完整事实描述。例如: \"我喜欢在山里徒步\"
    """
    print(f"💾 [Tool-Memory] 正在存储语义记忆: {content}...")
    return memory_store.save(content)

def recall_memory(query: str):
    """
    检索一条长期记忆。
    当你需要回忆用户的某个特定信息时使用。
    输入应该是检索关键词或描述性问题。
    """
    print(f"🧠 [Tool-Memory] 正在回忆语义记忆: {query}...")
    return memory_store.recall(query)

def list_all_memories():
    """
    列出所有已记录的长期记忆。
    """
    print(f"📖 [Tool-Memory] 正在列出所有记忆...")
    return memory_store.list_all()

# 工具映射表，方便 main.py 根据名称直接调用
TOOL_MAP = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "search_knowledge": search_knowledge,
    "save_memory": save_memory,
    "recall_memory": recall_memory,
    "list_all_memories": list_all_memories
}
