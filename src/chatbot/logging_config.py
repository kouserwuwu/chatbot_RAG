import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """
    配置根日志器：控制台 + 文件 (RotatingFileHandler)。

    Args:
        level: 日志级别，可通过 LOG_LEVEL 环境变量覆盖。
        log_dir: 日志目录，默认为项目根目录下的 logs/。
        max_bytes: 单个日志文件最大字节数。
        backup_count: 保留的历史日志文件数量。
    """
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"

    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("chatbot")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler（多次调用 setup_logging 时）
    if logger.handlers:
        return

    # 格式
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 文件 handler（自动轮转）
    file_handler = RotatingFileHandler(
        log_dir / "chatbot.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # 抑制第三方库的日志噪音
    for lib in ("chromadb", "sentence_transformers", "httpx", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取 chatbot 命名空间下的日志器。"""
    return logging.getLogger(f"chatbot.{name}")
