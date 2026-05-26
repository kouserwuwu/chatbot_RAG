from pathlib import Path
from openai import OpenAI

from chatbot.config.settings import settings
from chatbot.logging_config import get_logger


def _load_system_prompt(path: Path | None = None) -> str:
    """从文件或默认路径加载系统提示词。"""
    path = path or settings.SYSTEM_PROMPT_PATH
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    # 文件不存在时使用内嵌默认值
    return (
        "你是一个专业的企业智能客服助手，为企业客户提供准确、高效的咨询服务。\n\n"
        "【工具使用指南】：\n"
        "你现在拥有调用外部工具的能力。如果你无法直接回答问题，请使用格式 Action: tool_name(args)\n\n"
        "可用工具：calculate, get_current_time, search_knowledge, save_memory, recall_memory, list_all_memories\n\n"
        "注意：请仅在必要时调用工具。"
    )


class LLMClient:
    """
    LLM 客户端，封装 API 调用和分级记忆管理。

    系统提示词可通过 system_prompt_path 参数从文件加载，
    支持多租户部署时按客户定制 Agent 人设。

    三级记忆系统:
      L0 (短期) → 保留最近 N 轮对话，满后触发 L1 压缩
      L1 (阶段摘要) → 保留 N 个阶段性总结，满后触发 L2 压缩
      L2 (全局共识) → 单条核心摘要，永不丢失的关键信息
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        system_prompt_path: Path | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        max_history: int | None = None,
        max_l1: int | None = None,
        summary_temperature: float | None = None,
    ):
        self.logger = get_logger("llm_client")

        base_url = base_url or settings.API_BASE_URL
        model_name = model_name or settings.MODEL_NAME

        self.client = OpenAI(base_url=base_url, api_key=settings.API_KEY)

        # ── 记忆参数 ──
        self.max_history = max_history or settings.MAX_HISTORY
        self.max_l1 = max_l1 or settings.MAX_L1_SUMMARIES
        self.summary_temperature = summary_temperature or settings.SUMMARY_TEMPERATURE

        # ── 系统提示词（优先级：直接传入 > 文件路径 > settings 默认路径） ──
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = _load_system_prompt(system_prompt_path)

        self.logger.info("LLMClient 已初始化，提示词来源: %s",
                         "直接传入" if system_prompt else str(system_prompt_path or settings.SYSTEM_PROMPT_PATH))

        # ── 记忆状态 ──
        self.history: list[dict] = []
        self.l1_summaries: list[str] = []
        self.l2_summary: str = ""

    def _generate_summary(self, content_to_summarize: str, level: str) -> str:
        """调用 LLM 将给定内容压缩成摘要。"""
        prompt = (
            f"你现在是一个记忆管理模块。请将以下【{level}】内容压缩成一个极简的摘要。\n"
            f"要求：保留所有核心事实、关键决策、用户偏好和重要共识，剔除礼貌用语和冗余信息。\n"
            f"摘要必须精炼，直接给出结果，不要说'总结如下'。\n\n"
            f"【待压缩内容】：\n{content_to_summarize}"
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a professional memory compressor."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.summary_temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning("摘要压缩失败: %s", e)
            return "（摘要生成失败）"

    def compact(self) -> None:
        """执行分级压缩: L0 → L1 → L2。"""
        self.logger.info("触发内存压缩流程...")

        # L0 → L1
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.history])
        l1_summary = self._generate_summary(history_text, "短期对话历史")
        self.l1_summaries.append(l1_summary)
        self.history = []
        self.logger.info("L1 压缩完成: 已生成第 %d 个阶段性摘要", len(self.l1_summaries))

        # L1 → L2
        if len(self.l1_summaries) >= self.max_l1:
            self.logger.info("L1 已满，触发 L2 全局压缩...")
            l1_text = "\n---\n".join(self.l1_summaries)
            self.l2_summary = self._generate_summary(l1_text, "所有一级摘要")
            self.l1_summaries = []
            self.logger.info("L2 压缩完成: 已更新全局核心共识")

    def get_response(self, user_input: str) -> str:
        """
        发送请求给 LLM 并获取回复，同时维护分级记忆。

        Args:
            user_input: 用户输入文本。

        Returns:
            AI 回复文本。
        """
        try:
            # 组装上下文 (L2 → L1 → History)
            messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

            if self.l2_summary:
                messages.append({
                    "role": "system",
                    "content": f"【全局核心共识 (L2)】:\n{self.l2_summary}",
                })

            if self.l1_summaries:
                l1_combined = "\n".join(
                    [f"阶段摘要{i+1}: {s}" for i, s in enumerate(self.l1_summaries)]
                )
                messages.append({
                    "role": "system",
                    "content": f"【近期阶段摘要 (L1)】:\n{l1_combined}",
                })

            messages.extend(self.history)
            messages.append({"role": "user", "content": user_input})

            # 调用 API
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                temperature=settings.TEMPERATURE,
            )

            ai_response = response.choices[0].message.content

            # 更新 L0
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": ai_response})

            # 检查是否触发压缩
            if len(self.history) >= self.max_history:
                self.compact()

            return ai_response

        except Exception as e:
            self.logger.error("LLM API 调用失败: %s", e)
            return f"❌ 发生错误: {str(e)}"

    def clear_memory(self) -> str:
        """清空所有层级记忆。"""
        self.history = []
        self.l1_summaries = []
        self.l2_summary = ""
        self.logger.info("所有记忆已清空")
        return "所有记忆（短期、一级、二级）已彻底清空！"


# 注意：生产环境不应使用全局单例。
# 多用户场景请通过 SessionManager 获取独立的 LLMClient。
# CLI 等单用户场景可使用 create_default_client() 快捷函数。


def create_default_client() -> LLMClient:
    """创建一个使用 settings 默认值的 LLMClient 实例。"""
    return LLMClient()
