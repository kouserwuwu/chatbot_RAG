from openai import OpenAI
from config.settings import settings

class LLMClient:
    def __init__(self):
        # 初始化 OpenAI 客户端，指向 Ollama 本地 API
        self.client = OpenAI(
            base_url=settings.API_BASE_URL,
            api_key=settings.API_KEY
        )
        # 定义 Agent 的初始人格 (System Prompt) - v6.1 工业级语义记忆版
        self.system_prompt = (
            "你是一个友好的 AI 助手，正在帮助一名 Agent 零基础学习者进行实操练习。请用简洁且专业的方式回答问题。\n\n"
            "【工具使用指南】：\n"
            "你现在拥有调用外部工具的能力。如果你无法直接回答问题，请使用以下格式请求调用工具：\n"
            "Action: tool_name(args)\n\n"
            "可用工具列表：\n"
            "1. calculate(expression): 执行数学计算。 例如：Action: calculate(2**10)\n"
            "2. get_current_time(): 获取当前系统时间。调用格式：Action: get_current_time()\n"
            "3. search_knowledge(query): 检索本地知识库（静态文档）。当你需要查询私有信息或特定文档内容时使用。例如：Action: search_knowledge(\"我的名字是什么\")\n"
            "4. save_memory(content): 保存一条长期语义记忆。当你意识到用户提供了个人偏好、重要事实或关键信息时使用。不需要 Key，直接存入事实描述。例如：Action: save_memory(\"用户喜欢在山里徒步\")\n"
            "5. recall_memory(query): 检索长期语义记忆。当你需要回忆用户的个人特质、偏好或之前提到的事实时使用。无需精确 Key，使用描述性查询。例如：Action: recall_memory(\"用户喜欢什么户外活动\")\n"
            "6. list_all_memories(): 列出所有已记录的长期记忆。调用格式：Action: list_all_memories()\n\n"
            "注意：请仅在必要时调用工具。如果调用一个检索工具（如 recall_memory）未获得结果，建议尝试另一个检索工具（如 search_knowledge）以确保信息覆盖完整。调用后，我会给你返回工具的执行结果，请你基于该结果给出最终回答。"
        )

        # --- v7.0 分级记忆系统 ---
        # 1. 短期对话历史 (L0)
        self.history = []
        self.max_history = 10  # 达到 10 条则触发 L1 压缩

        # 2. 一级摘要记录 (L1) - 存储最近 10 个片段总结
        self.l1_summaries = []
        self.max_l1 = 10       # 达到 10 个 L1 则触发 L2 压缩

        # 3. 二级全局摘要 (L2) - 存储一个最终的核心共识
        self.l2_summary = ""

    def _generate_summary(self, content_to_summarize: str, level: str) -> str:
        """
        调用 LLM 将给定的内容压缩成摘要
        """
        prompt = (
            f"你现在是一个记忆管理模块。请将以下【{level}】内容压缩成一个极简的摘要。\n"
            f"要求：保留所有核心事实、关键决策、用户偏好和重要共识，剔除礼貌用语和冗余信息。\n"
            f"摘要必须精炼，直接给出结果，不要说'总结如下'。\n\n"
            f"【待压缩内容】：\n{content_to_summarize}"
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[{"role": "system", "content": "You are a professional memory compressor."},
                          {"role": "user", "content": prompt}],
                temperature=0.3 # 低温度保证稳定性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ 压缩失败: {e}")
            return "（摘要生成失败）"

    def compact(self):
        """
        执行分级压缩逻辑: History -> L1 -> L2
        """
        print("\n🌀 [Memory-Compact] 触发内存压缩流程...")

        # --- 步骤 1: L0 -> L1 ---
        # 将当前的 history 转化为一个 L1 摘要
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.history])
        l1_summary = self._generate_summary(history_text, "短期对话历史")
        self.l1_summaries.append(l1_summary)

        # 清空 L0
        self.history = []
        print(f"✅ L1 压缩完成: 已生成第 {len(self.l1_summaries)} 个阶段性摘要")

        # --- 步骤 2: L1 -> L2 ---
        if len(self.l1_summaries) >= self.max_l1:
            print("🌀 [Memory-Compact] L1 已满，触发二级 L2 全局压缩...")
            l1_text = "\n---\n".join(self.l1_summaries)
            self.l2_summary = self._generate_summary(l1_text, "所有一级摘要")

            # 清空 L1
            self.l1_summaries = []
            print(f"✅ L2 压缩完成: 已更新全局核心共识")

    def get_response(self, user_input: str) -> str:
        """
        发送请求给 LLM 并获取回复，同时维护分级记忆
        """
        try:
            # 1. 组装上下文 (L2 -> L1s -> History)
            messages = [{"role": "system", "content": self.system_prompt}]

            # 注入 L2 全局摘要
            if self.l2_summary:
                messages.append({"role": "system", "content": f"【全局核心共识 (L2)】:\n{self.l2_summary}"})

            # 注入 L1 阶段摘要
            if self.l1_summaries:
                l1_combined = "\n".join([f"阶段摘要{i+1}: {s}" for i, s in enumerate(self.l1_summaries)])
                messages.append({"role": "system", "content": f"【近期阶段摘要 (L1)】:\n{l1_combined}"})

            # 注入当前对话历史
            messages.extend(self.history)

            # 加入当前用户输入
            messages.append({"role": "user", "content": user_input})

            # 2. 调用 API
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                temperature=settings.TEMPERATURE
            )

            ai_response = response.choices[0].message.content

            # 3. 更新 L0 记忆
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": ai_response})

            # 4. 检查是否触发压缩 (当 history 达到 max_history)
            if len(self.history) >= self.max_history:
                self.compact()

            return ai_response

        except Exception as e:
            return f"❌ 发生错误: {str(e)}"

    def clear_memory(self):
        """
        清空所有层级记忆
        """
        self.history = []
        self.l1_summaries = []
        self.l2_summary = ""
        return "所有记忆（短期、一级、二级）已彻底清空！"

# 实例化客户端
llm_client = LLMClient()
