import pytest
from chatbot.core.agent import Agent, AgentResult, AgentStep
from chatbot.core.tools import calculate, get_current_time


def test_agent_direct_answer(fake_llm):
    """无工具调用：直接返回最终回答。"""
    fake_llm.responses = ["你好！我是AI助手。"]
    agent = Agent(llm_client=fake_llm, tool_registry={}, max_iterations=3)

    result = agent.run("你好")

    assert isinstance(result, AgentResult)
    assert result.answer == "你好！我是AI助手。"
    assert result.iterations == 1
    assert len(result.steps) == 1
    assert result.steps[0].tool_name is None


def test_agent_single_tool_call(fake_llm):
    """1次工具调用，然后最终回答。"""
    fake_llm.responses = [
        'Action: calculate(2**10)\n让我计算一下。',
        "2的10次方等于1024。",
    ]
    agent = Agent(
        llm_client=fake_llm,
        tool_registry={"calculate": calculate},
        max_iterations=3,
    )

    result = agent.run("2的10次方是多少")

    assert result.answer == "2的10次方等于1024。"
    assert result.iterations == 2
    assert len(result.steps) == 2
    assert result.steps[0].tool_name == "calculate"
    assert result.steps[0].tool_args == "2**10"


def test_agent_multi_tool_call(fake_llm):
    """多次工具调用，然后最终回答。"""
    fake_llm.responses = [
        'Action: get_current_time()\n先获取时间。',
        'Action: calculate(100*2)\n再计算一下。',
        "计算结果是200。",
    ]
    agent = Agent(
        llm_client=fake_llm,
        tool_registry={"get_current_time": get_current_time, "calculate": calculate},
        max_iterations=3,
    )

    result = agent.run("现在几点，然后算100乘2")

    assert result.answer == "计算结果是200。"
    assert result.iterations == 3
    assert result.steps[0].tool_name == "get_current_time"
    assert result.steps[1].tool_name == "calculate"


def test_agent_max_iterations(fake_llm):
    """达到最大迭代次数，返回超限结果。"""
    fake_llm.responses = [
        'Action: tool_a()\n试试工具A。',
        'Action: tool_b()\n试试工具B。',
        'Action: tool_c()\n再试工具C。',
    ]
    agent = Agent(llm_client=fake_llm, tool_registry={}, max_iterations=3)

    result = agent.run("测试")

    assert result.iterations == 3
    assert "迭代上限" in result.answer


def test_agent_tool_not_found(fake_llm):
    """工具不存在时，Agent 告知错误并尝试恢复。"""
    fake_llm.responses = [
        'Action: nonexistent_tool(some arg)\n试试不存在的工具。',
        "抱歉，我无法完成这个请求。",
    ]
    agent = Agent(llm_client=fake_llm, tool_registry={}, max_iterations=3)

    result = agent.run("用不存在的工具")
    assert result.iterations == 2
    assert result.steps[0].tool_name == "nonexistent_tool"


def test_agent_empty_input(fake_llm):
    """空输入也应正常执行（LLM 决定如何回应）。"""
    fake_llm.responses = ["请输入一些内容。"]
    agent = Agent(llm_client=fake_llm, tool_registry={}, max_iterations=3)

    result = agent.run("")
    assert result.answer is not None
