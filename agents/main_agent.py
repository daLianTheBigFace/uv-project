from typing import Any, Iterator, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from agents.tool_registry import get_main_tools

load_dotenv()


def build_main_agent(model_name: str = "deepseek-chat"):
    system_prompt = (
        "你是一个中文助手。"
        "你需要结合历史对话上下文连续回答，不要丢失会话状态。"
        "当用户询问天气、气温、降雨、风力等信息时，优先调用 get_weather 工具。"
        "get_weather 返回的是结构化天气事实或错误信息，你要基于这些事实自然总结，不要机械照抄字段名，也不要编造缺失的数据。"
        "当用户询问现在几点、当前日期时间、某个时区时间时，优先调用 get_time 工具。"
        "工具返回后，用自然语言给出结论，简洁清晰。"
        "不要每次回答都要先来一句根据查询结果"
    )
    llm = ChatDeepSeek(model=model_name, temperature=0.7)
    return create_agent(model=llm, tools=get_main_tools(), system_prompt=system_prompt)


def _normalize_messages(messages: list[dict]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_text(agent_result: dict[str, Any]) -> str:
    messages = agent_result.get("messages") or []
    if not messages:
        return "未获得模型回复"

    content = getattr(messages[-1], "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def ask_main_agent(messages: list[dict], model_name: str = "deepseek-chat") -> str:
    agent = build_main_agent(model_name=model_name)
    payload = cast(Any, {"messages": _normalize_messages(messages)})
    result = agent.invoke(payload)
    return _extract_text(result)


def stream_main_agent(messages: list[dict], model_name: str = "deepseek-chat") -> Iterator[dict[str, str]]:
    agent = build_main_agent(model_name=model_name)
    payload = cast(Any, {"messages": _normalize_messages(messages)})
    tool_status_sent = False
    for event in agent.stream(payload, stream_mode="messages"):
        if not isinstance(event, tuple) or not event:
            continue

        message_chunk = event[0]
        meta = event[1] if len(event) > 1 and isinstance(event[1], dict) else {}
        node = str(meta.get("langgraph_node", ""))

        if node == "tools" and not tool_status_sent:
            tool_status_sent = True
            tool_name = str(getattr(message_chunk, "name", "") or "")
            status = "🔧 正在调用工具查询..."
            if tool_name == "get_weather":
                status = "🔧 正在查询天气..."
            elif tool_name == "get_time":
                status = "🕒 正在查询时间..."
            yield {"event": "status", "type": "status", "content": status}
            continue

        if node != "model":
            continue

        if getattr(message_chunk, "tool_call_chunks", None):
            continue

        content = getattr(message_chunk, "content", "")
        if isinstance(content, str) and content:
            yield {"event": "token", "type": "token", "content": content}

