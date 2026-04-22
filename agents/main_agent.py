from functools import lru_cache
from typing import Any, Iterator, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek

from agents.tool_registry import get_main_tools

load_dotenv()


SYSTEM_PROMPT = (
    "你是一个中文助手。"
    "你需要结合历史对话上下文连续回答，不要丢失会话状态。"
    "当用户询问天气、气温、降雨、风力等信息时，优先调用 get_weather 工具。"
    "get_weather 返回的是结构化天气事实或错误信息，你要基于这些事实自然总结，不要机械照抄字段名，也不要编造缺失的数据。"
    "当用户询问现在几点、当前日期时间、某个时区时间时，优先调用 get_time 工具。"
    "当用户输入一句台词并想知道出处、来源作品、角色时，优先调用 get_quote_source 工具。"
    "get_quote_source 返回的是候选来源和匹配分数，你要用自然语言总结Top候选并提示结果仅供参考。"
    "当用户询问动漫信息、番剧简介、评分、集数、年份或想找某部动漫时，优先调用 get_anime_info 工具。"
    "get_anime_info 返回的是 Jikan 的结构化结果，你要提炼重点并用中文简洁回答。"
    "工具返回后，用自然语言给出结论，简洁清晰。"
    "不要每次回答都要先来一句根据查询结果"
)


TOOL_STATUS_TEXT = {
    "get_weather": "🔧 正在查询天气...",
    "get_time": "🕒 正在查询时间...",
    "get_quote_source": "🎬 正在分析台词来源...",
    "get_anime_info": "📺 正在检索动漫信息...",
}


@lru_cache(maxsize=4)
def _build_cached_main_agent(model_name: str):
    llm = ChatDeepSeek(model=model_name, temperature=0.7)
    return create_agent(model=llm, tools=get_main_tools(), system_prompt=SYSTEM_PROMPT)


def build_main_agent(model_name: str = "deepseek-chat"):
    return _build_cached_main_agent(model_name)


def _normalize_content(content: Any) -> str | list[Any]:
    if isinstance(content, (str, list)):
        return content
    return str(content)


def _to_langchain_message(message: dict[str, Any]) -> BaseMessage:
    role = str(message.get("role", "user"))
    content = _normalize_content(message.get("content", ""))

    if role in {"user", "human"}:
        return HumanMessage(content=content)
    if role in {"assistant", "ai"}:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            return AIMessage(content=content, tool_calls=cast(list[dict[str, Any]], tool_calls))
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        tool_call_id = str(message.get("tool_call_id", "tool_call"))
        return ToolMessage(content=content, tool_call_id=tool_call_id)

    return HumanMessage(content=content)


def _normalize_messages(messages: list[dict]) -> list[BaseMessage]:
    normalized: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, dict):
            normalized.append(_to_langchain_message(cast(dict[str, Any], message)))
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
            elif hasattr(item, "get") and callable(getattr(item, "get")):
                value = item.get("text")
                if value is not None:
                    parts.append(str(value))
                else:
                    parts.append(str(item))
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
    last_tool_name = ""
    for event in agent.stream(payload, stream_mode="messages"):
        if not isinstance(event, tuple) or not event:
            continue

        message_chunk = event[0]
        meta = event[1] if len(event) > 1 and isinstance(event[1], dict) else {}
        node = str(meta.get("langgraph_node", ""))

        if node == "tools":
            tool_name = str(getattr(message_chunk, "name", "") or "")
            if tool_name and tool_name != last_tool_name:
                last_tool_name = tool_name
                status = TOOL_STATUS_TEXT.get(tool_name, "🔧 正在调用工具查询...")
                yield {"event": "status", "type": "status", "content": status}
            continue

        if node != "model":
            continue

        if getattr(message_chunk, "tool_call_chunks", None):
            continue

        content = getattr(message_chunk, "content", "")
        if isinstance(content, str) and content:
            yield {"event": "token", "type": "token", "content": content}

