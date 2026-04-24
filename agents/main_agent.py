from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from deepseek_compat import ChatDeepSeekCompat
from deepseek_defaults import resolve_chat_model
from agents.trace_logger import finish_run, log_event, start_run
from agents.tool_registry import get_main_tools

load_dotenv()


_PROMPT_EXAMPLES_PATH = Path(__file__).resolve().parent / "prompts" / "main_agent_few_shot.txt"


BASE_SYSTEM_PROMPT = """
你是一个中文助手，始终以简洁、准确、负责任的中文回答用户。

总体规则：
- 优先使用可用工具获取事实性信息；只有在明确没有可用工具或需要澄清时，才直接用常识回答。
- 任何基于工具返回的数据的结论，必须以工具返回内容为依据；不得编造工具没有提供的信息。
- 当工具返回不确定结果（如 weak_match / no_match / error），请说明不确定性并给出可行的下一步建议（如补充上下文或改用其他关键词）。
- 回答要简洁、直接，必要时给出 1-2 句补充说明或建议。始终以用户可操作的信息为主。

工具使用规范：
- get_weather(city): 查询天气相关问题（天气、气温、降雨、风力等）。必须先调用该工具并以其结构化结果为事实基础组织回答。
- get_time(...): 查询当前时间/时区相关问题。必须调用该工具获取精确时间。
- get_quote_source(line[, work_hint, language]): 当用户提供台词或询问出处时使用。工具返回候选作品、匹配分数和来源链接。将 Top 候选以自然语言列出并标注置信度与免责声明（“仅供参考”）。
- get_anime_info(query): 查询动漫元信息（简介、评分、集数、年份等）。以工具的结构化字段为依据提炼要点。

多工具与复杂任务：
- 你可以决定连续调用多个工具以完成复杂任务（例如先用 web_search 获取候选，再调用 get_quote_source 再验证），但每次调用后都要基于工具输出决定下一步。
- 当需要澄清用户意图（例如台词不完整、地点/时间歧义）时，先提问并等待用户确认，而不是盲目调用工具。

输出格式与约束：
- 回答必须为中文。
- 对于事实性结论，优先给简洁结论；若有证据来源，附上简短来源说明（例如来源于 OpenSubtitles / Jikan / 网页）。
- 遇到错误、缺少配置信息（如 API Key）或工具异常时，明确返回友好的错误说明并提示用户可能的解决方法。
- 对于泛建议类问题（如旅游规划、日常建议）即使没有专用工具，也应先给出通用可执行建议，再明确哪些部分可以通过现有工具补充（例如天气）。
- 不要直接以“没有相关工具所以无法提供建议”作为主要回复。
"""


@lru_cache(maxsize=1)
def _load_few_shot_examples() -> str:
    try:
        text = _PROMPT_EXAMPLES_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text


def _build_system_prompt() -> str:
    examples = _load_few_shot_examples()
    if not examples:
        return BASE_SYSTEM_PROMPT.strip()
    return f"{BASE_SYSTEM_PROMPT.strip()}\n\n{examples}"


TOOL_STATUS_TEXT = {
    "get_weather": "🔧 正在查询天气...",
    "get_time": "🕒 正在查询时间...",
    "get_quote_source": "🎬 正在分析台词来源...",
    "get_anime_info": "📺 正在检索动漫信息...",
}


@lru_cache(maxsize=4)
def _build_cached_main_agent(model_name: str):
    llm = ChatDeepSeekCompat(
        model=model_name,
        temperature=0.7,
        use_responses_api=False,
        extra_body={"thinking":{"type":"disabled"}}
    )
    return create_agent(model=llm, tools=get_main_tools(), system_prompt=_build_system_prompt())


def build_main_agent(model_name: str | None = None):
    return _build_cached_main_agent(resolve_chat_model(model_name))


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
        additional_kwargs: dict[str, Any] = {}
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            additional_kwargs["reasoning_content"] = reasoning_content
        if isinstance(tool_calls, list):
            return AIMessage(
                content=content,
                tool_calls=cast(list[dict[str, Any]], tool_calls),
                additional_kwargs=additional_kwargs,
            )
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
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


def _extract_reasoning_content(message: Any) -> str:
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        reasoning = additional_kwargs.get("reasoning_content")
        if isinstance(reasoning, str):
            return reasoning
    return ""


def _extract_last_assistant_payload(agent_result: dict[str, Any]) -> dict[str, Any]:
    messages = agent_result.get("messages") or []
    if not messages:
        return {"content": "未获得模型回复", "reasoning_content": ""}

    assistant_message = messages[-1]
    content = _extract_text({"messages": [assistant_message]})
    payload: dict[str, Any] = {"content": content}

    reasoning_content = _extract_reasoning_content(assistant_message)
    if reasoning_content:
        payload["reasoning_content"] = reasoning_content

    tool_calls = getattr(assistant_message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        payload["tool_calls"] = tool_calls

    return payload


def _log_tool_events_from_messages(run_id: str, messages: list[Any]) -> int:
    seq = 0
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                seq += 1
                tool_name = str(call.get("name", "") or "")
                log_event(run_id, seq, "tool_call", call, tool_name=tool_name)

        if isinstance(message, ToolMessage):
            seq += 1
            payload = {
                "tool_call_id": getattr(message, "tool_call_id", ""),
                "content": getattr(message, "content", ""),
                "name": getattr(message, "name", ""),
            }
            tool_name = str(getattr(message, "name", "") or "")
            log_event(run_id, seq, "tool_result", payload, tool_name=tool_name)
    return seq


def ask_main_agent(
    messages: list[dict],
    model_name: str | None = None,
    conversation_id: str | None = None,
) -> str:
    return ask_main_agent_full(
        messages=messages,
        model_name=model_name,
        conversation_id=conversation_id,
    )["content"]


def ask_main_agent_full(
    messages: list[dict],
    model_name: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    resolved_model_name = resolve_chat_model(model_name)
    agent = build_main_agent(model_name=resolved_model_name)
    payload = cast(Any, {"messages": _normalize_messages(messages)})
    run_id = start_run(
        mode="ask",
        model_name=resolved_model_name,
        input_messages=messages,
        conversation_id=conversation_id,
    )
    try:
        result = agent.invoke(payload)
        result_messages = result.get("messages") or []
        _log_tool_events_from_messages(run_id, result_messages)
        assistant_payload = _extract_last_assistant_payload(result)
        finish_run(run_id, final_output=str(assistant_payload.get("content", "")))
        return assistant_payload
    except Exception as exc:
        finish_run(run_id, error=str(exc))
        raise


def stream_main_agent(
    messages: list[dict],
    model_name: str | None = None,
    conversation_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    resolved_model_name = resolve_chat_model(model_name)
    agent = build_main_agent(model_name=resolved_model_name)
    payload = cast(Any, {"messages": _normalize_messages(messages)})
    last_tool_name = ""
    seq = 0
    output_parts: list[str] = []
    reasoning_parts: list[str] = []
    run_id = start_run(
        mode="stream",
        model_name=resolved_model_name,
        input_messages=messages,
        conversation_id=conversation_id,
    )
    for event in agent.stream(payload, stream_mode="messages"):
        try:
            if not isinstance(event, tuple) or not event:
                continue
            message_chunk = event[0]
            meta = event[1] if len(event) > 1 and isinstance(event[1], dict) else {}
            node = str(meta.get("langgraph_node", ""))

            if node == "tools":
                tool_name = str(getattr(message_chunk, "name", "") or "")
                if tool_name and tool_name != last_tool_name:
                    last_tool_name = tool_name
                    seq += 1
                    log_event(
                        run_id,
                        seq,
                        "tool_call",
                        {"status": "started", "node": node},
                        tool_name=tool_name,
                    )
                    status = TOOL_STATUS_TEXT.get(tool_name, "🔧 正在调用工具查询...")
                    yield {"event": "status", "type": "status", "content": status}
                continue

            if node != "model":
                continue

            if getattr(message_chunk, "tool_call_chunks", None):
                continue

            chunk_reasoning = _extract_reasoning_content(message_chunk)
            if chunk_reasoning:
                reasoning_parts.append(chunk_reasoning)

            content = getattr(message_chunk, "content", "")
            if isinstance(content, str) and content:
                output_parts.append(content)
                yield {"event": "token", "type": "token", "content": content}
        except Exception as exc:
            finish_run(run_id, final_output="".join(output_parts) or None, error=str(exc))
            raise

    final_output = "".join(output_parts)
    finish_run(run_id, final_output=final_output)

    final_reasoning = "".join(reasoning_parts)
    if final_reasoning:
        yield {
            "event": "assistant_meta",
            "type": "assistant_meta",
            "reasoning_content": final_reasoning,
        }

