from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import uuid
from typing import Any, Iterator, Literal
from agents.main_agent import ask_main_agent, stream_main_agent
from main_client.ai_server_client import StreamAIClient
from main_client.ai_vision_client import AIVisionClient

app= FastAPI()
legacy_stream_client = StreamAIClient()
vision_client = AIVisionClient()


def _content_to_text(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
        else:
            parts.append(str(item))
    return "".join(parts)


def _normalize_messages(messages: list["ChatMessage"]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.role
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        content = _content_to_text(message.content)
        normalized.append({"role": role, "content": content})
    return normalized


def _serialize_messages(messages: list["ChatMessage"]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls is not None:
            item["tool_calls"] = message.tool_calls
        serialized.append(item)
    return serialized


def _new_conversation_id() -> str:
    generator = getattr(uuid, "uuid7", None)
    if callable(generator):
        return str(generator()).replace("-", "")
    return uuid.uuid4().hex


def _resolve_conversation_id(conversation_id: str | None) -> str:
    if isinstance(conversation_id, str) and conversation_id.strip():
        return conversation_id.strip()
    return _new_conversation_id()


def _has_images(messages: list[ChatMessage]) -> bool:
    """检测消息列表中是否包含图片内容（content 为 list 即为多模态消息）"""
    for msg in messages:
        if isinstance(msg.content, list):
            return True
    return False


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    messages: list["ChatMessage"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "human", "ai"] = "user"
    content: str | list[Any] = ""
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class WeatherRequest(BaseModel):
    conversation_id: str | None = None
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):
    conversation_id = _resolve_conversation_id(request.conversation_id)
    response = ask_main_agent(_serialize_messages(request.messages), conversation_id=conversation_id)
    return {"conversation_id": conversation_id, "response": response}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    conversation_id = _resolve_conversation_id(request.conversation_id)

    if _has_images(request.messages):
        # 含图片 → 走 vision 直调路径
        return StreamingResponse(
            _stream_vision(request.messages, conversation_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-Id": conversation_id,
            },
        )

    # 纯文本 → 走原 LangChain Agent 路径
    def event_generator():
        try:
            meta_payload = json.dumps(
                {"event": "meta", "type": "meta", "conversation_id": conversation_id},
                ensure_ascii=False,
            )
            yield f"event: meta\ndata: {meta_payload}\n\n"

            for event in stream_main_agent(
                _serialize_messages(request.messages), conversation_id=conversation_id
            ):
                event_name = str(event.get("event", "token"))
                payload = json.dumps(event, ensure_ascii=False)
                yield f"event: {event_name}\ndata: {payload}\n\n"

            done_payload = json.dumps(
                {"event": "done", "type": "done", "content": "[DONE]"},
                ensure_ascii=False,
            )
            yield f"event: done\ndata: {done_payload}\n\n"
        except Exception as exc:
            payload = json.dumps(
                {"event": "error", "type": "error", "error": str(exc)},
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Conversation-Id": conversation_id,
        },
    )


@app.post("/chat/stream/legacy")
async def chat_stream_legacy(request: ChatRequest):
    def event_generator():
        try:
            for token in legacy_stream_client.generate_response(_normalize_messages(request.messages)):
                payload = json.dumps({"event": "token", "type": "token", "content": token}, ensure_ascii=False)
                yield f"event: token\ndata: {payload}\n\n"

            done_payload = json.dumps({"event": "done", "type": "done", "content": "[DONE]"}, ensure_ascii=False)
            yield f"event: done\ndata: {done_payload}\n\n"
        except Exception as exc:
            payload = json.dumps({"event": "error", "type": "error", "error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _stream_vision(
    messages: list[ChatMessage],
    conversation_id: str,
) -> Iterator[str]:
    """处理多模态（图片+文字）消息，直调 DeepSeek Vision API"""
    serialized = _serialize_messages(messages)
    try:
        meta_payload = json.dumps(
            {"event": "meta", "type": "meta", "conversation_id": conversation_id},
            ensure_ascii=False,
        )
        yield f"event: meta\ndata: {meta_payload}\n\n"

        for event in vision_client.stream_chat(serialized):
            event_name = str(event.get("event", "token"))
            payload = json.dumps(event, ensure_ascii=False)
            yield f"event: {event_name}\ndata: {payload}\n\n"

        done_payload = json.dumps(
            {"event": "done", "type": "done", "content": "[DONE]"},
            ensure_ascii=False,
        )
        yield f"event: done\ndata: {done_payload}\n\n"
    except Exception as exc:
        payload = json.dumps(
            {"event": "error", "type": "error", "error": str(exc)},
            ensure_ascii=False,
        )
        yield f"event: error\ndata: {payload}\n\n"


@app.post("/weather/chat")
async def weather_chat(request: WeatherRequest):
    conversation_id = _resolve_conversation_id(request.conversation_id)
    response = ask_main_agent(
        [{"role": "user", "content": request.question}],
        conversation_id=conversation_id,
    )
    return {"conversation_id": conversation_id, "response": response}

