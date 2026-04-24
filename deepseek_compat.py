from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


class ChatDeepSeekCompat(ChatDeepSeek):
    """DeepSeek compatibility wrapper.

    Ensures `reasoning_content` carried in assistant messages is passed back to the
    chat-completions API for follow-up turns, including internal tool-call loops.
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        payload_messages = payload.get("messages")
        if not isinstance(payload_messages, list):
            return payload

        for source_message, payload_message in zip(messages, payload_messages, strict=False):
            if not isinstance(source_message, AIMessage):
                continue
            if not isinstance(payload_message, dict):
                continue
            additional_kwargs = getattr(source_message, "additional_kwargs", None)
            if not isinstance(additional_kwargs, dict):
                continue
            reasoning_content = additional_kwargs.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content:
                payload_message["reasoning_content"] = reasoning_content
        return payload

