import os
from typing import Any, Iterator

from dotenv import load_dotenv
from openai import OpenAI

from deepseek_defaults import resolve_vision_model

load_dotenv()


class AIVisionClient:
    """直调 DeepSeek Vision API，用于多模态图文消息处理"""

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY 配置")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = resolve_vision_model()
        timeout_raw = os.getenv("DEEPSEEK_VISION_TIMEOUT_SEC", "35").strip()
        try:
            self.timeout_sec = max(5.0, float(timeout_raw))
        except ValueError:
            self.timeout_sec = 35.0

    def stream_chat(
        self, messages: list[dict[str, Any]]
    ) -> Iterator[dict[str, str]]:
        """
        流式调用 DeepSeek Vision API。
        messages 已是 OpenAI 标准格式（含 image_url 的 content array）。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            timeout=self.timeout_sec,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield {"event": "token", "type": "token", "content": content}
