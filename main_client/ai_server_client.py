from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import cast, List, Dict, Iterator

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

class AIClient:
    def __init__(self):
        self.client = OpenAI(api_key=api_key,base_url="https://api.deepseek.com")

    def generate_response(self, user_msg: List[Dict[str, str]]) -> str:
        client_response = self.client.chat.completions.create(  # type: ignore[arg-type]
            model="deepseek-chat",
            messages=user_msg,
            stream=False
        )
        return cast(str, client_response.choices[0].message.content)
class StreamAIClient(AIClient):
    def generate_response(self, user_msg: List[Dict[str, str]]) -> Iterator[str]:
        client_response = self.client.chat.completions.create(  # type: ignore[arg-type]
            model="deepseek-chat",
            messages=user_msg,
            stream=True,
            # temperature=1.0,
            # top_p=1.0
        )
        for stream_chunk in client_response:
            content = stream_chunk.choices[0].delta.content
            if content:
                yield content