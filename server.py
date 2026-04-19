from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from agents.main_agent import ask_main_agent, stream_main_agent

app= FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: list[dict]


class WeatherRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):
    response = ask_main_agent(request.messages)
    return {"response": response}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    def event_generator():
        try:
            for event in stream_main_agent(request.messages):
                event_name = str(event.get("event", "token"))
                payload = json.dumps(event, ensure_ascii=False)
                yield f"event: {event_name}\ndata: {payload}\n\n"

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


@app.post("/weather/chat")
async def weather_chat(request: WeatherRequest):
    response = ask_main_agent([{"role": "user", "content": request.question}])
    return {"response": response}

