# uv-project Main Agent

A FastAPI project with a LangChain main agent powered by DeepSeek and Open-Meteo.

## Requirements

- Python 3.13+
- `DEEPSEEK_API_KEY` in `.env`

## Run API Server

```bash
cd '/Users/guoxiaojie/Desktop/personal project/uv-project'
uv run uvicorn server:app --reload
```

## Chat Endpoints

- `POST /chat`: non-stream response
- `POST /chat/stream`: SSE stream response
- Both endpoints route to one main agent that can auto-call `get_weather` based on intent.

Example body:

```json
{
  "messages": [
    {"role": "user", "content": "我们周末出门"},
    {"role": "assistant", "content": "好呀，想去哪里？"},
    {"role": "user", "content": "先看看上海今天什么天气"}
  ]
}
```

## Weather Endpoint (optional)

- `POST /weather/chat`
- Kept for compatibility; internally also uses the main agent.

## CLI Quick Test

```bash
cd '/Users/guoxiaojie/Desktop/personal project/uv-project'
uv run python agents/get_weather.py "北京今天天气怎么样？"
```

## Notes

- Weather data source: Open-Meteo (no API key required)
- The agent is implemented in `agents/get_weather.py`

