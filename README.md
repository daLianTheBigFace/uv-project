# uv-project Main Agent

A FastAPI project with a LangChain main agent powered by DeepSeek and Open-Meteo.

## Requirements

- Python 3.13+
- `DEEPSEEK_API_KEY` in `.env`
- `OPENSUBTITLES_API_KEY` in `.env` (for quote source tool)
- Optional for better quota/auth: `OPENSUBTITLES_USERNAME`, `OPENSUBTITLES_PASSWORD`
- Optional: `OPENSUBTITLES_USER_AGENT` (default: `uv-project-quote-source/0.1`)
- Jikan API is used for anime lookup (no API key required)

## Run API Server

```bash
cd '/Users/guoxiaojie/Desktop/personal project/uv-project'
uv run uvicorn server:app --reload
```

## Chat Endpoints

- `POST /chat`: non-stream response
- `POST /chat/stream`: SSE stream response
- `POST /chat/stream/legacy`: SSE stream response via old `StreamAIClient` (no tools)
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
uv run python agents/get_quote_source.py "我命由我不由天"
uv run python agents/get_anime_info.py "进击的巨人"
```

## Notes

- Weather data source: Open-Meteo (no API key required)
- The agent is implemented in `agents/get_weather.py`
- Quote source skeleton tool: `agents/get_quote_source.py`
- Quote source data path: OpenSubtitles API (`/api/v1/subtitles`)
- Anime tool: `agents/get_anime_info.py` (Jikan `/v4/anime`)

