# Project Skills Guide

This document captures how to design, implement, and evolve this repo safely.

## 1) Architecture Skills

- Keep `agents/main_agent.py` as the single orchestration entry.
- Keep domain logic in tools under `agents/` and register through `agents/tool_registry.py`.
- Prefer this split:
  - `tool`: deterministic data fetch/transform
  - `agent`: intent routing + natural language summarization
- For streaming UX, emit structured events (`status`, `token`, `done`, `error`) from server endpoints.

## 2) Tool Authoring Skills

When adding a tool:

1. Define `args_schema` with Pydantic (`BaseModel`, `Field`) and clear Chinese descriptions.
2. Return JSON text with a stable shape:
   - success: `{"status":"ok", ...}`
   - no result: `{"status":"no_match", ...}`
   - weak result: `{"status":"weak_match", ...}`
   - failure: `{"status":"error", ...}`
3. Add explicit network error handling (`HTTPError`, `URLError`, timeout, decode errors).
4. Add a CLI entry for quick local checks.
5. Register it in `agents/tool_registry.py`.

Suggested output contract:

```json
{
  "status": "ok|no_match|weak_match|error",
  "message": "optional human-readable message",
  "matches": []
}
```

## 3) Prompt Skills

In `main_agent.py` system prompt:

- Tell the model when to call each tool.
- Tell the model not to invent missing fields.
- Tell the model to summarize tool output naturally, not field-by-field parroting.
- Keep prompt additive and explicit when adding new tools.

## 4) Reliability Skills

- Load env once per module before reading env vars.
- Validate required keys at runtime entry points (for example `DEEPSEEK_API_KEY`).
- Keep tool functions tolerant of malformed upstream responses.
- Never depend on a single third-party endpoint without fallback messaging.

## 5) Streaming Skills

`stream_main_agent()` convention:

- Tool phase emits one short status text.
- Model phase emits only user-visible text tokens.
- Suppress protocol/tool-call chunks.

Use these event shapes from server SSE:

```json
[
  {"event":"status","type":"status","content":"..."},
  {"event":"token","type":"token","content":"..."},
  {"event":"done","type":"done","content":"[DONE]"},
  {"event":"error","type":"error","error":"..."}
]
```

## 6) Quote Retrieval Skills (Current + Next)

Current tool (`get_quote_source`) is metadata-heuristic retrieval. Keep user-facing disclaimer.

For future improvements:

- Normalize/clean user quote text before search.
- Add optional hints (`work_hint`, `character_hint`, `language`) in schema when ready.
- Introduce retrieval + rerank architecture for multilingual quote lookup.
- Add a feedback loop for wrong matches (`correct/incorrect`) to build eval sets.

## 7) Review Checklist (Use Before Merge)

- [ ] Tool input schema clear and strict
- [ ] Output JSON status contract preserved
- [ ] Network error paths handled and tested
- [ ] Added/updated prompt instructions if tool behavior changed
- [ ] Streaming behavior still emits valid SSE events
- [ ] README env names and usage examples are up to date
- [ ] Quick smoke run completed

## 8) Quick Smoke Commands

```bash
cd '/Users/guoxiaojie/Desktop/personal project/uv-project'
uv run python agents/get_time.py
uv run python agents/get_weather.py "上海今天天气"
uv run python agents/get_anime_info.py "进击的巨人"
uv run uvicorn server:app --reload
```

