import argparse
import difflib
import json
import os
import re
import socket
import urllib.parse
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv
from langchain.tools import tool
from pydantic import BaseModel, Field


class QuoteSourceInput(BaseModel):
    line: str = Field(..., description="需要分析来源的台词，例如: 我命由我不由天")


load_dotenv()
OPEN_SUBTITLES_BASE_URL = os.getenv("OPEN_SUBTITLES_BASE_URL", "https://api.opensubtitles.com/api/v1")


_PUNT_PATTERN = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_QUERY_PREFIX_PATTERN = re.compile(r"^(帮我|请|麻烦)?(查一下|查查|查询一下|看看)?(这句)?(台词)?(出处|来源)?[:：]?", flags=re.UNICODE)
_QUERY_SUFFIX_PATTERN = re.compile(r"(的)?(出处|来源)(是)?$", flags=re.UNICODE)


def _normalize_text(text: str) -> str:
    compact = _PUNT_PATTERN.sub("", text).lower()
    return compact


def _clean_query(text: str) -> str:
    cleaned = text.strip().strip("\"'“”‘’")
    cleaned = _QUERY_PREFIX_PATTERN.sub("", cleaned).strip()
    cleaned = cleaned.rstrip("？?。.!！").strip()
    cleaned = _QUERY_SUFFIX_PATTERN.sub("", cleaned).strip()
    return cleaned


def _fetch_json(
    url: str,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, method=method, data=body)
    if headers:
        for key, value in headers.items():
            request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _quote_error(message: str, line: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "message": message,
            "input": line,
        },
        ensure_ascii=False,
    )


def _build_api_headers(api_key: str, user_agent: str, token: str | None = None) -> dict[str, str]:
    headers = {
        "Api-Key": api_key,
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _login_open_subtitles(api_key: str, user_agent: str, username: str, password: str) -> str | None:
    login_url = f"{OPEN_SUBTITLES_BASE_URL}/login"
    headers = _build_api_headers(api_key=api_key, user_agent=user_agent)
    headers["Content-Type"] = "application/json"
    payload = {"username": username, "password": password}
    response = _fetch_json(login_url, headers=headers, method="POST", payload=payload)
    token = response.get("token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def _search_open_subtitles(query: str, api_key: str, user_agent: str, token: str | None, limit: int = 10) -> list[dict[str, Any]]:
    endpoint = f"{OPEN_SUBTITLES_BASE_URL}/subtitles"
    languages = os.getenv("OPEN_SUBTITLES_LANGUAGES", "").strip()
    query_params: dict[str, Any] = {
        "query": query,
        "order_by": "download_count",
        "order_direction": "desc",
        "page": 1,
    }
    if languages:
        query_params["languages"] = languages

    params = urllib.parse.urlencode(
        query_params
    )
    headers = _build_api_headers(api_key=api_key, user_agent=user_agent, token=token)
    data = _fetch_json(f"{endpoint}?{params}", headers=headers)

    rows = data.get("data")
    if not isinstance(rows, list):
        return []

    items: list[dict[str, Any]] = []
    for row in rows[:limit]:
        attributes = row.get("attributes") if isinstance(row, dict) else {}
        if not isinstance(attributes, dict):
            continue

        details = attributes.get("feature_details") if isinstance(attributes.get("feature_details"), dict) else {}
        title = str(details.get("title", "")).strip() or str(attributes.get("release", "")).strip()
        if not title:
            continue

        release = str(attributes.get("release", "")).strip()
        file_name = ""
        files = attributes.get("files")
        if isinstance(files, list) and files:
            first_file = files[0]
            if isinstance(first_file, dict):
                file_name = str(first_file.get("file_name", "")).strip()

        matched_text = release or file_name or title
        subtitle_url = str(attributes.get("url", "")).strip()
        if not subtitle_url:
            sub_id = str(row.get("id", "")).strip()
            subtitle_url = f"https://www.opensubtitles.com/en/subtitles/{sub_id}" if sub_id else ""

        items.append(
            {
                "work_title": title,
                "source_type": f"OpenSubtitles/{details.get('feature_type', 'Unknown')}",
                "character": "",
                "matched_quote": matched_text,
                "language": str(attributes.get("language", "")).strip(),
                "imdb_id": str(details.get("imdb_id", "")).strip(),
                "year": details.get("year"),
                "source_url": subtitle_url,
            }
        )
    return items


def _score_candidate(query_norm: str, alias_norm: str) -> float:
    if not query_norm or not alias_norm:
        return 0.0
    if query_norm == alias_norm:
        return 1.0
    if query_norm in alias_norm or alias_norm in query_norm:
        return 0.9
    return difflib.SequenceMatcher(None, query_norm, alias_norm).ratio()


@tool("get_quote_source", args_schema=QuoteSourceInput)
def get_quote_source(line: str) -> str:
    """根据台词内容推测可能的来源作品。"""
    query = _clean_query(line)
    if not query:
        return _quote_error("台词不能为空", line)

    api_key = os.getenv("OPEN_SUBTITLES_API_KEY", "").strip()
    if not api_key:
        return _quote_error("缺少 OPEN_SUBTITLES_API_KEY 配置", query)

    user_agent = os.getenv("OPEN_SUBTITLES_USER_AGENT", "uv-project-quote-source/0.1").strip()
    username = os.getenv("OPEN_SUBTITLES_USERNAME", "").strip()
    password = os.getenv("OPEN_SUBTITLES_PASSWORD", "").strip()

    token: str | None = None
    if username and password:
        try:
            token = _login_open_subtitles(api_key, user_agent, username, password)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError):
            token = None

    query_norm = _normalize_text(query)
    ranked: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_urls: set[str] = set()
    weak_ranked: list[dict[str, Any]] = []

    try:
        candidates = _search_open_subtitles(query, api_key=api_key, user_agent=user_agent, token=token)
    except urllib.error.HTTPError as exc:
        errors.append(f"HTTP {exc.code}")
        candidates = []
    except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        candidates = []
    for item in candidates:
        source_url = str(item.get("source_url", ""))
        dedupe_key = source_url or str(item.get("work_title", ""))
        if not dedupe_key or dedupe_key in seen_urls:
            continue
        seen_urls.add(dedupe_key)

        title_norm = _normalize_text(str(item.get("work_title", "")))
        snippet_norm = _normalize_text(str(item.get("matched_quote", "")))
        best_score = max(
            _score_candidate(query_norm, title_norm),
            _score_candidate(query_norm, snippet_norm),
        )
        row = {
            "work_title": item.get("work_title", ""),
            "source_type": item.get("source_type", ""),
            "character": item.get("character", ""),
            "matched_quote": item.get("matched_quote", ""),
            "score": round(best_score, 4),
            "language": item.get("language", ""),
            "imdb_id": item.get("imdb_id", ""),
            "year": item.get("year"),
            "source_url": source_url,
        }
        weak_ranked.append(row)
        if best_score >= 0.25:
            ranked.append(row)

    if not ranked and errors:
        return _quote_error(f"OpenSubtitles 查询失败: {errors[0]}", query)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    weak_ranked.sort(key=lambda x: x["score"], reverse=True)
    top_matches = ranked[:3]

    if not top_matches and weak_ranked:
        return json.dumps(
            {
                "status": "weak_match",
                "input": query,
                "message": "已检索到字幕候选，但未形成高置信度匹配。OpenSubtitles 更擅长按影片/字幕元数据检索，不是逐句台词检索。",
                "matches": weak_ranked[:3],
                "disclaimer": "建议补充作品名、角色名或上下文后再查。",
            },
            ensure_ascii=False,
        )

    if not top_matches:
        return json.dumps(
            {
                "status": "no_match",
                "input": query,
                "message": "OpenSubtitles 未找到可用候选，可尝试补充作品名、语言或上下文再查询。",
                "matches": [],
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": "ok",
            "input": query,
            "matches": top_matches,
            "disclaimer": "这是基于 OpenSubtitles 在线检索的启发式匹配，仅供参考。",
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="台词来源分析工具（最小骨架）")
    parser.add_argument("line", help="例如: 我命由我不由天")
    args = parser.parse_args()
    print(get_quote_source.invoke({"line": args.line}))

