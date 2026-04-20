import argparse
import json
import urllib.parse
import urllib.request
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field


JIKAN_BASE_URL = "https://api.jikan.moe/v4"


class AnimeInput(BaseModel):
    query: str = Field(..., description="动漫名称或关键词，例如：进击的巨人")
    limit: int = Field(3, ge=1, le=5, description="返回候选数量，默认 3")


def _fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _anime_error(message: str, query: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "message": message,
            "query": query,
        },
        ensure_ascii=False,
    )


def _extract_names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
    return names


def _search_anime(query: str, limit: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "limit": limit,
            "sfw": "true",
            "order_by": "score",
            "sort": "desc",
        }
    )
    url = f"{JIKAN_BASE_URL}/anime?{params}"
    data = _fetch_json(url)
    rows = data.get("data")
    if not isinstance(rows, list):
        return []

    items: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        aired = row.get("aired") if isinstance(row.get("aired"), dict) else {}
        prop = aired.get("prop") if isinstance(aired.get("prop"), dict) else {}
        from_info = prop.get("from") if isinstance(prop.get("from"), dict) else {}

        season = str(row.get("season", "")).strip()
        year = row.get("year")
        if year is None:
            year = from_info.get("year")

        items.append(
            {
                "mal_id": row.get("mal_id"),
                "title": row.get("title", ""),
                "title_japanese": row.get("title_japanese", ""),
                "type": row.get("type", ""),
                "episodes": row.get("episodes"),
                "status": row.get("status", ""),
                "score": row.get("score"),
                "season": season,
                "year": year,
                "genres": _extract_names(row.get("genres")),
                "synopsis": row.get("synopsis", ""),
                "url": row.get("url", ""),
            }
        )
    return items


@tool("get_anime_info", args_schema=AnimeInput)
def get_anime_info(query: str, limit: int = 3) -> str:
    """根据动漫名称或关键词检索候选作品信息。"""
    text = query.strip()
    if not text:
        return _anime_error("查询关键词不能为空", query)

    try:
        results = _search_anime(text, limit=limit)
    except Exception as exc:
        return _anime_error(f"Jikan 查询失败: {exc}", text)

    if not results:
        return json.dumps(
            {
                "status": "no_match",
                "query": text,
                "message": "没有找到匹配的动漫结果。",
                "matches": [],
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": "ok",
            "query": text,
            "matches": results,
            "source": "Jikan API (MyAnimeList unofficial)",
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动漫信息查询工具（Jikan API）")
    parser.add_argument("query", help="例如: 火影忍者")
    parser.add_argument("--limit", type=int, default=3, help="返回候选数量，默认 3")
    args = parser.parse_args()
    print(get_anime_info.invoke({"query": args.query, "limit": args.limit}))

