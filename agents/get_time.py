from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import tool
from pydantic import BaseModel, Field


class TimeInput(BaseModel):
    timezone: str = Field(
        default="Asia/Shanghai",
        description="IANA 时区名，例如 Asia/Shanghai、UTC、America/New_York",
    )


TIMEZONE_ALIASES = {
    "beijing": "Asia/Shanghai",
    "北京时间": "Asia/Shanghai",
    "中国时间": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "utc+8": "Asia/Shanghai",
}


def _normalize_timezone(raw_timezone: str) -> str:
    timezone = raw_timezone.strip()
    if not timezone:
        return "Asia/Shanghai"
    return TIMEZONE_ALIASES.get(timezone.lower(), timezone)


def _is_tz_database_missing() -> bool:
    try:
        ZoneInfo("UTC")
        return False
    except ZoneInfoNotFoundError:
        return True


@tool("get_time", args_schema=TimeInput)
def get_time(timezone: str = "Asia/Shanghai") -> str:
    """根据时区返回当前本地时间。"""
    timezone = _normalize_timezone(timezone)
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        if _is_tz_database_missing():
            return (
                "当前运行环境缺少时区数据库，无法解析时区。"
                "请先安装 tzdata（例如执行: uv add tzdata）。"
            )
        return f"无效时区: {timezone}。请使用 IANA 时区名，例如 Asia/Shanghai。"

    now = datetime.now(tz)
    return now.strftime("当前时间(%Z): %Y-%m-%d %H:%M:%S")

