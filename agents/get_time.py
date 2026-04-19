from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import tool
from pydantic import BaseModel, Field


class TimeInput(BaseModel):
    timezone: str = Field(
        default="Asia/Shanghai",
        description="IANA 时区名，例如 Asia/Shanghai、UTC、America/New_York",
    )


@tool("get_time", args_schema=TimeInput)
def get_time(timezone: str = "Asia/Shanghai") -> str:
    """根据时区返回当前本地时间。"""
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"无效时区: {timezone}。请使用 IANA 时区名，例如 Asia/Shanghai。"

    now = datetime.now(tz)
    return now.strftime("当前时间(%Z): %Y-%m-%d %H:%M:%S")

