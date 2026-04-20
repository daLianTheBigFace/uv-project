from collections.abc import Sequence
from typing import Any

from agents.get_anime_info import get_anime_info
from agents.get_quote_source import get_quote_source
from agents.get_time import get_time
from agents.get_weather import get_weather

# Central place to register tools for the main agent.
REGISTERED_TOOLS: list[Any] = [get_weather, get_time, get_quote_source, get_anime_info]


def get_main_tools() -> Sequence[Any]:
    return REGISTERED_TOOLS

