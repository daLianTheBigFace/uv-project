from collections.abc import Sequence
from typing import Any

from agents.get_time import get_time
from agents.get_weather import get_weather

# Central place to register tools for the main agent.
REGISTERED_TOOLS: list[Any] = [get_weather, get_time]


def get_main_tools() -> Sequence[Any]:
    return REGISTERED_TOOLS

