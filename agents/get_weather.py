import argparse
import json
import urllib.parse
import urllib.request
from typing import Any, Iterator, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

load_dotenv()


class WeatherInput(BaseModel):
	city: str = Field(..., description="需要查询天气的城市名，例如: 上海")


def _fetch_json(url: str) -> dict[str, Any]:
	with urllib.request.urlopen(url, timeout=10) as response:
		return json.loads(response.read().decode("utf-8"))


def _weather_code_to_text(code: int) -> str:
	mapping = {
		0: "晴朗",
		1: "大体晴",
		2: "局部多云",
		3: "阴",
		45: "雾",
		48: "雾凇",
		51: "小毛毛雨",
		53: "毛毛雨",
		55: "强毛毛雨",
		61: "小雨",
		63: "中雨",
		65: "大雨",
		71: "小雪",
		73: "中雪",
		75: "大雪",
		80: "阵雨",
		81: "较强阵雨",
		82: "强阵雨",
		95: "雷雨",
	}
	return mapping.get(code, f"未知天气代码({code})")


@tool("get_weather", args_schema=WeatherInput)
def get_weather(city: str) -> str:
	"""根据城市名称查询实时天气信息。"""
	encoded_city = urllib.parse.quote(city.strip())
	geo_url = (
		"https://geocoding-api.open-meteo.com/v1/search?"
		f"name={encoded_city}&count=1&language=zh&format=json"
	)
	geo_data = _fetch_json(geo_url)
	results = geo_data.get("results") or []
	if not results:
		return f"没有找到城市: {city}"

	location = results[0]
	latitude = location["latitude"]
	longitude = location["longitude"]
	city_name = location.get("name", city)
	country = location.get("country", "")

	weather_url = (
		"https://api.open-meteo.com/v1/forecast?"
		f"latitude={latitude}&longitude={longitude}"
		"&current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m"
		"&timezone=auto"
	)
	weather_data = _fetch_json(weather_url)
	current = weather_data.get("current") or {}
	if not current:
		return f"获取天气失败: {city_name}"

	weather_desc = _weather_code_to_text(int(current.get("weather_code", -1)))
	return (
		f"{city_name} {country} 当前天气: {weather_desc}; "
		f"温度 {current.get('temperature_2m')}°C; "
		f"体感温度 {current.get('apparent_temperature')}°C; "
		f"湿度 {current.get('relative_humidity_2m')}%; "
		f"风速 {current.get('wind_speed_10m')} km/h"
	)


def build_weather_agent(model_name: str = "deepseek-chat"):
	system_prompt = (
		"你是一个天气助手。"
		"当用户询问天气时，必须优先调用 get_weather 工具后再回答。"
		"回答保持简洁、中文输出。"
        "可以回答的很有诗意，有时会让用户惊喜，但是也不要太夸张"
	)
	llm = ChatDeepSeek(model=model_name, temperature=0)
	return create_agent(model=llm, tools=[get_weather], system_prompt=system_prompt)


def _extract_text(agent_result: dict[str, Any]) -> str:
	messages = agent_result.get("messages") or []
	if not messages:
		return "未获得模型回复"

	content = getattr(messages[-1], "content", "")
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts: list[str] = []
		for item in content:
			if isinstance(item, dict) and "text" in item:
				parts.append(str(item["text"]))
			else:
				parts.append(str(item))
		return "".join(parts)
	return str(content)


def ask_weather(question: str, model_name: str = "deepseek-chat") -> str:
	agent = build_weather_agent(model_name=model_name)
	payload = cast(Any, {"messages": [{"role": "user", "content": question}]})
	result = agent.invoke(payload)
	return _extract_text(result)


def stream_weather(question: str, model_name: str = "deepseek-chat") -> Iterator[str]:
	agent = build_weather_agent(model_name=model_name)
	payload = cast(Any, {"messages": [{"role": "user", "content": question}]})
	saw_tool_step = False
	for event in agent.stream(payload, stream_mode="messages"):
		if not isinstance(event, tuple) or not event:
			continue
		message_chunk = event[0]
		meta = event[1] if len(event) > 1 and isinstance(event[1], dict) else {}
		node = str(meta.get("langgraph_node", ""))

		# Tool node usually returns a full weather string in one chunk; skip it to keep UX smooth.
		if node == "tools":
			saw_tool_step = True
			continue

		# Skip pre-tool assistant preamble like "我来帮你查询...".
		if node != "model" or not saw_tool_step:
			continue

		# Skip tool-call protocol chunks, keep only user-visible text tokens.
		if getattr(message_chunk, "tool_call_chunks", None):
			continue

		content = getattr(message_chunk, "content", "")
		if isinstance(content, str) and content:
			yield content


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="天气查询 Agent")
	parser.add_argument("question", help="例如: 北京今天天气怎么样")
	parser.add_argument("--model", default="deepseek-chat", help="DeepSeek 模型名")
	args = parser.parse_args()
	print(ask_weather(args.question, model_name=args.model))

