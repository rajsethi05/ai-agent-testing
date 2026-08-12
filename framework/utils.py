import json

import pytest

from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot
from config import GOLDENS_DIR

_chatbot_cache: dict[str, YTChatbot] = {}


def get_chatbot(video_id: str) -> YTChatbot:
    if video_id not in _chatbot_cache:
        bot = YTChatbot(video_id)
        bot.create_retriever()
        _chatbot_cache[video_id] = bot
    return _chatbot_cache[video_id]


def load_test_cases(fields: list[str]) -> list:
    test_cases = []
    for json_file in sorted(GOLDENS_DIR.glob("*.json")):
        video_id = json_file.stem
        entries = json.loads(json_file.read_text())
        for i, entry in enumerate(entries):
            params = [video_id] + [entry[f] for f in fields]
            test_cases.append(pytest.param(*params, id=f"{video_id}[{i}]"))
    return test_cases [:1]


def log_metrics(**kwargs) -> str:
    metrics = ""
    for metric, value in kwargs.items():
        if type(value) is float:
            value = round(value, 5)
        metrics = metrics + f' {metric}={value},'
    return metrics
