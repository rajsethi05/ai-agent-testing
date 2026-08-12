import json

import pytest

from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot
from config import GOLDENS_DIR
import test_run_config

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
        if test_run_config.VIDEO_IDS is not None and video_id not in test_run_config.VIDEO_IDS:
            continue
        entries = json.loads(json_file.read_text())
        for i, entry in enumerate(entries):
            params = [video_id] + [entry[f] for f in fields]
            test_cases.append(pytest.param(*params, id=f"{video_id}[{i}]"))
    if test_run_config.MAX_TEST_CASES is not None:
        test_cases = test_cases[:test_run_config.MAX_TEST_CASES]
    return test_cases


def slice_patterns(patterns: list) -> list:
    if test_run_config.MAX_INJECTION_PATTERNS is not None:
        return patterns[:test_run_config.MAX_INJECTION_PATTERNS]
    return patterns


def log_metrics(**kwargs) -> str:
    metrics = ""
    for metric, value in kwargs.items():
        if type(value) is float:
            value = round(value, 5)
        metrics = metrics + f' {metric}={value},'
    return metrics
