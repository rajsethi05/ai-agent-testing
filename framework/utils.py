from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot

_chatbot_cache: dict[str, YTChatbot] = {}


def get_chatbot(video_id: str) -> YTChatbot:
    if video_id not in _chatbot_cache:
        bot = YTChatbot(video_id)
        bot.create_retriever()
        _chatbot_cache[video_id] = bot
    return _chatbot_cache[video_id]


def log_metrics(**kwargs) -> str:
    metrics = ""
    for metric, value in kwargs.items():
        if type(value) is float:
            value = round(value, 5)
        metrics = metrics + f' {metric}={value},'
    return metrics
