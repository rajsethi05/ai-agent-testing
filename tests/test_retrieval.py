"""
Test file to test retrieval quality.
It loops through all the golden dataset and evaluates the retrieval quality on the below metrics:

"""
import json
import os.path
from pathlib import Path

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
import pytest

from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot
from framework.metrices import Correctness_metric

load_dotenv()

parent_dir = Path(os.path.dirname(__file__)).parent
goldens_dir = os.path.join(parent_dir, "framework/golden_dataset/datasets")

def _load_test_cases():
    test_cases = []
    for json_file in sorted(Path(goldens_dir).glob("*.json")):
        video_id = json_file.stem
        entries = json.loads(json_file.read_text())
        for i, entry in enumerate(entries):
            test_cases.append(pytest.param(video_id, entry["input"], entry["expected_output"], entry["context"],
                id=f"{video_id}[{i}]", ))
    return test_cases

@pytest.mark.parametrize("video_id,question,expected_output,context", _load_test_cases())
def test_correctness(video_id, question, expected_output, context):
    chatbot = YTChatbot(video_id)
    chatbot.create_retriever()
    actual_output = chatbot.get_answer(question)

    test_case = LLMTestCase(input=question, actual_output=actual_output, expected_output=expected_output,
        retrieval_context=context, )
    assert_test(test_case, [Correctness_metric])
