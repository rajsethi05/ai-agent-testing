"""
Correctness metric is not the part of actual plan, so we need to think how and where to implement it.
For now, let's park it in this file.

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
    # todo: fix
    #  get_answer() doesn't expose retrieved chunks (yt_chatbot.py:132)**
    #  The retriever runs internally inside the chain but the chunks are never returned — only the final answer string is. For all three DeepEval contextual metrics, you need the actual retrieved chunks as retrieval_context. You'll need to either add a separate method that returns the raw retrieved docs, or modify get_answer() to return both the answer and the chunks.
    actual_output = chatbot.get_answer(question)

    # todo: fix
    #  test_retrieval.py passes golden context instead of real retrieved chunks (tests/test_retrieval.py:38)**
    #  test_case = LLMTestCase(..., retrieval_context=context)  # context = golden dataset field
    #  This is using the pre-generated golden context, not what the retriever actually fetched at test time. The contextual metrics need the actual runtime-retrieved chunks to evaluate retrieval quality meaningfully. Without this fix, you're not testing retrieval at all — you're testing the LLM against a known-good context.
    test_case = LLMTestCase(input=question, actual_output=actual_output, expected_output=expected_output,
                            retrieval_context=context, )
    assert_test(test_case, [Correctness_metric])
