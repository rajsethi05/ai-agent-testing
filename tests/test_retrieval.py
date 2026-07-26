"""
Test file to test retrieval quality.
It loops through all the golden dataset and evaluates the retrieval quality on the below metrics:
- ContextualRecallMetric: checks whether expected answer facts are attributable to the retrieved chunks
"""
import json
import os.path
from pathlib import Path

from deepeval import assert_test
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
import pytest

from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot

load_dotenv()

parent_dir = Path(os.path.dirname(__file__)).parent
goldens_dir = os.path.join(parent_dir, "framework/golden_dataset/datasets")

_chatbot_cache = {}

def _get_chatbot(video_id):
    if video_id not in _chatbot_cache:
        _chatbot_cache[video_id] = YTChatbot(video_id)
    return _chatbot_cache[video_id]

def _load_test_cases():
    test_cases = []
    for json_file in sorted(Path(goldens_dir).glob("*.json")):
        video_id = json_file.stem
        entries = json.loads(json_file.read_text())
        for i, entry in enumerate(entries):
            test_cases.append(pytest.param(video_id, entry["input"], entry["expected_output"], entry["context"],
                                           id=f"{video_id}[{i}]", ))
    return test_cases[:3]  # to test with limited inputs
    # return test_cases

@pytest.mark.parametrize("video_id,query,expected_output,golden_context", _load_test_cases())
def test_contextual_recall(video_id, query, expected_output, golden_context):
    """
    Checks whether the retrieved context contains all the information needed to fully answer the question.
    Aim: catch cases where your retriever missed relevant chunks. If the ground-truth answer has 5 facts and your
    context only supports 3, recall is low — your retrieval pipeline is under-fetching.

    Deepeval definition: The contextual recall metric uses LLM-as-a-judge to measure the quality of your RAG pipeline's
    retriever by evaluating the extent of which the retrieval_context aligns with the expected_output

    :param video_id:
    :param query:
    :param expected_output:
    :param golden_context:
    :return:
    """
    chatbot = _get_chatbot(video_id)
    actual_output = chatbot.get_answer(query)
    retrieval_context = [doc.page_content for doc in chatbot.retriever.invoke(query)]

    test_case = LLMTestCase(input=query, actual_output=actual_output, expected_output=expected_output,
        retrieval_context=retrieval_context, )
    assert_test(test_case, [ContextualRecallMetric(threshold=0.5)])


@pytest.mark.parametrize("video_id,query,expected_output,golden_context", _load_test_cases())
def test_contextual_precision(video_id, query, expected_output, golden_context):
    """
    Checks whether the retrieved chunks that are relevant to the question are ranked higher than irrelevant ones.
    Aim: catch cases where your retriever returns the right chunks but buries them behind noise. If chunk #1 is
    irrelevant and chunk #2 contains the answer, precision is penalised — your ranking needs improvement.

    Deepeval definition: The contextual precision metric uses LLM-as-a-judge to measure the quality of your RAG
    pipeline's retriever by evaluating whether relevant nodes in your retrieval_context are ranked higher than
    irrelevant ones.
    """
    chatbot = _get_chatbot(video_id)
    actual_output = chatbot.get_answer(query)
    retrieval_context = [doc.page_content for doc in chatbot.retriever.invoke(query)]

    test_case = LLMTestCase(input=query, actual_output=actual_output, expected_output=expected_output,
        retrieval_context=retrieval_context, )
    assert_test(test_case, [ContextualPrecisionMetric(threshold=0.5)])


@pytest.mark.parametrize("video_id,query,expected_output,golden_context", _load_test_cases())
def test_contextual_relevancy(video_id, query, expected_output, golden_context):
    """
    Checks whether the retrieved chunks are actually relevant to the input query.
    Aim: catch cases where your retriever returns chunks that technically match keywords but don't address the question.
    Unlike recall (which checks against the expected answer), relevancy judges the retrieved context against the
    input alone — so it catches noise even when the answer can still be constructed.

    Deepeval definition: The contextual relevancy metric uses LLM-as-a-judge to measure the quality of your RAG
    pipeline's retriever by evaluating the overall relevance of the information presented in your retrieval_context
    given an input.
    """
    chatbot = _get_chatbot(video_id)
    actual_output = chatbot.get_answer(query)
    retrieval_context = [doc.page_content for doc in chatbot.retriever.invoke(query)]

    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=retrieval_context)
    assert_test(test_case, [ContextualRelevancyMetric(threshold=0.5)])
