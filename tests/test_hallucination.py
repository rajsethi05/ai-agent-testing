"""
Hallucination detection tests for the RAG pipeline.
Metrics in this module check whether the LLM's generated answers are grounded
in the retrieved context, rather than containing invented or unsupported facts.
"""
import json
import os.path
from pathlib import Path

from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
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
            test_cases.append(pytest.param(video_id, entry["input"], entry["expected_output"],
                                           id=f"{video_id}[{i}]"))
    return test_cases[:3]  # to test with limited inputs
    # return test_cases


@pytest.mark.parametrize("video_id,query,expected_output", _load_test_cases())
def test_faithfulness(video_id, query, expected_output):
    """
    Detects hallucinations by checking whether the LLM's answer is grounded in the
    retrieved context.

    WHAT IS FAITHFULNESS?
    ---------------------
    Faithfulness measures the fraction of factual claims in the LLM's actual_output
    that are directly supported by the retrieval_context:

        faithfulness = (claims supported by retrieval_context)
                       / (total claims in actual_output)

    A score of 1.0 means every claim the LLM made can be traced back to a retrieved
    chunk. A score of 0.0 means the LLM's answer has no grounding in the context at all.

    WHY HALLUCINATION HAPPENS IN RAG
    ---------------------------------
    In a RAG pipeline, the LLM only sees what the retriever fetches. If the retriever
    returns irrelevant or incomplete chunks, the LLM has two options: say "I don't know"
    or fill the gap with its parametric knowledge (i.e., hallucinate). Most LLMs default
    to the latter.

    This means hallucination in RAG is often a downstream symptom of poor retrieval.
    That is why this test uses the live retriever — the same chunks the LLM actually
    received — rather than the golden context:
      - It reflects production behaviour end-to-end
      - A faithfulness failure signals a real user-facing problem
      - Cross-referencing with retrieval metrics (test_retrieval.py) reveals the root
        cause: if retrieval metrics are also low, the retriever is responsible; if
        retrieval metrics are healthy but faithfulness is low, the LLM itself is
        hallucinating beyond the provided context

    HOW DEEPEVAL COMPUTES IT
    ------------------------
    FaithfulnessMetric uses LLM-as-a-judge:
      1. Extracts individual factual claims from actual_output
      2. For each claim, checks whether it is entailed by retrieval_context
      3. Returns the fraction of supported claims as the score

    THRESHOLD
    ---------
    0.7 — at least 70% of the LLM's claims must be grounded in the retrieved context.
    This is deliberately stricter than the retrieval thresholds (0.5) because an
    unsupported claim is directly harmful: it reaches the user as a false fact.
    """
    chatbot = _get_chatbot(video_id)
    actual_output = chatbot.get_answer(query)
    retrieval_context = [doc.page_content for doc in chatbot.retriever.invoke(query)]

    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=retrieval_context)
    assert_test(test_case, [FaithfulnessMetric(threshold=0.7)])
