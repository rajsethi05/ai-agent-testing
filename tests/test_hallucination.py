"""
Hallucination detection tests for the RAG pipeline.
Metrics in this module check whether the LLM's generated answers are grounded
in the retrieved context, rather than containing invented or unsupported facts.
"""
import json
import os.path
from pathlib import Path

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
import pytest

from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot
from framework.metrices import Consistency_metric

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
            test_cases.append(pytest.param(video_id, entry["input"], entry["expected_output"], id=f"{video_id}[{i}]"))
    return test_cases[:3]  # to test with limited inputs  # return test_cases

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

groundedness_metric = GEval(name="Groundedness", evaluation_steps=[
    "Check if the answer introduces any information, conclusions, or recommendations that are not explicitly present in the retrieval context.",
    "Penalise answers that extrapolate or draw inferences beyond what the context directly states.",
    "Penalise answers that use generalisations such as 'generally', 'typically', 'experts say', or 'studies show' when the context does not support such claims.",
    "Reward answers that accurately reflect the scope and limitations of the provided context, including saying 'I don't know' when the context is insufficient.", ],
                            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                               LLMTestCaseParams.RETRIEVAL_CONTEXT], )

@pytest.mark.parametrize("video_id,query,expected_output", _load_test_cases())
def test_answer_relevancy(video_id, query, expected_output):
    """
    Checks whether the LLM's answer actually addresses the question that was asked.

    WHAT IS ANSWER RELEVANCY?
    -------------------------
    Answer relevancy measures the proportion of statements in the actual_output that
    are directly relevant to the input query:

        answer_relevancy = (statements relevant to the input)
                           / (total statements in actual_output)

    A score of 1.0 means every part of the answer addresses the question.
    A score of 0.0 means the answer is entirely off-topic.

    HOW IT DIFFERS FROM FAITHFULNESS AND GROUNDEDNESS
    --------------------------------------------------
    Faithfulness and groundedness both check the relationship between the answer and
    the retrieved context. Answer relevancy checks the relationship between the answer
    and the input query — an orthogonal dimension:

      Faithfulness   — Does the answer stick to the context? (answer vs context)
      Groundedness   — Does the answer stay within context boundaries? (answer vs context)
      AnswerRelevancy — Does the answer address the question? (answer vs input)

    An answer can score perfectly on faithfulness and groundedness while still failing
    relevancy. For example:
      Query: "What is shrinkflation?"
      Context: contains information about both shrinkflation and inflation.
      LLM answers: "Inflation is the general rise in price levels across an economy."
      → Faithful (supported by context), Grounded (no extrapolation), but NOT relevant
        to the specific question asked.

    WHY THIS MATTERS IN RAG
    -----------------------
    RAG pipelines can retrieve chunks that contain the answer but also a lot of surrounding
    content. If the LLM latches onto the surrounding content instead of the answer, the
    user gets a response that is technically accurate but unhelpful. Answer relevancy
    catches this failure mode.

    HOW DEEPEVAL COMPUTES IT
    ------------------------
    AnswerRelevancyMetric uses LLM-as-a-judge:
      1. Extracts individual statements from actual_output
      2. For each statement, judges whether it is relevant to the input query
      3. Returns the fraction of relevant statements as the score

    THRESHOLD
    ---------
    0.7 — at least 70% of the answer's statements must address the question. This
    allows for brief contextual framing while penalising answers that substantially
    drift from the query.
    """
    chatbot = _get_chatbot(video_id)
    actual_output = chatbot.get_answer(query)
    retrieval_context = [doc.page_content for doc in chatbot.retriever.invoke(query)]

    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=retrieval_context)
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])

@pytest.mark.parametrize("video_id,query,expected_output", _load_test_cases())
def test_consistency(video_id, query, expected_output):
    """
    Detects hallucination instability by checking whether the LLM gives consistent
    answers when asked the same question twice.

    WHAT IS CONSISTENCY?
    --------------------
    Consistency measures whether two independently generated answers to the same query
    agree on the facts. If they contradict each other, at least one answer contains
    a hallucination — a model that is genuinely grounded in the retrieved context
    should produce stable, non-contradictory answers across runs.

        consistency = absence of factual contradictions between answer_1 and answer_2

    WHY INCONSISTENCY SIGNALS HALLUCINATION
    ----------------------------------------
    When the LLM answers from the retrieved context, the context acts as an anchor —
    both answers should reflect the same source material and therefore agree. When the
    LLM generates from parametric memory (hallucination), its outputs are stochastic
    and can vary between runs, producing contradictions.

    Inconsistency is therefore a hallucination signal even when we cannot verify which
    answer is correct — we know that at minimum one is wrong.

    HOW IT DIFFERS FROM DETERMINISTIC TESTING (TODO #3)
    ----------------------------------------------------
    Consistency and determinism are related but distinct:

      Consistency (this metric) — Do two answers agree on the facts? Tolerates
                                   differences in phrasing or detail level.
      Determinism (TODO #3)     — Is the exact output character-for-character identical
                                   at temperature=0? A stricter, reproducibility-focused check.

    HOW THE TEST WORKS
    ------------------
    get_answer() is called twice for the same query. The two responses are placed in
    actual_output and expected_output of LLMTestCase — repurposing expected_output as
    a second answer rather than a ground-truth label. The GEval consistency_metric then
    compares the two for factual contradictions.

    THRESHOLD
    ---------
    0.7 — consistent with other hallucination metrics. Some minor variation between
    runs is acceptable; the metric penalises factual contradictions, not stylistic ones.
    """
    chatbot = _get_chatbot(video_id)
    answer_1 = chatbot.get_answer(query)
    answer_2 = chatbot.get_answer(query)

    test_case = LLMTestCase(input=query, actual_output=answer_1, expected_output=answer_2)
    assert_test(test_case, [Consistency_metric])
