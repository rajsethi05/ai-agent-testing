"""
Deterministic behaviour tests for the YouTube RAG chatbot.

WHAT IS DETERMINISTIC TESTING?
-------------------------------
Deterministic testing verifies that the agent produces identical (or near-identical)
outputs when given the same input under controlled conditions — specifically,
temperature=0 (greedy decoding). It is the strictest form of reproducibility check.

  Consistency (test_hallucination.py) — Do two answers agree on the facts?
  Determinism (this module)           — Is the output identical character-for-character?

WHY temperature=0?
------------------
At temperature=0, the LLM always selects the highest-probability next token
(greedy decoding), which should — in theory — produce the same sequence every
time for the same input. Any deviation is a signal of:
  - Server-side batching variance (OpenAI may process requests differently)
  - Retriever non-determinism (documents returned in a different order)
  - Model version drift (silent weight updates between API calls)

HOW THE CHAIN IS BUILT
-----------------------
Rather than modifying YTChatbot.get_answer(), this test builds a parallel chain
that reuses the chatbot's existing retriever and Config.prompt but substitutes
a temperature=0 LLM. This follows the same pattern as test_prompt_injection.py
and avoids changing agent production code.

METRICS USED
------------
All three metrics are defined in framework/deterministic/checker.py:

  ExactMatchRate         — Fraction of output pairs that are character-for-character
                           identical across N=3 runs. Threshold: 1.0 (all must match).
                           Pure Python assertion, no LLM-as-judge.

  OutputLengthCVMetric   — Coefficient of Variation of response lengths across N=3
                           runs. Threshold: CV < 0.05 (less than 5% length variance).
                           Pure Python assertion, no LLM-as-judge.

  SemanticEquivalenceMetric — GEval (LLM-as-judge). Checks that two temperature=0
                           answers carry exactly the same information at the same
                           level of detail. Stricter than Consistency_metric because
                           it also penalises informational asymmetry (one answer says
                           more or less than the other). Threshold: 0.8.
"""

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
import pytest

from agents.rag_youtube_chatbot.yt_chatbot import Config, YTChatbot, parser
from config import LENGTH_CV_THRESHOLD, N_RUNS
from framework.deterministic.checker import (SemanticEquivalenceMetric, compute_exact_match_rate, compute_length_cv, )
from framework.utils import get_chatbot, load_test_cases, log_metrics

load_dotenv()

_llm_t0 = ChatOpenAI(model=Config.llm_model_name, temperature=0)

def _get_answer_t0(chatbot: YTChatbot, question: str) -> str:
    """Run the chatbot's RAG chain at temperature=0, reusing its retriever and prompt."""
    chain = (RunnableParallel({"context": chatbot.retriever | RunnableLambda(chatbot.format_context),
                               "question": RunnablePassthrough(), }) | Config.prompt | _llm_t0 | parser)
    return chain.invoke(question)


@pytest.mark.parametrize("video_id,query", load_test_cases(["input"]))
def test_exact_match(video_id, query, request):
    """
    Asserts that N=3 calls at temperature=0 produce character-for-character identical outputs.

    METRIC: ExactMatchRate (pure Python, no LLM-as-judge)
    ------------------------------------------------------
    Runs the temperature=0 RAG chain three times for the same query and checks
    that all three outputs are exactly equal as Python strings.

    A rate of 1.0 means all C(3,2)=3 pairs matched — perfect determinism.
    Any rate below 1.0 is reported along with the differing outputs to aid debugging.

    WHAT A FAILURE SIGNALS
    ----------------------
    - Retriever non-determinism: the vector store returned documents in a different
      order on one call, changing the assembled context and therefore the LLM output.
    - OpenAI server-side variance: despite temperature=0, the API occasionally
      introduces minor token-level differences due to server-side batching.
    - Model version drift: the underlying model weights were silently updated.
    """
    chatbot = get_chatbot(video_id)
    outputs = [_get_answer_t0(chatbot, query) for _ in range(N_RUNS)]
    rate = compute_exact_match_rate(outputs)
    request.node.metric_val = log_metrics(match_rate=rate)
    assert rate == 1.0, (f"ExactMatchRate={rate:.2f} — not all {N_RUNS} outputs were identical.\n" + "\n".join(
        f"Run {i}: {repr(o)}" for i, o in enumerate(outputs)))

@pytest.mark.parametrize("video_id,query", load_test_cases(["input"]))
def test_output_length_stability(video_id, query, request):
    """
    Asserts that N=3 temperature=0 outputs have less than 5% variation in length.

    METRIC: OutputLengthCVMetric (pure Python, no LLM-as-judge)
    ------------------------------------------------------------
    Computes the Coefficient of Variation (CV = stdev / mean) of response
    character-lengths across three runs. A CV below 0.05 means the model produces
    structurally stable responses at temperature=0.

    THRESHOLD: CV < 0.05

    WHY LENGTH COMPLEMENTS EXACT MATCH
    -----------------------------------
    Exact match is a binary check. When it fails, the length CV tells you
    how different the outputs are structurally:

      Low CV + exact match fails  → same structure, minor wording difference
                                    (e.g., synonym swap, extra comma)
      High CV + exact match fails → qualitatively different responses
                                    (e.g., two sentences vs. two paragraphs)

    The second case is the more serious failure and warrants deeper investigation
    into retriever ordering or model version drift.
    """
    chatbot = get_chatbot(video_id)
    outputs = [_get_answer_t0(chatbot, query) for _ in range(N_RUNS)]
    cv = compute_length_cv(outputs)
    request.node.metric_val = log_metrics(length_cv=cv)
    assert cv < LENGTH_CV_THRESHOLD, (f"OutputLengthCV={cv:.4f} exceeds threshold {LENGTH_CV_THRESHOLD}.\n"
                                      f"Lengths: {[len(o) for o in outputs]}")

@pytest.mark.parametrize("video_id,query", load_test_cases(["input"]))
def test_semantic_equivalence(video_id, query, request):
    """
    Asserts that two temperature=0 answers are informationally identical (GEval).

    METRIC: SemanticEquivalenceMetric (GEval, LLM-as-judge)
    --------------------------------------------------------
    Calls the temperature=0 chain twice and places the two answers in
    actual_output and expected_output of LLMTestCase. The GEval judge then
    checks whether the two convey exactly the same information — same facts,
    same scope, same level of detail.

    This is stricter than Consistency_metric (framework/metrices.py), which only
    penalises direct factual contradictions. SemanticEquivalence also penalises
    informational asymmetry: if one answer says more or less than the other, the
    score is reduced even if there is no direct contradiction.

    THRESHOLD: 0.8 (higher than other GEval metrics in this project at 0.7)

    DEEPEVAL DEFINITION: GEval
    --------------------------
    GEval is DeepEval's general-purpose LLM-as-a-judge metric. It sends the
    evaluation_steps to a judge LLM as a scoring rubric and normalises the result
    to [0.0, 1.0]. Reference: https://docs.confident-ai.com/docs/metrics-llm-evals

    WHAT A FAILURE SIGNALS
    ----------------------
    Even when exact match passes (identical strings), this test can fail if a
    future model update changes the output in a semantically meaningful way.
    More commonly, if exact match fails and the CV is also elevated, this metric
    confirms whether the informational content actually diverged or just the wording.
    """
    chatbot = get_chatbot(video_id)
    answer_1 = _get_answer_t0(chatbot, query)
    answer_2 = _get_answer_t0(chatbot, query)

    test_case = LLMTestCase(input=query, actual_output=answer_1, expected_output=answer_2, )
    request.node.metric_val = log_metrics(semantic_score=SemanticEquivalenceMetric.score)
    assert_test(test_case, [SemanticEquivalenceMetric])
