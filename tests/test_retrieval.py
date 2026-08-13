"""
Test file to test retrieval quality.
It loops through all the golden dataset and evaluates the retrieval quality on the below metrics:
- ContextualRecallMetric: checks whether expected answer facts are attributable to the retrieved chunks
"""
from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
import pytest

from config import LATENCY_THRESHOLD_MS
from framework.retrieval.retriever_quality import context_hit_rate, retrieve_with_latency
from framework.utils import get_chatbot, load_test_cases, log_metrics

load_dotenv()


@pytest.mark.parametrize("video_id,query,expected_output,golden_context", load_test_cases(["input", "expected_output", "context"]))
def test_contextual_recall(video_id, query, expected_output, golden_context, request):
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
    chatbot = get_chatbot(video_id)
    actual_output = chatbot.get_answer(query)
    retrieval_context = [doc.page_content for doc in chatbot.retriever.invoke(query)]

    test_case = LLMTestCase(input=query, actual_output=actual_output, expected_output=expected_output,
                            retrieval_context=retrieval_context, )
    request.node.metric_val= log_metrics(context_recall=ContextualRecallMetric.score)

    assert_test(test_case, [ContextualRecallMetric(threshold=0.5)])

@pytest.mark.parametrize("video_id,query,expected_output,golden_context", load_test_cases(["input", "expected_output", "context"]))
def test_contextual_precision(video_id, query, expected_output, golden_context, request):
    """
    Checks whether the retrieved chunks that are relevant to the question are ranked higher than irrelevant ones.
    Aim: catch cases where your retriever returns the right chunks but buries them behind noise. If chunk #1 is
    irrelevant and chunk #2 contains the answer, precision is penalised — your ranking needs improvement.

    Deepeval definition: The contextual precision metric uses LLM-as-a-judge to measure the quality of your RAG
    pipeline's retriever by evaluating whether relevant nodes in your retrieval_context are ranked higher than
    irrelevant ones.
    """
    chatbot = get_chatbot(video_id)
    actual_output = chatbot.get_answer(query)
    retrieval_context = [doc.page_content for doc in chatbot.retriever.invoke(query)]

    test_case = LLMTestCase(input=query, actual_output=actual_output, expected_output=expected_output,
                            retrieval_context=retrieval_context, )
    request.node.metric_val= log_metrics(context_precision=ContextualPrecisionMetric.score)
    assert_test(test_case, [ContextualPrecisionMetric(threshold=0.5)])

@pytest.mark.parametrize("video_id,query,expected_output,golden_context", load_test_cases(["input", "expected_output", "context"]))
def test_contextual_relevancy(video_id, query, expected_output, golden_context, request):
    """
    Checks whether the retrieved chunks are actually relevant to the input query.
    Aim: catch cases where your retriever returns chunks that technically match keywords but don't address the question.
    Unlike recall (which checks against the expected answer), relevancy judges the retrieved context against the
    input alone — so it catches noise even when the answer can still be constructed.

    Deepeval definition: The contextual relevancy metric uses LLM-as-a-judge to measure the quality of your RAG
    pipeline's retriever by evaluating the overall relevance of the information presented in your retrieval_context
    given an input.
    """
    chatbot = get_chatbot(video_id)
    actual_output = chatbot.get_answer(query)
    retrieval_context = [doc.page_content for doc in chatbot.retriever.invoke(query)]

    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=retrieval_context)
    request.node.metric_val= log_metrics(context_relevancy=ContextualRelevancyMetric.score)
    assert_test(test_case, [ContextualRelevancyMetric(threshold=0.5)])

@pytest.mark.parametrize("video_id,query,expected_output,golden_context", load_test_cases(["input", "expected_output", "context"]))
def test_context_hit_rate(video_id, query, expected_output, golden_context, request):
    """
    Deterministic metric that checks whether the live retriever fetches the same source
    chunks that were used to generate the golden Q&A pair.

    WHAT IS CONTEXT HIT RATE?
    -------------------------
    Each golden dataset entry has a `context` field — the exact transcript chunks the
    Synthesizer saw when it created the question and expected answer. These are the
    ground-truth source chunks: what a perfect retriever would return.

    Context Hit Rate measures how much of that ground truth the live retriever actually
    covers:

        hit_rate = (golden chunks matched by at least one retrieved chunk)
                   / (total golden chunks)

    A score of 1.0 means every golden chunk was found. A score of 0.0 means none were.

    HOW MATCHING WORKS
    ------------------
    Exact string equality is not used because chunk sizes differ: the Synthesizer used
    chunk_size=150 tokens while the chatbot uses chunk_size=600 chars. Instead, Jaccard
    token overlap is computed between each pair of (retrieved chunk, golden chunk).
    A pair is considered a match when overlap >= 0.5 (at least half the tokens overlap).

    WHY THIS METRIC MATTERS
    -----------------------
    The three DeepEval contextual metrics (recall, precision, relevancy) use an LLM as
    judge — they are powerful but slow and costly. Context Hit Rate is:
      - Deterministic  : no LLM calls, no API cost
      - Fast           : pure string computation
      - Foundational   : if hit rate is 0, the retriever never found the source material,
                         and no downstream LLM quality can compensate

    It is the cheapest first signal that retrieval is working at all, and complements
    the LLM-judge metrics rather than replacing them.

    THRESHOLD
    ---------
    Asserts hit_rate >= 0.5, meaning at least half the golden chunks must be covered.
    With k=2 chunks retrieved and golden context often containing 2-3 chunks, this is
    a meaningful but achievable bar.
    """
    chatbot = get_chatbot(video_id)
    retriever = chatbot.create_retriever()
    retrieval_context = [doc.page_content for doc in retriever.invoke(query)]

    hit_rate = context_hit_rate(retrieval_context, golden_context)
    request.node.metric_val= log_metrics(context_hit_rate=hit_rate)
    assert hit_rate >= 0.5, f"Context hit rate {hit_rate:.2f} below threshold 0.5 for query: '{query}'"

@pytest.mark.parametrize("video_id,query,expected_output,golden_context", load_test_cases(["input", "expected_output", "context"]))
def test_retrieval_latency(video_id, query, expected_output, golden_context, request):
    """
    Non-functional metric that asserts the retriever responds within an acceptable time.

    WHAT IS MEASURED
    ----------------
    Wall-clock time from the moment retriever.invoke(query) is called to the moment
    results are returned. This window covers two operations:
      1. Query embedding  — HTTP call to OpenAI text-embedding-3-small to convert the
                            query string into a vector
      2. Vector search    — cosine similarity search over the Chroma index on disk
                            to find the top-k most similar chunks (k=2)

    LLM generation (get_answer) is explicitly excluded. Isolating retrieval latency
    makes it possible to detect regressions in the retrieval pipeline independently
    of any changes to the generation step.

    WHY LATENCY MATTERS FOR RAG
    ---------------------------
    In a production RAG system, retrieval is on the critical path of every user request.
    A slow retriever directly adds to end-to-end response time. Tracking latency per
    test case catches:
      - Network degradation to the embedding API
      - Index bloat as more vectors are added over time
      - Regressions introduced by changes to chunk size, overlap, or k

    THRESHOLD
    ---------
    2000ms (LATENCY_THRESHOLD_MS). The embedding API call typically takes 200-500ms
    under normal conditions. 2000ms provides enough headroom for network variance while
    still catching genuine slowdowns. This value should be tightened once a baseline
    is established from real test runs.

    IMPLEMENTATION NOTE
    -------------------
    Uses time.perf_counter() via retrieve_with_latency() for sub-millisecond precision.
    Each test case invokes the retriever independently so latency is measured per query,
    not as a batch average. The chatbot cache (get_chatbot) ensures the Chroma index
    is already loaded before timing starts — cold-start load time is not included.
    """
    chatbot = get_chatbot(video_id)
    retriever = chatbot.create_retriever()
    _, latency_ms = retrieve_with_latency(retriever, query)
    request.node.metric_val= log_metrics(retrieval_latency=latency_ms)
    assert latency_ms < LATENCY_THRESHOLD_MS, (
        f"Retrieval latency {latency_ms:.1f}ms exceeded threshold {LATENCY_THRESHOLD_MS}ms for query: '{query}'")
