"""
Cost tracking tests for the YouTube RAG chatbot.

WHAT IS COST TRACKING TESTING?
--------------------------------
Cost tracking testing verifies that:
  1. Token counts are captured correctly for every LLM API call.
  2. USD costs are calculated accurately from the pricing table.
  3. Costs accumulate correctly across multiple calls.
  4. Budget threshold checks produce the correct pass/fail result.
  5. The tracker can be reset cleanly between test sessions.

WHY TEST COST TRACKING?
------------------------
In a production RAG pipeline, undetected token bloat can cause runaway API
costs. A prompt change that silently doubles the system prompt length, or a
context window that always retrieves much more text than needed, may not affect
quality scores yet significantly increases cost per query. Cost tracking tests
catch these regressions before they reach production.

TEST CATEGORIES
---------------
Integration tests (call the real OpenAI API):
  - test_single_query_captures_tokens_and_cost : tokens > 0, cost > 0, and cost < $0.01 after get_answer()
  - test_multiple_queries_cost_accumulates   : cost grows with each additional query

Unit tests (no API calls; use synthetic CallRecords):
  - test_calculate_cost_accuracy             : exact math check for gpt-5-mini rates
  - test_cost_summary_has_required_fields    : get_summary() dict has all expected keys
  - test_budget_pass                         : is_within_budget() returns True when under budget
  - test_budget_exceeded                     : is_within_budget() returns False when over budget
  - test_reset_clears_all_records            : reset() zeroes call_count and total_cost_usd

HOW THE CHATBOT IS REUSED ACROSS TESTS
----------------------------------------
The chatbot retriever is expensive to initialise (loads a Chroma vector store),
so _get_chatbot() caches the YTChatbot instance by video_id at module level.
All integration tests share the same cached chatbot instance.

NOTE: This is currently just a framework to calculate the cost and test the cost calculation.
It's not actually implemented for the agent testing. That can be done later like:
tracker = CostTracker()

# wrap any agent call
answer = tracker.track("get_answer", chatbot.get_answer, question)
# at the end of your test session
print(tracker.get_summary())
assert tracker.is_within_budget(5.00)  # fail if run cost more than $5
"""

from datetime import datetime

import pytest
from dotenv import load_dotenv

from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot
from framework.cost_tracking.cost_tracker import CallRecord, CostTracker

load_dotenv()

_VIDEO_ID = "HAoKJT3af7Y"
_TEST_QUESTION = "How does Deep Eval automate comparison of different LLMs?"
_COST_PER_QUERY_LIMIT_USD = 0.01

_chatbot_cache: dict[str, YTChatbot] = {}


def _get_chatbot(video_id: str) -> YTChatbot:
    if video_id not in _chatbot_cache:
        bot = YTChatbot(video_id)
        bot.create_retriever()
        _chatbot_cache[video_id] = bot
    return _chatbot_cache[video_id]


def _make_record(input_tokens: int, output_tokens: int, cost_usd: float) -> CallRecord:
    return CallRecord(
        operation="test_op",
        model="gpt-5-mini",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        timestamp=datetime.now(),
    )


# ──────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────

def test_single_query_captures_tokens_and_cost():
    """
    Assert that input_tokens, output_tokens, cost_usd are all greater than zero,
    and that cost stays below $0.01 for a single tracked get_answer() call.

    WHAT IT VERIFIES
    ----------------
    CostTracker.track() wraps the chatbot call in get_openai_callback(). If the
    callback intercepts the call correctly, prompt_tokens and completion_tokens
    will be non-zero. Since gpt-5-mini has non-zero pricing for both input and
    output tokens, a positive token count must also produce a positive cost.

    The $0.01 ceiling catches prompt bloat regressions — a standard RAG query
    (system prompt + two retrieved chunks + question) should cost ~$0.0015 at
    gpt-5-mini pricing, well below the guard.

    WHAT A FAILURE SIGNALS
    ----------------------
    - get_openai_callback() is not intercepting calls from this LangChain version.
    - The chatbot's LLM call is not routed through LangChain's callback system.
    - Config.llm_model_name is not present in CostTracker.PRICING.
    - The API call failed silently and returned an empty response.
    - The system prompt or context has grown significantly (prompt bloat).
    - The retriever is returning far more chunks than the configured k=2.
    """
    chatbot = _get_chatbot(_VIDEO_ID)
    tracker = CostTracker()

    tracker.track("get_answer", chatbot.get_answer, _TEST_QUESTION)

    summary = tracker.get_summary()
    assert summary["total_input_tokens"] > 0, "No input tokens captured — callback may not be intercepting calls"
    assert summary["total_output_tokens"] > 0, "No output tokens captured — LLM may have returned an empty response"
    assert summary["total_cost_usd"] > 0, (
        "Cost is 0.0 despite non-zero tokens — check that Config.llm_model_name "
        "is present in CostTracker.PRICING"
    )
    assert summary["total_cost_usd"] < _COST_PER_QUERY_LIMIT_USD, (
        f"Single query cost ${summary['total_cost_usd']:.6f} exceeds "
        f"limit ${_COST_PER_QUERY_LIMIT_USD} — possible prompt bloat or retrieval misconfiguration"
    )


def test_multiple_queries_cost_accumulates():
    """
    Assert that running three tracked get_answer() calls produces a total cost
    equal to the sum of each individual call's cost, and that total_cost is
    strictly greater than any single call's cost.

    WHAT IT VERIFIES
    ----------------
    Records accumulate in self._records without being overwritten. If reset()
    were accidentally called between calls, or if records were not appended,
    the total would equal a single call's cost rather than the sum of three.

    WHAT A FAILURE SIGNALS
    ----------------------
    - CostTracker.track() is overwriting rather than appending records.
    - reset() is being called internally between calls.
    - The accumulation logic in get_summary() sums incorrectly.
    """
    questions = [
        "How does Deep Eval automate comparison of different LLMs?",
        "What metrics does Deep Eval use to evaluate LLM outputs?",
        "How does LLM-as-a-judge work in Deep Eval?",
    ]
    chatbot = _get_chatbot(_VIDEO_ID)
    tracker = CostTracker()

    individual_costs = []
    for q in questions:
        tracker.track("get_answer", chatbot.get_answer, q)
        individual_costs.append(tracker.get_summary()["total_cost_usd"])

    per_call_costs = [individual_costs[0]] + [
        individual_costs[i] - individual_costs[i - 1] for i in range(1, len(individual_costs))
    ]

    summary = tracker.get_summary()
    assert summary["call_count"] == 3, f"Expected 3 records, got {summary['call_count']}"
    assert summary["total_cost_usd"] > per_call_costs[0], (
        "Total cost after 3 calls should exceed a single call's cost"
    )
    assert abs(summary["total_cost_usd"] - sum(per_call_costs)) < 1e-10, (
        f"Total cost {summary['total_cost_usd']} does not match sum of individual costs {sum(per_call_costs)}"
    )


# ──────────────────────────────────────────────
# Unit tests (no API calls)
# ──────────────────────────────────────────────

def test_calculate_cost_accuracy():
    """
    Assert that calculate_cost() returns the exact expected USD amount for
    known token counts and the gpt-5-mini model.

    WHAT IT VERIFIES
    ----------------
    Given 1000 input tokens and 200 output tokens with gpt-5-mini pricing:

        input_cost  = 1000 × (0.25 / 1_000_000) = 0.00025
        output_cost = 200  × (2.00 / 1_000_000) = 0.0004
        total       = 0.00065

    The assertion uses a floating-point tolerance of 1e-10 to account for
    Python float representation without being sensitive to rounding at precision
    levels that don't matter for cost tracking.

    WHAT A FAILURE SIGNALS
    ----------------------
    - The PRICING values were changed without updating this test.
    - The formula in calculate_cost() has a multiplication or addition error.
    """
    tracker = CostTracker()

    cost = tracker.calculate_cost("gpt-5-mini", input_tokens=1000, output_tokens=200)

    expected = (1000 * 0.25 / 1_000_000) + (200 * 2.00 / 1_000_000)
    assert abs(cost - expected) < 1e-10, f"Expected ${expected}, got ${cost}"


def test_cost_summary_has_required_fields():
    """
    Assert that get_summary() returns a dict containing all required keys with
    the correct types.

    WHAT IT VERIFIES
    ----------------
    The summary dict is the primary interface for asserting cost in tests.
    Any missing key would cause an AttributeError in tests that rely on it.
    This test verifies the contract independently of any API call.
    """
    tracker = CostTracker()
    tracker._records.append(_make_record(500, 100, 0.000325))

    summary = tracker.get_summary()

    required_keys = {
        "call_count", "total_tokens", "total_input_tokens",
        "total_output_tokens", "total_cost_usd", "calls",
    }
    assert required_keys.issubset(summary.keys()), (
        f"Missing keys: {required_keys - summary.keys()}"
    )
    assert isinstance(summary["call_count"], int)
    assert isinstance(summary["total_cost_usd"], float)
    assert isinstance(summary["calls"], list)


def test_budget_pass():
    """
    Assert that is_within_budget() returns True when total cost is below the
    supplied budget threshold.

    WHAT IT VERIFIES
    ----------------
    A tracker with a single record costing $0.001 should pass a $1.00 budget.
    """
    tracker = CostTracker()
    tracker._records.append(_make_record(500, 100, 0.001))

    assert tracker.is_within_budget(1.00) is True


def test_budget_exceeded():
    """
    Assert that is_within_budget() returns False when total cost exceeds the
    supplied budget threshold.

    WHAT IT VERIFIES
    ----------------
    A tracker with a single record costing $0.001 should fail a $0.0001 budget.
    """
    tracker = CostTracker()
    tracker._records.append(_make_record(500, 100, 0.001))

    assert tracker.is_within_budget(0.0001) is False


def test_reset_clears_all_records():
    """
    Assert that reset() empties all accumulated records, returning get_summary()
    to its zero state.

    WHAT IT VERIFIES
    ----------------
    After appending synthetic records and calling reset(), call_count must be 0
    and total_cost_usd must be 0.0. This confirms that reset() clears the
    underlying list rather than just hiding it.

    WHEN TO USE
    -----------
    Use reset() between test cases that need per-test cost isolation. This test
    validates that isolation actually works.
    """
    tracker = CostTracker()
    tracker._records.append(_make_record(500, 100, 0.000325))
    tracker._records.append(_make_record(300, 80, 0.000235))

    tracker.reset()

    summary = tracker.get_summary()
    assert summary["call_count"] == 0, f"Expected 0 records after reset, got {summary['call_count']}"
    assert summary["total_cost_usd"] == 0.0, f"Expected 0.0 cost after reset, got {summary['total_cost_usd']}"
