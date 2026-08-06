"""
Cost tracking utilities for LLM API calls in the YouTube RAG testing framework.

OVERVIEW
--------
Cost tracking measures the token consumption and USD cost of every LLM API call
made by the agent under test. In a RAG pipeline, costs accumulate across two
dimensions:

  1. LLM calls  — the language model that generates answers (gpt-5-mini).
  2. Embedding  — the embedding model that encodes chunks and queries
                  (text-embedding-3-small). These only occur when a vector store
                  is first built; subsequent runs hit the cached store.

WHY COST TRACKING MATTERS
--------------------------
- Budget enforcement: uncontrolled token usage can cause unexpected API bills,
  especially during test runs that fire many queries in a loop.
- Regression detection: a prompt change that silently doubles input tokens is a
  regression even if answer quality stays the same.
- Model comparison: when evaluating whether to upgrade from gpt-5-mini to a more
  capable model, cost-per-query is a key decision input alongside quality metrics.
- Audit trail: recording cost per operation (get_answer vs clean_transcript) lets
  you see which part of the pipeline is the cost driver.

HOW CAPTURE WORKS
-----------------
LangChain's get_openai_callback() is a context manager that intercepts all
OpenAI API calls made through LangChain within its scope and accumulates:
  - cb.prompt_tokens     — tokens in the request (system + user + context)
  - cb.completion_tokens — tokens in the response

Cost is then calculated from CostTracker.PRICING using those token counts.
LangChain's own cb.total_cost is intentionally not used because its internal
pricing table may not include newly released models (e.g. gpt-5-mini).

CLASSES IN THIS MODULE
-----------------------
  CallRecord  — dataclass storing metadata for one tracked API call
  CostTracker — accumulates CallRecords and provides summary and budget checks
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

from langchain_community.callbacks import get_openai_callback

from agents.rag_youtube_chatbot.yt_chatbot import Config


@dataclass
class CallRecord:
    """
    Stores metadata for a single tracked LLM API call.

    FIELDS
    ------
    operation      : str      — human-readable label for the call, e.g. "get_answer".
                                Passed by the caller; not inferred automatically.
    model          : str      — the model name as it appears in PRICING, e.g. "gpt-5-mini".
    input_tokens   : int      — prompt tokens consumed (system prompt + context + question).
    output_tokens  : int      — completion tokens generated (the answer text).
    cost_usd       : float    — total USD cost for this call, calculated from PRICING.
    timestamp      : datetime — wall-clock time when track() captured the record.
    """

    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.now)


class CostTracker:
    """
    Tracks LLM token usage and API costs for LangChain-routed calls.

    WHAT IT TRACKS
    --------------
    Every call wrapped with track() produces one CallRecord containing:
      - prompt_tokens and completion_tokens (from LangChain's OpenAI callback)
      - USD cost calculated from the built-in PRICING table
      - The operation label and model name supplied by the caller

    All records accumulate in memory until reset() is called. get_summary()
    returns a snapshot of the accumulated state; is_within_budget() compares
    total cost against a caller-supplied threshold.

    USAGE PATTERN
    -------------
        tracker = CostTracker()
        answer = tracker.track("get_answer", chatbot.get_answer, question)
        summary = tracker.get_summary()
        assert tracker.is_within_budget(budget_usd=1.00)

    PRICING TABLE
    -------------
    Pricing is per token in USD, derived from the OpenAI API pricing page
    (https://developers.openai.com/api/docs/models/gpt-5-mini) as of August 2026:

        gpt-5-mini             : $0.25  / 1M input tokens
                                 $2.00  / 1M output tokens
        text-embedding-3-small : $0.02  / 1M tokens (input only; no output)

    WHY MANUAL PRICING INSTEAD OF cb.total_cost
    --------------------------------------------
    LangChain's get_openai_callback() exposes a total_cost field, but its
    internal pricing table is updated on LangChain's own release cycle. Newly
    released models (such as gpt-5-mini) may show total_cost=0.0 even when
    tokens were consumed. Maintaining a local PRICING dict avoids this gap and
    makes pricing changes explicit and reviewable in version control.
    """

    PRICING: dict[str, dict[str, float]] = {
        "gpt-5-mini": {
            "input": 0.25 / 1_000_000,
            "output": 2.00 / 1_000_000,
        },
        "text-embedding-3-small": {
            "input": 0.02 / 1_000_000,
            "output": 0.0,
        },
    }

    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    def track(self, operation: str, fn: Callable, *args, **kwargs) -> Any:
        """
        Execute fn(*args, **kwargs) inside LangChain's OpenAI callback scope,
        capture token usage, compute cost, and append a CallRecord.

        WHAT IT MEASURES
        ----------------
        Wraps the callable in get_openai_callback() so that any OpenAI API
        calls made through LangChain during fn's execution are intercepted.
        After fn returns, prompt_tokens and completion_tokens are read from
        the callback and cost is calculated via calculate_cost().

        HOW IT WORKS
        ------------
        1. Open get_openai_callback() context — LangChain registers a handler
           that accumulates token counts for every OpenAI call in this scope.
        2. Call fn(*args, **kwargs) — the agent's LangChain chain runs normally;
           no changes to agent code are needed.
        3. Read cb.prompt_tokens and cb.completion_tokens after fn returns.
        4. Calculate cost using calculate_cost(Config.llm_model_name, ...).
        5. Append a CallRecord to self._records.
        6. Return fn's result unchanged.

        SCOPE NOTE
        ----------
        If fn internally makes multiple LangChain-routed calls (e.g. retriever +
        LLM), the callback accumulates all of them. This is correct for
        get_answer(), which fires one retriever call (no tokens) and one LLM
        call. If fn were to call clean_transcript() as well, both token counts
        would be merged into a single CallRecord under the supplied operation label.

        Args:
            operation: label stored in CallRecord.operation, e.g. "get_answer".
            fn:        callable to execute (typically chatbot.get_answer).
            *args:     positional arguments forwarded to fn.
            **kwargs:  keyword arguments forwarded to fn.

        Returns:
            The return value of fn(*args, **kwargs) — identical to calling fn
            directly, so track() is transparent to the caller.
        """
        with get_openai_callback() as cb:
            result = fn(*args, **kwargs)

        model = Config.llm_model_name
        cost = self.calculate_cost(model, cb.prompt_tokens, cb.completion_tokens)
        self._records.append(
            CallRecord(
                operation=operation,
                model=model,
                input_tokens=cb.prompt_tokens,
                output_tokens=cb.completion_tokens,
                cost_usd=cost,
            )
        )
        return result

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Compute the USD cost for a given model and token counts.

        WHAT IT MEASURES
        ----------------
        A pure arithmetic calculation — no API calls, no LangChain involvement.
        Looks up the model in PRICING and applies:

            cost = (input_tokens  × input_price_per_token)
                 + (output_tokens × output_price_per_token)

        FALLBACK BEHAVIOUR
        ------------------
        If the model name is not found in PRICING, the gpt-5-mini rates are used
        as a conservative fallback. This prevents a KeyError from crashing a test
        run when a model name changes, while still producing a meaningful cost estimate.

        Args:
            model:        model name key, e.g. "gpt-5-mini".
            input_tokens: number of prompt tokens (from cb.prompt_tokens).
            output_tokens: number of completion tokens (from cb.completion_tokens).

        Returns:
            Cost in USD as a float. Returns 0.0 when both token counts are 0.
        """
        pricing = self.PRICING.get(model, self.PRICING["gpt-5-mini"])
        return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

    def get_summary(self) -> dict:
        """
        Return an aggregated snapshot of all tracked calls.

        WHAT IT RETURNS
        ---------------
        A dict with the following keys:

            call_count          : int   — number of track() calls recorded so far.
            total_tokens        : int   — sum of input + output tokens across all calls.
            total_input_tokens  : int   — sum of prompt tokens across all calls.
            total_output_tokens : int   — sum of completion tokens across all calls.
            total_cost_usd      : float — sum of cost_usd across all calls.
            calls               : list  — list of dicts, one per CallRecord (via dataclasses.asdict).

        WHEN TO USE
        -----------
        Call after one or more track() invocations to see cumulative usage. The
        summary is a point-in-time snapshot — subsequent track() calls are not
        reflected unless get_summary() is called again.

        Returns:
            dict with keys defined above. Returns zero-valued fields when no
            calls have been recorded yet.
        """
        return {
            "call_count": len(self._records),
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in self._records),
            "total_input_tokens": sum(r.input_tokens for r in self._records),
            "total_output_tokens": sum(r.output_tokens for r in self._records),
            "total_cost_usd": sum(r.cost_usd for r in self._records),
            "calls": [asdict(r) for r in self._records],
        }

    def reset(self) -> None:
        """
        Clear all accumulated CallRecords, resetting the tracker to its initial state.

        WHAT IT DOES
        ------------
        Empties self._records in place. All previously accumulated token counts,
        costs, and operation labels are discarded. Subsequent calls to get_summary()
        will return zero-valued fields.

        WHEN TO USE
        -----------
        Call between test cases when you want per-test cost isolation. Without
        reset(), costs from earlier tests carry over into later tests' summaries,
        making it impossible to assert per-test budget compliance.
        """
        self._records.clear()

    def is_within_budget(self, budget_usd: float) -> bool:
        """
        Check whether the accumulated total cost is at or below a budget threshold.

        WHAT IT MEASURES
        ----------------
        Compares the sum of cost_usd across all records against budget_usd.
        Returns True if total_cost <= budget_usd, False otherwise.

        WHEN TO USE
        -----------
        Assert at the end of a test (or a group of tests) to enforce a per-run
        spend cap. Combine with reset() between test cases for per-test budgets,
        or call once after a full test suite for a session-level budget assertion.

        Example:
            tracker = CostTracker()
            for q in questions:
                tracker.track("get_answer", chatbot.get_answer, q)
            assert tracker.is_within_budget(0.10), "Test suite exceeded $0.10 budget"

        Args:
            budget_usd: maximum acceptable total cost in USD (inclusive).

        Returns:
            True  — total cost is within budget.
            False — total cost exceeds budget.
        """
        return sum(r.cost_usd for r in self._records) <= budget_usd
