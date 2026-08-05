"""
Deterministic testing utilities for LLM agents.

OVERVIEW
--------
Deterministic testing verifies that an LLM agent produces the same output
every time when given the same input under controlled conditions (temperature=0).
It is distinct from — and stricter than — consistency testing:

  Consistency (test_hallucination.py)
      Do two answers agree on the facts? Tolerates phrasing differences.
      Failure signals factual instability / hallucination risk.

  Determinism (this module)
      Is the output character-for-character identical across N runs?
      Failure signals non-reproducibility in the LLM backend or pipeline.

WHY DETERMINISM MATTERS
-----------------------
- Reproducibility: a bug reported by a user must be reproducible to be fixed.
  If the agent gives different answers to the same question, root-cause analysis
  is extremely difficult.
- Regression detection: a test that compares against a stored "golden" output
  only works if the model is deterministic. Non-determinism makes golden-answer
  tests flaky.
- Trust: users who ask the same question twice and get meaningfully different
  answers lose confidence in the system.

HOW temperature=0 RELATES TO DETERMINISM
-----------------------------------------
Setting temperature=0 tells the LLM to always pick the highest-probability
next token (greedy decoding). In theory this produces identical outputs for
identical inputs. In practice, minor non-determinism can still arise from:
  - Floating-point rounding differences across GPU/CPU runs
  - Server-side batching (OpenAI may batch requests differently across calls)
  - Model version drift (silently updated model weights)
  - Retriever non-determinism (vector DB returning documents in a different order)

This module gives you tools to measure and assert all of these properties.

METRICS IN THIS MODULE
----------------------
1. compute_exact_match_rate   — pure Python, no LLM-as-judge
2. compute_length_cv          — pure Python, no LLM-as-judge
3. SemanticEquivalenceMetric  — GEval (LLM-as-judge, stricter than Consistency)
"""

import statistics

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def compute_exact_match_rate(outputs: list[str]) -> float:
    """
    Compute the fraction of output pairs that are character-for-character identical.

    WHAT IT MEASURES
    ----------------
    Given N outputs produced by repeated calls at temperature=0, all C(N,2)
    unique pairs are compared. The metric reports what fraction of those pairs
    are exactly equal as Python strings (same characters, same whitespace, same
    case, same punctuation).

    FORMULA
    -------
        identical_pairs = count of (a, b) pairs where a == b
        total_pairs     = N * (N - 1) / 2   [all unique unordered pairs]

        exact_match_rate = identical_pairs / total_pairs

    INTERPRETATION
    --------------
    1.0  Perfect determinism — every run produced the same output.
    0.67 Two of three runs matched; one deviated. Partial non-determinism.
    0.0  No two runs produced the same output. Highly non-deterministic.

    PASSING THRESHOLD
    -----------------
    >= 1.0 (all N outputs must be character-for-character identical).
    Any deviation at temperature=0 is noteworthy and warrants investigation.
    The test asserts exact_match_rate == 1.0 and fails with the actual rate
    and the differing outputs for debugging.

    COMMON FAILURE CAUSES
    ----------------------
    - OpenAI server-side batching inserts minor token-level variance
    - The retriever returned documents in a different order on one call,
      changing the assembled context string and therefore the LLM output
    - A model version was silently updated between calls
    - Trailing whitespace or newline handling differs across SDK versions

    Args:
        outputs: List of N response strings from repeated calls at temperature=0.
                 Must contain at least 2 elements; returns 1.0 for a single element.

    Returns:
        Float in [0.0, 1.0]. 1.0 = all pairs identical; 0.0 = no pairs identical.
    """
    if len(outputs) < 2:
        return 1.0
    pairs = [
        (outputs[i], outputs[j])
        for i in range(len(outputs))
        for j in range(i + 1, len(outputs))
    ]
    identical = sum(1 for a, b in pairs if a == b)
    return identical / len(pairs)


def compute_length_cv(outputs: list[str]) -> float:
    """
    Compute the Coefficient of Variation (CV) of response lengths across N outputs.

    WHAT IT MEASURES
    ----------------
    CV measures how much the character-length of responses varies relative to
    their average length. A low CV means every run produced a response of roughly
    the same length. A high CV means the model sometimes produces a one-liner and
    sometimes a multi-paragraph answer for the same input — structural instability.

    FORMULA
    -------
        lengths = [len(output) for output in outputs]
        CV      = stdev(lengths) / mean(lengths)

    INTERPRETATION
    --------------
    0.00  All outputs are exactly the same length.
    <0.05 Less than 5% variation — structurally stable (passing threshold).
    0.05–0.20  Moderate variation — worth investigating but may be acceptable.
    >0.20 High variation — the model's output structure is unstable at temperature=0.

    WHY LENGTH MATTERS EVEN WHEN EXACT MATCH FAILS
    -----------------------------------------------
    A response can differ in exact wording but still be structurally equivalent
    (same paragraph count, same level of detail). If the length CV is also high,
    the model is generating qualitatively different responses — not just re-phrasing.
    Length CV therefore complements exact_match_rate: together they distinguish
    "same content, slight wording difference" from "fundamentally different answer".

    PASSING THRESHOLD
    -----------------
    CV < 0.05. This permits up to ~5% variation in response length, which
    accommodates minor punctuation or whitespace differences without letting
    through responses that differ substantially in content depth.

    Args:
        outputs: List of N response strings from repeated calls at temperature=0.
                 Returns 0.0 for a single element or if all outputs are empty.

    Returns:
        Float >= 0. Lower is more deterministic.
    """
    if len(outputs) < 2:
        return 0.0
    lengths = [len(o) for o in outputs]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    return statistics.stdev(lengths) / mean


SemanticEquivalenceMetric = GEval(
    name="SemanticEquivalence",
    evaluation_steps=[
        "Compare 'actual output' (first answer) and 'expected output' (second answer) to the same question.",
        "Penalise if one answer includes specific facts, numbers, names, or examples that the other omits — both omissions and additions reduce the score.",
        "Penalise if the scope of one answer is broader or narrower than the other — for example, one qualifies a claim ('usually', 'in most cases') while the other states it absolutely.",
        "Penalise if one answer draws a conclusion, makes a recommendation, or expresses a caveat that the other does not.",
        "Minor differences in phrasing, sentence order, or synonymous word choice are acceptable and should not be penalised.",
        "Award a high score only if both answers convey exactly the same information at the same level of detail and with the same scope.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
)
"""
SemanticEquivalenceMetric — GEval (LLM-as-judge)

WHAT IT MEASURES
----------------
SemanticEquivalence checks whether two temperature=0 answers are informationally
identical — not just non-contradictory (which is what Consistency_metric checks),
but carrying exactly the same facts, scope, caveats, and level of detail.

HOW IT DIFFERS FROM Consistency_metric
---------------------------------------
Consistency_metric (framework/metrices.py) passes as long as the two answers
do not directly contradict each other. Two answers can be consistent while one
omits key facts that the other includes — Consistency does not penalise omission.

SemanticEquivalence is stricter:
  - Consistency   → penalises contradictions only
  - SemanticEquivalence → penalises contradictions AND informational asymmetry
                          (one answer says more or less than the other)

This stricter bar is appropriate for temperature=0 output: if the model is
deterministic, both answers should be informationally identical, not merely
non-contradictory.

DEEPEVAL DEFINITION
-------------------
GEval is DeepEval's general-purpose LLM-as-a-judge metric. It sends the
evaluation_steps as a scoring rubric to a judge LLM (OpenAI by default), which
scores the test case on a 0–1 scale. GEval handles:
  - Normalising the raw judge score into [0.0, 1.0]
  - Extracting a reason string explaining the score
  - Applying the threshold to determine pass/fail

Reference: https://docs.confident-ai.com/docs/metrics-llm-evals

TEST CASE USAGE
---------------
    test_case = LLMTestCase(
        input=query,
        actual_output=answer_run_1,
        expected_output=answer_run_2,
    )
    assert_test(test_case, [SemanticEquivalenceMetric])

Note: expected_output is repurposed here as the second run's answer, not a
ground-truth label — the same convention used by Consistency_metric.

THRESHOLD
---------
0.8 — stricter than other GEval metrics in this project (which use 0.7) because
temperature=0 answers should be nearly identical in content.
"""
SemanticEquivalenceMetric.threshold = 0.8
