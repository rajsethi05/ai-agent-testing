"""
Retriever quality utilities for the RAG evaluation framework.
Contains deterministic (non-LLM) metrics for measuring retrieval performance.
"""
from typing import List


def token_overlap(text_a: str, text_b: str) -> float:
    """
    Computes the Jaccard similarity between two text strings based on word tokens.

    Jaccard similarity = |A ∩ B| / |A ∪ B|

    where A and B are the sets of lowercase word tokens from each string.
    The result is a float in [0.0, 1.0]:
      - 0.0 means no shared tokens at all
      - 1.0 means both texts contain exactly the same set of words

    This is used as a proxy for chunk overlap because exact string equality fails
    when the golden context chunks (generated at chunk_size=150 tokens by the
    Synthesizer) differ in size from the chatbot's retrieved chunks (chunk_size=600
    chars). Token overlap is robust to those size differences.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def context_hit_rate(retrieved_chunks: List[str], golden_chunks: List[str],
                     overlap_threshold: float = 0.5) -> float:
    """
    Computes the Context Hit Rate for a single test case.

    DEFINITION
    ----------
    Context Hit Rate = (number of golden chunks with at least one matching retrieved chunk)
                       / (total number of golden chunks)

    A retrieved chunk "matches" a golden chunk when their token_overlap() score is at or
    above overlap_threshold (default 0.5).

    WHY THIS METRIC EXISTS
    ----------------------
    The golden dataset's `context` field contains the exact chunks the Synthesizer
    used when generating each Q&A pair. These are the ground-truth source chunks —
    the pieces of text that an ideal retriever should fetch.

    Context Hit Rate answers: "Did the live retriever actually find that source material?"

    If the score is low, the retriever is missing the relevant chunks entirely. No
    amount of LLM quality can compensate for a retriever that never surfaces the right
    content in the first place.

    HOW IT DIFFERS FROM DEEPEVAL'S CONTEXTUAL METRICS
    --------------------------------------------------
    - ContextualRecallMetric    — LLM judges if retrieved chunks support the expected answer
    - ContextualPrecisionMetric — LLM judges if relevant chunks are ranked above irrelevant ones
    - ContextualRelevancyMetric — LLM judges if retrieved chunks are relevant to the query
    - Context Hit Rate          — deterministic; directly compares retrieved chunks against the
                                  golden chunks that were the source of the Q&A pair

    Because it is deterministic, it requires no LLM calls, incurs zero API cost, and
    runs significantly faster than the three metrics above.

    PARAMETERS
    ----------
    retrieved_chunks   : chunks returned by the live retriever for a given query
    golden_chunks      : ground-truth chunks from the golden dataset (entry["context"])
    overlap_threshold  : minimum Jaccard score for a retrieved chunk to count as a hit
                         (default 0.5 — at least half the tokens must overlap)

    RETURNS
    -------
    Float in [0.0, 1.0]. A score of 1.0 means every golden chunk was covered by at
    least one retrieved chunk. A score of 0.0 means none were.
    """
    if not golden_chunks:
        return 0.0
    hits = sum(
        1 for golden in golden_chunks
        if any(token_overlap(golden, retrieved) >= overlap_threshold for retrieved in retrieved_chunks)
    )
    return hits / len(golden_chunks)
