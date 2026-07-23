# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

An industry-grade testing framework for LangChain/LangGraph agents, demonstrating QA expertise for LLM systems. The primary agent under test is a YouTube RAG chatbot. The framework evaluates agents using [DeepEval](https://docs.confident-ai.com/) metrics (GEval, Faithfulness, AnswerRelevancy) and synthetic golden datasets.

## Environment Setup

Python 3.13, managed via `.venv/`. API keys go in `.env` (not committed):

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
```

```bash
source .venv/bin/activate
```

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_deepeval.py -v

# Run via DeepEval CLI (enables richer metric output)
deepeval test run tests/test_deepeval.py -v

# Run the RAG agent demo
python agents/rag_youtube_chatbot/runner.py

# Regenerate golden datasets from transcripts
python framework/golden_dataset/data_generation.py
```

## Architecture

### Agent Under Test

`agents/rag_youtube_chatbot/yt_chatbot.py` — `YTChatbot` class implementing a RAG pipeline:
1. Fetches YouTube transcripts via `YouTubeTranscriptApi`, caches to `agents/rag_youtube_chatbot/transcripts/`
2. Cleans transcripts (removes ads/sponsors) via GPT-4
3. Chunks with `RecursiveCharacterTextSplitter` (600 chars, 120 overlap)
4. Embeds via `text-embedding-3-small`, persists in Chroma at `agents/rag_youtube_chatbot/vector_stores/`
5. Answers queries: retrieves top-2 chunks → prompt → `gpt-4o-mini`

### Testing Framework

```
framework/
  golden_dataset/
    data_generation.py     # Uses DeepEval Synthesizer to generate Q&A pairs from transcripts
    datasets/              # JSON golden datasets (6 files, one per video)
  retriever_quality.py     # Retriever evaluation utilities
tests/
  test_deepeval.py         # DeepEval metric tests (GEval, AnswerRelevancy, Faithfulness)
  test_retrieval.py        # Retrieval-focused tests (in progress)
```

**Golden dataset generation:** `data_generation.py` iterates transcript files, calls `Synthesizer` (chunk_size=150 tokens, 2 Q&A pairs per chunk), and writes JSON to `datasets/`.

**Evaluation pattern:** Tests construct `LLMTestCase(input, actual_output, retrieval_context)` and assert DeepEval metrics. The LLM-as-a-judge pattern is used throughout — metrics call out to the configured LLM (OpenAI by default) to score outputs.

### Planned Modules (not yet implemented)

Per `TODO.md`: hallucination detection, prompt injection testing, deterministic behavior testing, output validation, cost tracking, metrics dashboard, regression pipeline, CI/CD.

## Key Design Decisions

- **DeepEval** is the evaluation backbone. Custom criteria go into `GEval(evaluation_steps=[...])`. Predefined metrics (`FaithfulnessMetric`, `AnswerRelevancyMetric`) require `retrieval_context` in the test case.
- **Chroma** is used as a local persistent vector store — the store is reused across runs if it already exists (`get_or_create_vector_store`).
- Golden datasets are pre-generated and stored as JSON so tests don't regenerate them on every run.
