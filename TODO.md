# TODOs:

#### Core Testing Modules to Build

1. Retrieval Quality Module
   - ContextualRecallMetric (DeepEval) ✅
   - ContextualPrecisionMetric (DeepEval) ✅
   - ContextualRelevancyMetric (DeepEval) ✅
   - Context hit rate ✅
   - Retrieval latency ✅

2. Hallucination Detection Module
   - Faithfulness scorer ✅
   - Answer Relevancy ✅
   - Consistency checker ✅

3. Prompt Injection Testing Module
   - Library of injection patterns (20-30 test cases)
   - Direct injection detector
   - Indirect injection simulator (malicious content in retrieved docs)
   - Success/failure tracker

4. Deterministic Testing Module
   * Temperature=0 consistency tests

5. Output Validation Module
   * Schema validator (Pydantic models)
   * Format checker (JSON, structured outputs)
   * Citation validator (for RAG responses)

6. Cost Tracking Module
   * Token counter (input + output)
   * API call tracker
   * Cost calculator (per model pricing)

7. Metrics Collection & Reporting
   * Precision/Recall/F1 calculator
   * Latency tracker
   * Success rate aggregator
   * Results visualizer (tables/charts)

8. Regression Pipeline
   * Test suite organization
   * CI/CD integration (GitHub Actions)
   * Automated test runner
   * Report generator