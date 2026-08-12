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
   - Library of injection patterns (20-30 test cases) ✅
   - Direct injection detector ✅
   - Indirect injection simulator (malicious content in retrieved docs) ✅
   - Success/failure tracker ✅

4. Deterministic Testing Module
   * Temperature=0 consistency tests ✅

5. Output Validation Module
   * Skipped — YouTube RAG chatbot returns plain text with no structured output schema to validate 🚫

6. Cost Tracking Module
   - Token counter (input + output) ✅
   - API call tracker ✅
   - Cost calculator (per model pricing) ✅

7. Metrics Collection & Reporting
   * html report implementation ✅

8. Polish the project
   * Fix the metric logging bug
   * Write the README
   * Clean dead code + un-hardcode the [:3]
   * Create utils and arrange params

9. Regression Pipeline
   * Test suite organization
   * CI/CD integration (GitHub Actions)
   * Automated test runner
   * Report generator