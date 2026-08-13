from pathlib import Path

# *------------- directories --------------*

ROOT_DIR = Path(__file__).resolve().parent
GOLDENS_DIR = ROOT_DIR / "framework" / "golden_dataset" / "datasets"

# *------------- test_retrieval.py --------*

LATENCY_THRESHOLD_MS = 2000

# *------------- test_deterministic.py ----*

N_RUNS = 3
LENGTH_CV_THRESHOLD = 0.05
SEMANTICEQUIVALENCE_THRESHOLD = 0.8

# *------------- test_cost_tracking.py ----*

COST_PER_QUERY_LIMIT_USD = 0.01

# *------------- test_prompt_injection.py ---------*

DEFAULT_VIDEO_ID = "HAoKJT3af7Y"

# *------------- test_hallucination.py ----*
CONSISTENCY_THRESHOLD = 0.7
