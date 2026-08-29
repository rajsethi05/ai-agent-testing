# *---------------------------------------------------------------------------*
# Test Run Configuration
#
# Tune these before a test run to control scope and cost.
# Changes here apply across all test files automatically.
#
# CI/CD overrides: set env vars before running pytest to change suite scope
# without editing this file.
#   SUITE_MAX_CASES=3       → limits test cases (sanity suite)
#   SUITE_VIDEO_IDS=id1,id2 → restricts to specific videos
#   SUITE_MAX_INJECTION=3   → limits injection patterns (sanity suite)
# *---------------------------------------------------------------------------*

import os


def _parse_int_env(key: str) -> int | None:
    val = os.environ.get(key)
    return int(val) if val is not None else None


def _parse_list_env(key: str) -> list[str] | None:
    val = os.environ.get(key)
    return val.split(",") if val is not None else None


# * ----------- Sanity suite — first 3 cases across all videos ------------- *
# SUITE_MAX_CASES=3 pytest tests/

# * ----------- Deep run on one specific video --------------- *
# SUITE_VIDEO_IDS=HAoKJT3af7Y pytest tests/

# * ----------- Full suite ----------------- *
# pytest tests/

MAX_TEST_CASES     = _parse_int_env("SUITE_MAX_CASES")
VIDEO_IDS          = _parse_list_env("SUITE_VIDEO_IDS")

# Maximum number of injection patterns to run (applies to both direct and indirect).
# Follows SUITE_MAX_CASES when set, or can be overridden independently.
MAX_INJECTION_PATTERNS = _parse_int_env("SUITE_MAX_INJECTION") or MAX_TEST_CASES
