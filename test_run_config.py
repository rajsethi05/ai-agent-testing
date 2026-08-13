# *---------------------------------------------------------------------------*
# Test Run Configuration
#
# Tune these before a test run to control scope and cost.
# Changes here apply across all test files automatically.
# *---------------------------------------------------------------------------*


# * ----------- Quick smoke run — first 3 cases across all videos ------------- *
MAX_TEST_CASES = 3 # will run only this number of tests
VIDEO_IDS = None # will run all tests for this videoID

# * ----------- Deep run on one specific video --------------- *
# MAX_TEST_CASES = None
# VIDEO_IDS = ["HAoKJT3af7Y"]


# * ----------- Full suite ----------------- *
# MAX_TEST_CASES = None
# VIDEO_IDS = None


# Maximum number of injection patterns to run (applies to both direct and indirect).
# Set to None to run all patterns.
# Example: MAX_INJECTION_PATTERNS = 3  → runs the first 3 patterns of each type
MAX_INJECTION_PATTERNS = MAX_TEST_CASES
