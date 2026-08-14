"""
Regression test runner.

Usage:
    python scripts/run_regression.py --suite sanity
    python scripts/run_regression.py --suite full

Suites:
    sanity  — All test files, 3 test cases each. Fast coverage pass.
    full    — All test files, all test cases. Weekly / pre-release run.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

SUITE_ENV = {
    "sanity": {
        "SUITE_MAX_CASES": "3",
        "SUITE_MAX_INJECTION": "3",
    },
    "full": {},
}


def run(suite: str) -> int:
    # Only apply suite env defaults when not already set by the caller (e.g. CI)
    env_overrides = {k: v for k, v in SUITE_ENV[suite].items() if k not in os.environ}
    env = {**os.environ, **env_overrides}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"regression_{suite}_{timestamp}.html"
    REPORTS_DIR.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest", "tests/",
        f"--html={report_path}",
        "--self-contained-html",
        "-v",
    ]

    print(f"\n{'='*60}")
    print(f"  Suite  : {suite.upper()}")
    print(f"  Report : {report_path.relative_to(ROOT)}")
    if env_overrides:
        for k, v in env_overrides.items():
            print(f"  {k} = {v}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env, cwd=ROOT)

    print(f"\n{'='*60}")
    print(f"  {'PASSED' if result.returncode == 0 else 'FAILED'}")
    print(f"  Report saved → {report_path.relative_to(ROOT)}")
    print(f"{'='*60}\n")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run regression test suite.")
    parser.add_argument(
        "--suite",
        choices=["sanity", "full"],
        default="sanity",
        help="sanity: 3 cases each (default). full: all cases.",
    )
    args = parser.parse_args()
    sys.exit(run(args.suite))


if __name__ == "__main__":
    main()
