"""
Download raw eval datasets into evals/datasets/raw/ (gitignored).

Sources (verified 2026-07):
- QQP:      HuggingFace GLUE parquet (validation split, 40k pairs)
- SimpleQA: OpenAI public blob CSV (4,332 factoid Q&A, MIT license)
- FreshQA:  Google Sheets CSV export - BEST EFFORT: the unauthenticated
            export sometimes 403s; we warn and continue because its
            volatility labels only matter for the later TTL feature.

Run: python evals/fetch_datasets.py [--only qqp|simpleqa|freshqa]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent / "datasets" / "raw"

QQP_URL = "https://huggingface.co/api/datasets/nyu-mll/glue/parquet/qqp/validation/0.parquet"
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
FRESHQA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "19lsv-6YVvsB8nlrhYE1O82LT4TpjLQink8lKYn5Cxmc/export?format=csv"
)

FETCH_TIMEOUT_SECS = 60

_TARGETS: dict[str, tuple[str, str]] = {
    "qqp": (QQP_URL, "qqp_validation.parquet"),
    "simpleqa": (SIMPLEQA_URL, "simple_qa_test_set.csv"),
    "freshqa": (FRESHQA_URL, "freshqa.csv"),
}


def fetch_file(url: str, dest: Path, timeout_secs: int = FETCH_TIMEOUT_SECS) -> Path | None:
    """Download a file, returning None (not raising) on failure.

    Args:
        url: Source URL.
        dest: Destination path.
        timeout_secs: Request timeout.

    Returns:
        The destination path on success, None on any failure.
    """
    try:
        resp = requests.get(url, timeout=timeout_secs, stream=True)
    except requests.RequestException as exc:
        print(f"  FAIL {url} ({exc})")
        return None
    if resp.status_code != 200:
        print(f"  FAIL {url} (HTTP {resp.status_code})")
        return None
    # Google returns an HTML login page instead of 403 in some cases - detect it.
    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        print(f"  FAIL {url} (got HTML, likely auth wall)")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    print(f"  OK   {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def main() -> None:
    """Fetch requested datasets; FreshQA failure is non-fatal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(_TARGETS), default=None)
    args = parser.parse_args()

    names = [args.only] if args.only else list(_TARGETS)
    failures: list[str] = []
    for name in names:
        url, filename = _TARGETS[name]
        print(f"Fetching {name}...")
        if fetch_file(url, RAW_DIR / filename) is None:
            failures.append(name)

    hard_failures = [n for n in failures if n != "freshqa"]
    if "freshqa" in failures:
        print("NOTE: FreshQA export unavailable - continuing without it "
              "(only needed for the later volatility-TTL feature).")
    if hard_failures:
        print(f"ERROR: required datasets failed: {hard_failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
