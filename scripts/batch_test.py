"""
Batch test runner for the Neo4j NL Agent.
Sends test questions to POST /api/ask and validates responses.

Usage:
    python -m scripts.batch_test
    python -m scripts.batch_test --base-url http://localhost:8000
    python -m scripts.batch_test --timeout 90
    python -m scripts.batch_test --output data/test_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── test definitions ──────────────────────────────────────────


@dataclass
class TestCase:
    id: str
    category: str
    question: str
    expected_keywords: list[str] = field(default_factory=list)


TEST_CASES: list[TestCase] = [
    # ── Simple Lookup ──
    TestCase(
        id="simple_01",
        category="simple_lookup",
        question="Who is Alice Chen?",
        expected_keywords=["Alice", "Engineering"],
    ),
    TestCase(
        id="simple_02",
        category="simple_lookup",
        question="Tell me about the Engineering department",
        expected_keywords=["Engineering"],
    ),

    # ── Entity Matching ──
    TestCase(
        id="entity_01",
        category="entity_matching",
        question="What department does Bob Martinez work in?",
        expected_keywords=["Bob", "Engineering"],
    ),
    TestCase(
        id="entity_02",
        category="entity_matching",
        question="What skills does Carol Johnson have?",
        expected_keywords=["Carol", "Python"],
    ),

    # ── Aggregation ──
    TestCase(
        id="agg_01",
        category="aggregation",
        question="How many people work in Engineering?",
        expected_keywords=["4"],
    ),
    TestCase(
        id="agg_02",
        category="aggregation",
        question="How many projects are there?",
        expected_keywords=["4"],
    ),

    # ── Multi-hop Traversal ──
    TestCase(
        id="hop_01",
        category="multi_hop",
        question="What projects does the Engineering department own?",
        expected_keywords=["Phoenix"],
    ),
    TestCase(
        id="hop_02",
        category="multi_hop",
        question="Which people work on projects owned by the Product department?",
        expected_keywords=["David"],
    ),

    # ── Fuzzy Matching (deliberate typos) ──
    TestCase(
        id="fuzzy_01",
        category="fuzzy_matching",
        question="Show me Alce Chen",
        expected_keywords=["Alice"],
    ),
    TestCase(
        id="fuzzy_02",
        category="fuzzy_matching",
        question="Tell me about Enginnering department",
        expected_keywords=["Engineering"],
    ),

    # ── Reverse Traversal ──
    TestCase(
        id="rev_01",
        category="reverse_traversal",
        question="Who manages the Engineering department?",
        expected_keywords=["Alice"],
    ),
    TestCase(
        id="rev_02",
        category="reverse_traversal",
        question="Who reports to Alice Chen?",
        expected_keywords=["Bob"],
    ),

    # ── Top-N / Filtering ──
    TestCase(
        id="topn_01",
        category="top_n",
        question="Which department has the highest budget?",
        expected_keywords=["Engineering"],
    ),
    TestCase(
        id="filter_01",
        category="property_filter",
        question="Who was hired after 2021?",
        expected_keywords=["Grace"],
    ),

    # ── Location ──
    TestCase(
        id="loc_01",
        category="location",
        question="Which departments are in San Francisco?",
        expected_keywords=["Engineering"],
    ),
]


# ── result model ──────────────────────────────────────────────


@dataclass
class TestResult:
    test_id: str
    category: str
    question: str
    passed: bool
    answer: str
    success: bool
    expected_keywords: list[str]
    matched_keywords: list[str]
    missing_keywords: list[str]
    elapsed_seconds: float
    error: str | None = None


# ── runner ────────────────────────────────────────────────────


async def run_single(
    client: httpx.AsyncClient,
    test: TestCase,
    base_url: str,
) -> TestResult:
    """Send one question and validate the response."""
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/api/ask",
            json={"question": test.question},
        )
        elapsed = time.perf_counter() - start

        if resp.status_code != 200:
            return TestResult(
                test_id=test.id, category=test.category, question=test.question,
                passed=False, answer="", success=False,
                expected_keywords=test.expected_keywords,
                matched_keywords=[], missing_keywords=test.expected_keywords,
                elapsed_seconds=elapsed,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        answer = data.get("answer", "")
        success = data.get("success", False)

        answer_lower = answer.lower()
        matched = [kw for kw in test.expected_keywords if kw.lower() in answer_lower]
        missing = [kw for kw in test.expected_keywords if kw.lower() not in answer_lower]

        passed = success and len(missing) == 0

        return TestResult(
            test_id=test.id, category=test.category, question=test.question,
            passed=passed, answer=answer, success=success,
            expected_keywords=test.expected_keywords,
            matched_keywords=matched, missing_keywords=missing,
            elapsed_seconds=elapsed,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return TestResult(
            test_id=test.id, category=test.category, question=test.question,
            passed=False, answer="", success=False,
            expected_keywords=test.expected_keywords,
            matched_keywords=[], missing_keywords=test.expected_keywords,
            elapsed_seconds=elapsed,
            error=str(exc),
        )


async def run_all(base_url: str, timeout: float) -> list[TestResult]:
    """Run all test cases sequentially."""
    results: list[TestResult] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, test in enumerate(TEST_CASES, 1):
            label = f"[{i:2d}/{len(TEST_CASES)}]"
            print(f"{label} {test.category}: {test.question[:60]}...", end=" ", flush=True)

            result = await run_single(client, test, base_url)

            tag = "PASS" if result.passed else "FAIL"
            print(f"{tag} ({result.elapsed_seconds:.1f}s)")

            if not result.passed:
                if result.error:
                    print(f"         Error: {result.error[:120]}")
                elif result.missing_keywords:
                    print(f"         Missing: {result.missing_keywords}")
                    print(f"         Answer:  {result.answer[:120]}...")

            results.append(result)

    return results


def print_summary(results: list[TestResult]):
    """Print pass/fail summary by category."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{total} passed ({passed / total * 100:.0f}%)")
    print(f"{'=' * 60}")

    # Group by category
    categories: dict[str, list[TestResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat in sorted(categories):
        cat_results = categories[cat]
        cat_passed = sum(1 for r in cat_results if r.passed)
        status = "OK" if cat_passed == len(cat_results) else "!!"
        print(f"  [{status}] {cat}: {cat_passed}/{len(cat_results)}")

    # Timing
    times = [r.elapsed_seconds for r in results]
    avg_t = sum(times) / len(times)
    print(f"\nTiming: avg={avg_t:.1f}s  min={min(times):.1f}s  max={max(times):.1f}s  total={sum(times):.1f}s")


def save_results(results: list[TestResult], output_path: str):
    """Save detailed results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = {
        "run_at": datetime.now().isoformat(),
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to {output_path}")


# ── main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Batch test runner for Neo4j NL Agent")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the running agent")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout per request in seconds")
    parser.add_argument("--output", default="data/test_results.json", help="Output JSON file path")
    args = parser.parse_args()

    print(f"Neo4j NL Agent — Batch Test Runner")
    print(f"Target: {args.base_url}")
    print(f"Tests:  {len(TEST_CASES)}")
    print(f"Timeout: {args.timeout}s per request")
    print(f"{'=' * 60}\n")

    results = asyncio.run(run_all(args.base_url, args.timeout))
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
