"""Batch test script for the Agentic Graph Query System."""

from __future__ import annotations

import asyncio
import json
import time
import httpx

BASE_URL = "http://localhost:8001"

TEST_QUESTIONS = [
    {
        "question": "show CNAPP solutions",
        "expected_strategy": "discovery_first",
        "description": "Unknown acronym discovery",
    },
    {
        "question": "how many applications are there?",
        "expected_strategy": "aggregation",
        "description": "Count queries",
    },
    {
        "question": "what's connected to Domain nodes?",
        "expected_strategy": "schema_exploration",
        "description": "Relationship exploration",
    },
    {
        "question": "show me Movie titled Inception",
        "expected_strategy": "direct_query",
        "description": "Clear entity reference",
    },
    {
        "question": "find HK applications",
        "expected_strategy": "discovery_first",
        "description": "Ambiguous term handling",
    },
    {
        "question": "How many movies are in the database?",
        "expected_strategy": "aggregation",
        "description": "Simple count query",
    },
    {
        "question": "List top 5 actors by number of movies they acted in.",
        "expected_strategy": "aggregation",
        "description": "Complex aggregation and ordering",
    },
    {
        "question": "Which director has directed the most movies?",
        "expected_strategy": "aggregation",
        "description": "Complex aggregation with MAX",
    },
    {
        "question": "What are the most popular genres?",
        "expected_strategy": "aggregation",
        "description": "Group by and count",
    },
    {
        "question": "Tell me about Tom Hanks movies.",
        "expected_strategy": "discovery_first",
        "description": "Entity discovery with contextual lookup",
    },
]


async def check_health() -> bool:
    """Check if the server is up and agentic system is initialized."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_URL}/api/v1/agentic/health", timeout=10)
            data = resp.json()
            print(f"  Health: {data['status']} (supervisor={data['supervisor_initialized']})")
            return data.get("supervisor_initialized", False)
        except Exception as exc:
            print(f"  Health check failed: {exc}")
            return False


async def run_test(question_data: dict) -> dict:
    """Send a single question and record the result."""
    q = question_data["question"]
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/agentic/chat",
                json={"question": q},
                timeout=120,
            )
            elapsed = time.time() - t0
            data = resp.json()
            return {
                "question": q,
                "description": question_data["description"],
                "expected_strategy": question_data["expected_strategy"],
                "actual_strategy": data.get("strategy_used", "unknown"),
                "attempts": data.get("attempts", 0),
                "success": data.get("success", False),
                "answer_preview": (data.get("answer", "")[:120] + "…")
                if len(data.get("answer", "")) > 120
                else data.get("answer", ""),
                "elapsed_s": round(elapsed, 2),
                "error": None,
            }
        except Exception as exc:
            return {
                "question": q,
                "description": question_data["description"],
                "expected_strategy": question_data["expected_strategy"],
                "actual_strategy": "error",
                "attempts": 0,
                "success": False,
                "answer_preview": "",
                "elapsed_s": round(time.time() - t0, 2),
                "error": str(exc),
            }


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("BATCH TEST RESULTS")
    print("=" * 100)
    print(
        f"{'#':<3} {'Question':<35} {'Expected':<20} {'Actual':<20} "
        f"{'OK':<5} {'Att':<4} {'Time':<7}"
    )
    print("-" * 100)

    passed = 0
    for i, r in enumerate(results, 1):
        strategy_match = r["expected_strategy"] == r["actual_strategy"]
        ok = "✓" if r["success"] else "✗"
        match_icon = "=" if strategy_match else "≠"
        print(
            f"{i:<3} {r['question'][:33]:<35} {r['expected_strategy']:<20} "
            f"{r['actual_strategy']:<20} {ok:<5} {r['attempts']:<4} {r['elapsed_s']:<7.1f}s"
        )
        if r["success"]:
            passed += 1

    print("-" * 100)
    print(f"Passed: {passed}/{len(results)}")
    print()

    # Detail section
    for i, r in enumerate(results, 1):
        print(f"\n--- Q{i}: {r['question']}")
        print(f"    Strategy: {r['actual_strategy']} | Attempts: {r['attempts']}")
        if r["error"]:
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    Answer: {r['answer_preview']}")


async def main() -> None:
    print("Agentic Graph Query System — Batch Test")
    print("=" * 50)
    print(f"Server: {BASE_URL}")
    print()

    # Health check
    print("[1/2] Health check …")
    healthy = await check_health()
    if not healthy:
        print("⚠  Supervisor not initialized — tests may fail")
        print()

    # Run tests
    print(f"\n[2/2] Running {len(TEST_QUESTIONS)} test questions …\n")
    results = []
    for i, q_data in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {q_data['question']} …")
        result = await run_test(q_data)
        results.append(result)
        status = "✓" if result["success"] else "✗"
        print(f"        {status} {result['actual_strategy']} ({result['elapsed_s']}s)")

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
