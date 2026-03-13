"""Batch test script for the Agentic Graph Query System.

Features:
  • Concurrent requests — configurable via --concurrency N (default: 1)
  • Full response logging — every request/response saved to response_data_time.json
  • Cypher attempt tracking — shows all generated Cypher + reasoning per question

Usage:
    python test_agentic_batch.py                  # sequential (1 at a time)
    python test_agentic_batch.py --concurrency 3  # 3 concurrent requests
    python test_agentic_batch.py -c 5             # 5 concurrent requests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

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


# ── Helpers ──────────────────────────────────────────────────────────────────


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


async def run_test(question_data: dict, client: httpx.AsyncClient) -> dict:
    """Send a single question and record the full result."""
    q = question_data["question"]
    t0 = time.time()
    try:
        resp = await client.post(
            f"{BASE_URL}/api/v1/agentic/chat",
            json={"question": q},
            timeout=300,
        )
        elapsed = time.time() - t0
        data = resp.json()

        answer = data.get("answer", "")
        cypher_attempts = data.get("cypher_attempts", [])

        return {
            "question": q,
            "description": question_data["description"],
            "expected_strategy": question_data["expected_strategy"],
            "actual_strategy": data.get("strategy_used", "unknown"),
            "attempts": data.get("attempts", 0),
            "success": data.get("success", False),
            "answer": answer,
            "answer_preview": (answer[:120] + "…") if len(answer) > 120 else answer,
            "elapsed_s": round(elapsed, 2),
            "trace_id": data.get("trace_id", ""),
            "cypher_attempts": cypher_attempts,
            "specialist_log": data.get("specialist_log", []),
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {
            "question": q,
            "description": question_data["description"],
            "expected_strategy": question_data["expected_strategy"],
            "actual_strategy": "error",
            "attempts": 0,
            "success": False,
            "answer": "",
            "answer_preview": "",
            "elapsed_s": round(time.time() - t0, 2),
            "trace_id": "",
            "cypher_attempts": [],
            "specialist_log": [],
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── Concurrency runner ──────────────────────────────────────────────────────


async def run_batch_concurrent(
    questions: list[dict], concurrency: int
) -> list[dict]:
    """Run all questions with bounded concurrency using a semaphore."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(questions)

    async def _worker(idx: int, q_data: dict, client: httpx.AsyncClient) -> None:
        async with semaphore:
            tag = f"[{idx + 1}/{len(questions)}]"
            print(f"  {tag} ▶ {q_data['question']}")
            result = await run_test(q_data, client)
            results[idx] = result
            status = "✓" if result["success"] else "✗"
            cyphers = len(result["cypher_attempts"])
            print(
                f"  {tag} {status} {result['actual_strategy']} "
                f"({result['elapsed_s']}s, {cyphers} cypher(s))"
            )

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(_worker(i, q, client))
            for i, q in enumerate(questions)
        ]
        await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


async def run_batch_sequential(questions: list[dict]) -> list[dict]:
    """Run all questions one at a time (concurrency=1)."""
    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for i, q_data in enumerate(questions):
            tag = f"[{i + 1}/{len(questions)}]"
            print(f"  {tag} {q_data['question']} …")
            result = await run_test(q_data, client)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            cyphers = len(result["cypher_attempts"])
            print(
                f"        {status} {result['actual_strategy']} "
                f"({result['elapsed_s']}s, {cyphers} cypher(s))"
            )
    return results


# ── Output ───────────────────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 110)
    print("BATCH TEST RESULTS")
    print("=" * 110)
    print(
        f"{'#':<3} {'Question':<35} {'Expected':<20} {'Actual':<20} "
        f"{'OK':<5} {'Att':<4} {'Cyphers':<8} {'Time':<7}"
    )
    print("-" * 110)

    passed = 0
    total_time = 0.0
    for i, r in enumerate(results, 1):
        ok = "✓" if r["success"] else "✗"
        cyphers = len(r["cypher_attempts"])
        print(
            f"{i:<3} {r['question'][:33]:<35} {r['expected_strategy']:<20} "
            f"{r['actual_strategy']:<20} {ok:<5} {r['attempts']:<4} "
            f"{cyphers:<8} {r['elapsed_s']:<7.1f}s"
        )
        if r["success"]:
            passed += 1
        total_time += r["elapsed_s"]

    print("-" * 110)
    print(f"Passed: {passed}/{len(results)}  |  Total time: {total_time:.1f}s")
    print()

    # Detail section — answer + cypher attempts
    for i, r in enumerate(results, 1):
        print(f"\n{'─' * 80}")
        print(f"Q{i}: {r['question']}")
        print(f"  Strategy: {r['actual_strategy']}  |  Attempts: {r['attempts']}  |  Trace: {r['trace_id']}")

        if r["error"]:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Answer: {r['answer_preview']}")

        # Show all cypher attempts with reasoning
        if r["cypher_attempts"]:
            for j, ca in enumerate(r["cypher_attempts"], 1):
                status = "✓" if ca.get("success") else "✗"
                print(f"\n  Cypher #{j} (attempt {ca.get('attempt', '?')}) {status}")
                print(f"    Query:     {ca.get('cypher', 'N/A')}")
                if ca.get("parameters"):
                    print(f"    Params:    {ca.get('parameters')}")
                if ca.get("reasoning"):
                    reasoning = ca["reasoning"]
                    # Wrap long reasoning
                    if len(reasoning) > 100:
                        reasoning = reasoning[:100] + "…"
                    print(f"    Reasoning: {reasoning}")
                if ca.get("error"):
                    print(f"    Error:     {ca['error']}")
                if ca.get("row_count"):
                    print(f"    Rows:      {ca['row_count']}")


def save_report(results: list[dict], concurrency: int, total_time: float, report_path: Path) -> None:
    """Save full request/response data to JSON file."""
    report = {
        "run_info": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": BASE_URL,
            "concurrency": concurrency,
            "total_questions": len(results),
            "passed": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "total_time_s": round(total_time, 2),
        },
        "results": results,
    }

    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n📄 Full report saved to: {report_path.resolve()}")


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test for Agentic Graph Query System")
    parser.add_argument(
        "-c", "--concurrency",
        type=int, default=1,
        help="Number of concurrent requests (default: 1 = sequential)",
    )
    args = parser.parse_args()
    concurrency: int = args.concurrency

    # Generate timestamped report filename
    run_start_dt = datetime.now()
    report_path = Path(f"response_{run_start_dt.strftime('%Y%m%d_%H%M%S')}.json")

    print("Agentic Graph Query System — Batch Test")
    print("=" * 50)
    print(f"Server:      {BASE_URL}")
    print(f"Concurrency: {concurrency}")
    print(f"Questions:   {len(TEST_QUESTIONS)}")
    print()

    # 1. Health check
    print("[1/3] Health check …")
    healthy = await check_health()
    if not healthy:
        print("⚠  Supervisor not initialized — tests may fail")
    print()

    # 2. Run tests
    mode = f"concurrent (×{concurrency})" if concurrency > 1 else "sequential"
    print(f"[2/3] Running {len(TEST_QUESTIONS)} questions ({mode}) …\n")

    run_start = time.time()
    if concurrency > 1:
        results = await run_batch_concurrent(TEST_QUESTIONS, concurrency)
    else:
        results = await run_batch_sequential(TEST_QUESTIONS)
    total_time = time.time() - run_start

    # 3. Summary + report
    print(f"\n[3/3] Summary")
    print_summary(results)
    save_report(results, concurrency, total_time, report_path)


if __name__ == "__main__":
    asyncio.run(main())
