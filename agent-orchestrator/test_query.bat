@echo off
REM ============================================================================
REM  test_query.bat — Test the normal /query endpoint
REM  Usage: test_query.bat [port]    (default port: 8000)
REM ============================================================================

setlocal enabledelayedexpansion

set PORT=%1
if "%PORT%"=="" set PORT=8000
set BASE=http://localhost:%PORT%

echo ============================================================
echo  Agent Orchestrator — Normal Endpoint Tests
echo  Target: %BASE%
echo ============================================================
echo.

REM ── Test 1: Health check ────────────────────────────────────────────────
echo [1/7] Health check...
curl -s -X GET %BASE%/health
echo.
echo.

REM ── Test 2: List skills ─────────────────────────────────────────────────
echo [2/7] List discovered skills...
curl -s -X GET %BASE%/skills
echo.
echo.

REM ── Test 3: List agents ─────────────────────────────────────────────────
echo [3/7] List discovered agents...
curl -s -X GET %BASE%/agents
echo.
echo.

REM ── Test 4: Simple query (no tools needed) ──────────────────────────────
echo [4/7] Simple query — "Hello, what can you do?"
curl -s -X POST %BASE%/query ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Hello, what can you do?\", \"session_id\": \"test-session-1\"}"
echo.
echo.

REM ── Test 5: Query that triggers Neo4j tools ─────────────────────────────
echo [5/7] Neo4j schema query — "What labels and relationships are in the graph?"
curl -s -X POST %BASE%/query ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"What labels and relationships are in the graph?\", \"session_id\": \"test-session-2\"}"
echo.
echo.

REM ── Test 6: Follow-up in same session (tests context) ──────────────────
echo [6/7] Follow-up query in same session — "How many nodes of the first label?"
curl -s -X POST %BASE%/query ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"How many nodes of the first label?\", \"session_id\": \"test-session-2\"}"
echo.
echo.

REM ── Test 7: Reload skills ──────────────────────────────────────────────
echo [7/7] Hot-reload skills...
curl -s -X POST %BASE%/skills/reload
echo.
echo.

echo ============================================================
echo  All tests completed.
echo ============================================================
pause
