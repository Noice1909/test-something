@echo off
REM ============================================================================
REM  test_stream.bat — Test the /query/stream SSE endpoint (token-by-token)
REM  Usage: test_stream.bat [port]    (default port: 8000)
REM
REM  SSE event types:
REM    token       — individual LLM output token
REM    tool_start  — before a tool executes
REM    tool_end    — after a tool completes
REM    hook_skip   — a hook skipped a tool call
REM    hook_block  — a hook blocked a tool call
REM    done        — final response with session info
REM    error       — something went wrong
REM ============================================================================

setlocal enabledelayedexpansion

set PORT=%1
if "%PORT%"=="" set PORT=8000
set BASE=http://localhost:%PORT%

echo ============================================================
echo  Agent Orchestrator — Streaming Endpoint Tests
echo  Target: %BASE%/query/stream
echo  (tokens stream in real-time via SSE)
echo ============================================================
echo.

REM ── Test 1: Simple streaming — no tools ─────────────────────────────────
echo [1/4] Simple streaming — "Say hello in 3 different languages"
echo      (watch tokens arrive one by one)
echo ---
curl -s -N -X POST %BASE%/query/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Say hello in 3 different languages, keep it brief.\", \"session_id\": \"stream-test-1\"}"
echo.
echo ---
echo.

REM ── Test 2: Streaming with tool use ─────────────────────────────────────
echo [2/4] Streaming with tools — "What is the graph schema?"
echo      (expect: tokens + tool_start + tool_end + more tokens + done)
echo ---
curl -s -N -X POST %BASE%/query/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"What is the graph schema?\", \"session_id\": \"stream-test-2\"}"
echo.
echo ---
echo.

REM ── Test 3: Streaming with search ───────────────────────────────────────
echo [3/4] Streaming with search — "Search for any person in the graph"
echo      (expect: tokens + tool calls + results streaming)
echo ---
curl -s -N -X POST %BASE%/query/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Search for any person in the graph\", \"session_id\": \"stream-test-3\"}"
echo.
echo ---
echo.

REM ── Test 4: Streaming with hook trigger (write attempt) ─────────────────
echo [4/4] Hook test — asking to create a node (should be blocked by safety hook)
echo      (expect: hook_block event from safety_write_blocker)
echo ---
curl -s -N -X POST %BASE%/query/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Create a new node with label TestNode and name property set to 'hello'\", \"session_id\": \"stream-test-4\"}"
echo.
echo ---
echo.

echo ============================================================
echo  All streaming tests completed.
echo ============================================================
pause
