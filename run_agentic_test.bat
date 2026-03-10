@echo off
echo ============================================
echo  Agentic Graph Query System — Batch Test
echo ============================================
echo.

cd /d "%~dp0"

echo [1] Checking Python...
python --version 2>NUL
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

echo [2] Running batch tests...
echo.
python test_agentic_batch.py
echo.
echo Done.
pause
