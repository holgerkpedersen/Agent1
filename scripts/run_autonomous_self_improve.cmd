@echo off
rem =====================================================================
rem Launcher for the Windows scheduled task "\Agent1\Autonomous Self Improve".
rem
rem WHY THIS EXISTS
rem ---------------
rem The task previously failed with ResultCode 2147942402 (0x80070002 =
rem ERROR_FILE_NOT_FOUND). That is NOT a bug in the Python driver
rem (scripts/autonomous_self_improve.py is fine and self-locates REPO_ROOT
rem via __file__). It means the scheduled task's ACTION pointed at a path
rem that did not resolve under the account it runs as.
rem
rem This launcher uses ONLY absolute paths and an explicit interpreter, so
rem it never depends on the calling account's PATH or working directory.
rem =====================================================================

setlocal EnableExtensions
rem Use UTF-8 so the driver's unicode output (em-dashes, etc.) logs cleanly.
chcp 65001 >nul 2>&1

rem --- Absolute locations (edit only if you move the repo) ---------------
set "REPO_ROOT=C:\Dev\Agent1"
set "PYTHON_EXE=C:\Program Files\Python312\python.exe"
set "DRIVER=%REPO_ROOT%\scripts\autonomous_self_improve.py"
set "LOG_DIR=%REPO_ROOT%\reports\harnessfix"
set "LOG_FILE=%LOG_DIR%\scheduled_run.log"

rem --- Sanity checks (fail loudly with a clear message, not 0x80070002) ---
if not exist "%PYTHON_EXE%" (
    echo [launcher] FATAL: python not found at "%PYTHON_EXE%" >&2
    exit /b 9009
)
if not exist "%DRIVER%" (
    echo [launcher] FATAL: driver not found at "%DRIVER%" >&2
    exit /b 9009
)

rem --- Environment the driver needs --------------------------------------
set "AGENT_AUTONOMOUS=1"
set "PYTHONUNBUFFERED=1"

rem --- Run -----------------------------------------------------------------
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

rem The driver runs `git` and writes reports/harnessfix/* RELATIVE to the
rem current directory, so we MUST be in the repo root. Do NOT rely on the
rem task's "Start in" being set correctly (it often isn't under System).
cd /d "%REPO_ROOT%" || (
    echo [launcher] FATAL: cannot cd to "%REPO_ROOT%" >&2
    exit /b 9009
)

rem NOTE on --model: it routes fix-generation AND the benchmark gate to a LIVE
rem LLM API (qwen/qwen3.8-27b via LM Studio). If that server is NOT running the
rem run will fail/hang at the benchmark gate. It does NOT change the outcome
rem here: the catalog repairs are already applied, so the run ends in
rem no_repair_catalogued. Drop --model (or add --no-benchmark) for the
rem offline-only gate when the server is down.
echo [%date% %time%] [launcher] starting autonomous self-improve >> "%LOG_FILE%"
"%PYTHON_EXE%" "%DRIVER%" --auto --max-iterations 5 --traces "%REPO_ROOT%\reports\traces" --model qwen/qwen3.8-27b --profile deep-analysis >> "%LOG_FILE%" 2>&1
set "RC=%ERRORLEVEL%"
echo [%date% %time%] [launcher] finished with exit code %RC% >> "%LOG_FILE%"

exit /b %RC%
