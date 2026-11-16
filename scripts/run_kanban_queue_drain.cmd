@echo off
rem =====================================================================
rem Launcher for the Windows scheduled task "\Agent1\Kanban Queue Drain".
rem
rem WHY THIS EXISTS
rem ---------------
rem The Kanban -> Agent1 queue was only ever drained by a background poller
rem that starts INSIDE an interactive agent session
rem (agent_core/run_utils.py -> start_inbound_processor).  When no agent
rem REPL is open, nothing reads the queue at all: messages pile up, the
rem offset stops moving, and the board and the agent drift apart.  Several
rem sessions also meant several pollers on the SAME directory, every one of
rem them applying the same lines - which is where the duplicated ids in
rem messages.jsonl came from.
rem
rem The fix has two halves and this script is the second one:
rem
rem   1. harnessfix.kanban_bridge.process_inbound() is the ONLY queue
rem      handler, and it now holds a cross-process lock (handler.lock in the
rem      queue directory), so a second handler steps aside instead of
rem      applying the same messages again.  See handling_lock().
rem   2. This launcher lets Task Scheduler be that single handler while no
rem      interactive agent runs.  Because of (1) it can safely coexist with
rem      a running session: it WAITS for the lock (--timeout), and if it
rem      never gets it, it exits 0 having applied nothing.
rem
rem It is a one-shot drain, not a daemon: an empty queue does nothing and
rem exits immediately, so a frequent schedule costs nothing.
rem
rem ABSOLUTE PATHS ONLY
rem -------------------
rem Same lesson as run_autonomous_self_improve.cmd: a task action that
rem depends on the calling account's PATH or working directory fails with
rem 0x80070002 (ERROR_FILE_NOT_FOUND).  Every path here is absolute.
rem =====================================================================

setlocal EnableExtensions
rem UTF-8 so the Python log (em-dashes, card titles) is readable.
chcp 65001 >nul 2>&1

rem --- Absolute locations (edit only if you move the repo) ---------------
set "REPO_ROOT=C:\Dev\Agent1"
set "PYTHON_EXE=C:\Program Files\Python312\python.exe"
set "LOG_DIR=%REPO_ROOT%\reports\harnessfix"
set "LOG_FILE=%LOG_DIR%\kanban_queue_drain.log"

rem Seconds to wait for the single handler lock before giving up.  Covers a
rem running agent session mid-drain; a poller holds the lock for milliseconds,
rem so this is generous and never stalls the schedule.
set "LOCK_WAIT_SECONDS=120"

rem --- Sanity checks (fail with a clear code, not 0x80070002) ------------
if not exist "%PYTHON_EXE%" (
    echo [drain] FATAL: python not found at "%PYTHON_EXE%" >&2
    exit /b 9009
)
if not exist "%REPO_ROOT%\harnessfix\kanban_bridge.py" (
    echo [drain] FATAL: harnessfix\kanban_bridge.py not found under "%REPO_ROOT%" >&2
    exit /b 9009
)

rem --- Environment ---------------------------------------------------------
set "PYTHONUNBUFFERED=1"
rem Card titles and message bodies are unicode; without this a console
rem codepage mismatch turns the log into mojibake.
set "PYTHONIOENCODING=utf-8"

rem --- Run -----------------------------------------------------------------
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

rem `-m` resolves the package from the current directory, so the cwd MUST be
rem the repo root.  Do not rely on the task's "Start in" being honoured.
cd /d "%REPO_ROOT%" || (
    echo [drain] FATAL: cannot cd to "%REPO_ROOT%" >&2
    exit /b 9009
)

echo [%date% %time%] [drain] start >> "%LOG_FILE%"
"%PYTHON_EXE%" -m harnessfix.kanban_bridge --process-in --timeout %LOCK_WAIT_SECONDS% >> "%LOG_FILE%" 2>&1
set "RC=%ERRORLEVEL%"
echo [%date% %time%] [drain] exit %RC% >> "%LOG_FILE%"

exit /b %RC%
