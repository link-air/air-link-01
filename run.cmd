@echo off
chcp 65001 >nul
setlocal

rem ---------------------------------------------------------------
rem  Double-click to run -- no command line needed.
rem  PURE ASCII ON PURPOSE: cmd.exe on a zh-CN (GBK) code page
rem  mis-parses multibyte characters, and after `chcp 65001` it
rem  is known to start executing from the middle of a line.
rem    run.cmd            open the memory workbench (web UI, recommended)
rem    run.cmd <other>    pass through to demo.py (script demo, no API key)
rem    run.cmd test       run the tests
rem    run.cmd exp        replay a dialogue experiment (report)
rem    run.cmd distill    stage-2 demo (distill and convergence)
rem    run.cmd memo       stage-3 demo (memos and mirror)
rem    run.cmd trend      stage-4 demo (drift detection)
rem    run.cmd fresh      reset the demo DB, then run
rem    run.cmd real       use the real LLM (configure env vars first, see README)
rem ---------------------------------------------------------------

cd /d "%~dp0"
set "PY="

rem Candidate 1: python from PATH.
rem NOTE: merely "found or not" is not enough -- the Windows Store
rem python.exe placeholder is on PATH too: it exists, but running it
rem pops the Store. Every candidate must actually run once to count.
for /f "delims=" %%i in ('where python 2^>nul') do call :try "%%i"

rem Candidate 2: a known local install path (this dev machine). :try skips if PY is set.
call :try "%USERPROFILE%\.workbuddy\binaries\python\versions\3.14.3\python.exe"

if not defined PY (
    echo [ERROR] No usable Python found. Install Python 3.11+ and retry.
    pause
    exit /b 1
)

if /i "%~1"=="test" (
    "%PY%" -m unittest discover -s tests -v
) else if /i "%~1"=="exp" (
    rem Experiment scripts are yours to supply (not shipped) -- they live
    rem under private\ ; usage:  run.cmd exp private\<script>.json
    "%PY%" run_experiment.py --script "%~2" --fresh
) else if /i "%~1"=="ui" (
    "%PY%" -m core.dashboard
) else if /i "%~1"=="distill" (
    "%PY%" demo_distill.py
) else if /i "%~1"=="memo" (
    "%PY%" demo_memo.py
) else if /i "%~1"=="trend" (
    "%PY%" demo_trend.py
) else if /i "%~1"=="fresh" (
    "%PY%" demo.py --fresh
) else if /i "%~1"=="real" (
    "%PY%" demo.py --real
) else if "%~1"=="" (
    rem no arguments -- open the workbench (the most common entry)
    "%PY%" -m core.dashboard
) else (
    "%PY%" demo.py %*
)

echo.
pause
exit /b 0

:try
if defined PY goto :eof
if not exist %1 goto :eof
%1 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
goto :eof
