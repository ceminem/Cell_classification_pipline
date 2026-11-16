@echo off
REM =====================================================================
REM  Cargo Analysis.bat   -   double-click to start
REM =====================================================================
REM
REM  Finds the Python environment that can run the analysis (preferring one
REM  with a WORKING GPU) and starts the program in it. You never need to
REM  activate an environment yourself. Works on any Windows PC: the lab PC
REM  with conda + AMD GPU, or a laptop with or without conda.
REM
REM  HOW IT CHOOSES
REM   1. conda is searched in many places (toolsind_conda.bat): the PATH,
REM      the registry, the usual Anaconda / Miniconda / Miniforge folders.
REM      For an unusual place, write the conda folder (e.g.
REM      D:\Programs\miniconda3) as the only line of conda_path.txt here.
REM      No conda at all is fine too.
REM   2. Candidates are tested, each in its own separate process:
REM          GPU + all packages  -> 2      CPU only -> 1      missing -> 0
REM       - .venv in this folder (made by "Setup on this PC.bat" without conda)
REM       - the environment in launcher_environment.txt (remembered choice)
REM       - otherwise every conda environment
REM      The highest score wins and is remembered in launcher_environment.txt
REM      so later starts are fast. Delete that file to search again.
REM   3. Nothing usable on this PC: it offers to run "Setup on this PC.bat",
REM      which installs everything once, then starts the program.
REM   4. The program is started with that environment's OWN python.exe (full
REM      path), so it cannot end up in another environment by accident.
REM
REM  TO FORCE AN ENVIRONMENT
REM   Put its name, e.g.  rocm312 , as the only line of a text file named
REM   launcher_environment.txt next to this .bat. (Do not edit this .bat for
REM   that - an edited .bat blocks updates from GitHub.)
REM =====================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "cargo_gui.py" (
    echo.
    echo ERROR: cargo_gui.py is not in  %CD%
    echo Keep this .bat inside the program folder. For a desktop icon use
    echo right-click - Send to - Desktop ^(create shortcut^).
    echo.
    pause & exit /b 1
)
if not exist "tools\env_probe.py" (
    echo ERROR: tools\env_probe.py is missing - get the latest version from GitHub.
    pause & exit /b 1
)
set "PROBE=%CD%\tools\env_probe.py"
set "MEMORY=%CD%\launcher_environment.txt"

set "TRIED_SETUP="
:search
REM ---- locate conda (optional) -----------------------------------------
call "%~dp0tools\find_conda.bat"
if defined CONDA_BAT (
    echo Using conda: !CONDA_BAT!
) else (
    echo conda: not installed on this PC ^(not needed^)
)
echo.

set "BEST_ENV="
set "BEST_SCORE=0"
set "BEST_GPU=CPU only"
set "BEST_PREFIX="
set "BEST_PY="
set "MEM_OK="

REM ---- 0. an environment inside this folder (.venv) -------------------
if exist "!CD!\.venv\Scripts\python.exe" (
    echo Checking the environment in this folder ^(.venv^)
    call :score_py "!CD!\.venv\Scripts\python.exe" .venv
    if !SCORE! GTR !BEST_SCORE! (
        set "BEST_ENV=.venv"
        set "BEST_SCORE=!SCORE!"
        set "BEST_GPU=!GPUNAME!"
        set "BEST_PY=!CD!\.venv\Scripts\python.exe"
    )
)

REM ---- 1. the remembered / chosen conda environment -----------------
if defined CONDA_BAT if exist "%MEMORY%" (
    set "WANT="
    set /p WANT=<"%MEMORY%"
    if defined WANT if /i not "!WANT!"==".venv" (
        echo Trying the saved environment: !WANT!
        call :score_env "!WANT!"
        if !SCORE! GEQ 1 (
            set "MEM_OK=1"
            if !SCORE! GTR !BEST_SCORE! (
                set "BEST_ENV=!WANT!"
                set "BEST_SCORE=!SCORE!"
                set "BEST_GPU=!GPUNAME!"
                set "BEST_PREFIX=!PREFIX!"
                set "BEST_PY="
            )
        ) else (
            echo    not usable - searching all environments instead
        )
    )
)

REM ---- 2. search every named conda environment -----------------------
if defined CONDA_BAT if not defined MEM_OK if !BEST_SCORE! LSS 2 (
    echo Checking your conda environments - this takes a little while...
    for /f "usebackq tokens=1 eol=#" %%E in (`call "!CONDA_BAT!" env list`) do (
        set "NAME=%%E"
        if "!NAME:\=!"=="!NAME!" (
            call :score_env "!NAME!"
            if !SCORE! GTR !BEST_SCORE! (
                set "BEST_ENV=!NAME!"
                set "BEST_SCORE=!SCORE!"
                set "BEST_GPU=!GPUNAME!"
                set "BEST_PREFIX=!PREFIX!"
                set "BEST_PY="
            )
        ) else (
            echo    !NAME! - skipped, not a named conda environment
        )
    )
    if defined BEST_ENV if /i not "!BEST_ENV!"==".venv" (
        >"%MEMORY%" echo !BEST_ENV!
        echo    remembered !BEST_ENV! in launcher_environment.txt for next time
    )
)

REM ---- 3. nothing usable: offer the one-time setup --------------------
if not defined BEST_ENV if not defined TRIED_SETUP (
    echo.
    echo ============================================================
    echo  This PC has no Python environment with Cellpose yet.
    echo  "Setup on this PC.bat" can install everything once
    echo  ^(about 1-3 GB download, 5-20 minutes^).
    echo ============================================================
    choice /c YN /m "Set it up now"
    if not errorlevel 2 (
        set "TRIED_SETUP=1"
        call "%~dp0Setup on this PC.bat" /fromlauncher
        if not errorlevel 1 (
            echo.
            goto :search
        )
    )
)

if not defined BEST_ENV (
    echo.
    echo ============================================================
    echo  No environment has everything the analysis needs.
    echo  Run "Setup on this PC.bat", or install what is missing
    echo  into your environment, e.g.:
    echo      conda activate rocm312
    echo      pip install -r requirements.txt
    echo ============================================================
    echo.
    pause & exit /b 1
)

echo.
echo ------------------------------------------------------------
echo  Environment : !BEST_ENV!
echo  Compute     : !BEST_GPU!
echo ------------------------------------------------------------
if !BEST_SCORE! EQU 1 (
    echo  NOTE: no environment with a working GPU was found, so the
    echo        analysis runs on the CPU and takes longer.
)
echo.

REM conda: activate for the environment's PATH (DLLs), then start ITS
REM python.exe by full path so the program cannot run in a different one.
REM .venv: its python.exe directly, nothing to activate.
if defined BEST_PY (
    set "PY=!BEST_PY!"
) else (
    call "!CONDA_BAT!" activate !BEST_ENV! >nul 2>&1
    set "PY=python"
    if defined BEST_PREFIX if exist "!BEST_PREFIX!\python.exe" set "PY=!BEST_PREFIX!\python.exe"
)
echo Starting with !PY!
"!PY!" cargo_gui.py
if errorlevel 1 (
    echo.
    echo Exited with an error - see above.
    pause
)
endlocal
exit /b 0


REM ---- helper: test a python.exe directly (.venv) ----------------------
REM Sets SCORE, GPUNAME, PREFIX like :score_env.
:score_py
set "LABEL=%~2"
set "SCORE=0"
set "STATUS="
set "GPUNAME=CPU only"
set "PREFIX="
set "INFO="
for /f "usebackq tokens=1-4 delims=|" %%A in (`call "%~1" "!PROBE!" 2^>nul`) do (
    if "%%A"=="PROBE" (
        set "STATUS=%%B"
        set "INFO=%%C"
        set "PREFIX=%%D"
    )
)
goto :interpret

REM ---- helper: test one environment in a separate process -------------
REM Sets SCORE (0/1/2), GPUNAME, PREFIX. Nothing is activated here.
:score_env
set "LABEL=%~1"
set "SCORE=0"
set "STATUS="
set "GPUNAME=CPU only"
set "PREFIX="
set "INFO="
for /f "usebackq tokens=1-4 delims=|" %%A in (`call "!CONDA_BAT!" run -n %~1 python "!PROBE!" 2^>nul`) do (
    if "%%A"=="PROBE" (
        set "STATUS=%%B"
        set "INFO=%%C"
        set "PREFIX=%%D"
    )
)
goto :interpret

REM ---- shared: turn the probe line into SCORE / GPUNAME ----------------
:interpret
if not defined STATUS (
    echo    !LABEL! - could not run Python
    exit /b 0
)
if "!STATUS!"=="MISSING" (
    echo    !LABEL! - missing: !INFO!
    exit /b 0
)
if "!INFO:~0,4!"=="GPU " (
    set "SCORE=2"
    set "GPUNAME=!INFO:~4!"
    echo    !LABEL! - OK, GPU: !GPUNAME!
) else (
    set "SCORE=1"
    echo    !LABEL! - OK, CPU only
)
exit /b 0
