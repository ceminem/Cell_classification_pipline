@echo off
REM =====================================================================
REM  Cargo Analysis.bat   -   double-click to start
REM =====================================================================
REM
REM  Finds the conda environment with a WORKING GPU and starts the program
REM  in it. You never need to activate an environment yourself.
REM
REM  HOW IT CHOOSES
REM   1. If launcher_environment.txt exists in this folder, the environment
REM      named in it is tried first (the launcher writes this file itself
REM      after finding a GPU environment, so later starts are fast).
REM   2. Otherwise every conda environment is tested in its own separate
REM      process ("conda run"), WITHOUT activating anything here:
REM          GPU + all packages  -> 2      CPU only -> 1      missing -> 0
REM      The highest score wins.
REM   3. The program is started with that environment's OWN python.exe (full
REM      path), so it cannot end up in another environment by accident.
REM
REM  TO FORCE AN ENVIRONMENT
REM   Put its name, e.g.  rocm312 , as the only line of a text file named
REM   launcher_environment.txt next to this .bat. Delete the file to make the
REM   launcher search again. (Do not edit this .bat for that - an edited .bat
REM   blocks updates from GitHub.)
REM
REM  WHY IT NO LONGER ACTIVATES DURING THE SEARCH
REM   The previous version activated each environment inside this script in
REM   turn. It measured rocm312 correctly as the GPU environment and still
REM   started "base": repeated activations inside one script are fragile.
REM   Testing in a separate process leaves this script's state untouched.
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

REM ---- locate conda -------------------------------------------------
set "CONDA_BAT="
for %%P in (
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\anaconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
    "%PROGRAMDATA%\miniconda3\condabin\conda.bat"
    "%PROGRAMDATA%\Anaconda3\condabin\conda.bat"
) do if exist %%P set "CONDA_BAT=%%~P"

if not defined CONDA_BAT (
    echo ERROR: conda.bat not found. Add its path to the list in this file.
    pause & exit /b 1
)
echo Using conda: !CONDA_BAT!
echo.

set "BEST_ENV="
set "BEST_SCORE=0"
set "BEST_GPU=CPU only"
set "BEST_PREFIX="

REM ---- 1. the remembered / chosen environment ------------------------
if exist "%MEMORY%" (
    set "WANT="
    set /p WANT=<"%MEMORY%"
    if defined WANT (
        echo Trying the saved environment: !WANT!
        call :score_env "!WANT!"
        if !SCORE! GEQ 1 (
            set "BEST_ENV=!WANT!"
            set "BEST_SCORE=!SCORE!"
            set "BEST_GPU=!GPUNAME!"
            set "BEST_PREFIX=!PREFIX!"
        ) else (
            echo    not usable - searching all environments instead
        )
    )
)

REM ---- 2. search every named environment -----------------------------
if not defined BEST_ENV (
    echo Checking your conda environments - this takes a little while...
    for /f "usebackq tokens=1" %%E in (`call "!CONDA_BAT!" env list ^| findstr /v /b "#"`) do (
        set "NAME=%%E"
        if "!NAME:\=!"=="!NAME!" (
            call :score_env "!NAME!"
            if !SCORE! GTR !BEST_SCORE! (
                set "BEST_ENV=!NAME!"
                set "BEST_SCORE=!SCORE!"
                set "BEST_GPU=!GPUNAME!"
                set "BEST_PREFIX=!PREFIX!"
            )
        ) else (
            echo    !NAME! - skipped, not a named conda environment
        )
    )
    if !BEST_SCORE! EQU 2 (
        >"%MEMORY%" echo !BEST_ENV!
        echo    remembered !BEST_ENV! in launcher_environment.txt for next time
    )
)

if not defined BEST_ENV (
    echo.
    echo ============================================================
    echo  No conda environment has everything the analysis needs.
    echo  Install what is missing into your GPU environment, e.g.:
    echo      conda activate rocm312
    echo      pip install cellpose tifffile pillow scipy matplotlib
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

REM Activate for the environment's PATH (DLLs), then start ITS python.exe
REM by full path so the program cannot run in a different environment.
call "!CONDA_BAT!" activate !BEST_ENV! >nul 2>&1
set "PY=python"
if defined BEST_PREFIX if exist "!BEST_PREFIX!\python.exe" set "PY=!BEST_PREFIX!\python.exe"
echo Starting with !PY!
"!PY!" cargo_gui.py
if errorlevel 1 (
    echo.
    echo Exited with an error - see above.
    pause
)
endlocal
exit /b 0


REM ---- helper: test one environment in a separate process -------------
REM Sets SCORE (0/1/2), GPUNAME, PREFIX. Nothing is activated here.
:score_env
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
if not defined STATUS (
    echo    %~1 - could not run Python
    exit /b 0
)
if "!STATUS!"=="MISSING" (
    echo    %~1 - missing: !INFO!
    exit /b 0
)
if "!INFO:~0,4!"=="GPU " (
    set "SCORE=2"
    set "GPUNAME=!INFO:~4!"
    echo    %~1 - OK, GPU: !GPUNAME!
) else (
    set "SCORE=1"
    echo    %~1 - OK, CPU only
)
exit /b 0
