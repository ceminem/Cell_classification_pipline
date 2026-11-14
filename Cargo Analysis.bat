@echo off
REM =====================================================================
REM  Cargo Analysis.bat   -   double-click to start
REM =====================================================================
REM
REM  CHANGE THE SCRIPT FOLDER
REM  ------------------------
REM  By default the scripts must sit in the SAME folder as this .bat.
REM  To keep this file elsewhere (Desktop, say), set the path here:
REM
set "SCRIPT_DIR="
REM      example:  set "SCRIPT_DIR=C:\Users\cemin\Documents\Molbio\analysis_pipline"
REM
REM  FORCE A PARTICULAR ENVIRONMENT
REM  ------------------------------
REM  Leave blank and it searches every conda environment, PREFERRING one
REM  with a working GPU. Set it to skip the search entirely:
REM
set "PREFERRED_ENV="
REM      example:  set "PREFERRED_ENV=rocm312"
REM
REM =====================================================================

setlocal EnableDelayedExpansion

if defined SCRIPT_DIR (
    cd /d "%SCRIPT_DIR%" 2>nul
    if errorlevel 1 (
        echo ERROR: SCRIPT_DIR does not exist: %SCRIPT_DIR%
        pause & exit /b 1
    )
) else (
    cd /d "%~dp0"
)

if not exist "cargo_gui.py" (
    echo.
    echo ERROR: cargo_gui.py is not in  %CD%
    echo Move the scripts next to this file, or set SCRIPT_DIR above.
    echo.
    pause & exit /b 1
)

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

REM ---- find the BEST environment, not merely the first --------------
REM  A previous version stopped at the first environment that imported
REM  everything - which was "base", holding a CPU-only torch. It never
REM  reached the ROCm environment. So now every environment is scored:
REM      2 = has all packages AND a working GPU
REM      1 = has all packages, CPU only
REM  and the highest score wins. GPU beats convenience.
set "BEST_ENV="
set "BEST_SCORE=0"
set "BEST_GPU=CPU only"

if defined PREFERRED_ENV (
    call :score_env "%PREFERRED_ENV%"
    if !SCORE! GTR 0 (
        set "BEST_ENV=%PREFERRED_ENV%"
        set "BEST_SCORE=!SCORE!"
        set "BEST_GPU=!GPUNAME!"
    ) else (
        echo PREFERRED_ENV "%PREFERRED_ENV%" is not usable - searching instead.
    )
)

if not defined BEST_ENV (
    echo Checking your conda environments...
    for /f "usebackq tokens=1" %%E in (`call "!CONDA_BAT!" env list ^| findstr /v "^#"`) do (
        set "NAME=%%E"
        REM skip full paths - conda lists some environments by directory
        echo !NAME! | findstr /c:"\" >nul
        if errorlevel 1 (
            call :score_env "!NAME!"
            if !SCORE! GTR !BEST_SCORE! (
                set "BEST_ENV=!NAME!"
                set "BEST_SCORE=!SCORE!"
                set "BEST_GPU=!GPUNAME!"
            )
        )
    )
)

if not defined BEST_ENV (
    echo.
    echo ============================================================
    echo  No conda environment has everything the analysis needs.
    echo.
    echo  Activate the one you want and install what is missing:
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
    echo  NOTE: no GPU environment was found, so segmentation will run
    echo        on the CPU and take several minutes per image.
)
echo.
call "!CONDA_BAT!" activate !BEST_ENV! >nul 2>&1
python cargo_gui.py
if errorlevel 1 (
    echo.
    echo Exited with an error - see above.
    pause
)
endlocal
exit /b 0

REM ---- helper: score one environment --------------------------------
:score_env
set "SCORE=0"
set "GPUNAME=CPU only"
call "!CONDA_BAT!" activate %~1 >nul 2>&1
if errorlevel 1 (
    echo    %~1 - cannot activate
    exit /b 0
)
set "MISS="
for %%M in (cellpose torch tifffile PIL scipy numpy) do (
    python -c "import %%M" >nul 2>&1
    if errorlevel 1 set "MISS=!MISS! %%M"
)
if defined MISS (
    echo    %~1 - missing:!MISS!
    exit /b 0
)
REM everything imports; now ask whether the GPU actually works
for /f "usebackq delims=" %%G in (`python -c "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2^>nul`) do set "GPUNAME=%%G"
if "!GPUNAME!"=="" (
    set "GPUNAME=CPU only"
    set "SCORE=1"
    echo    %~1 - OK, CPU only
) else (
    set "SCORE=2"
    echo    %~1 - OK, GPU: !GPUNAME!
)
exit /b 0
