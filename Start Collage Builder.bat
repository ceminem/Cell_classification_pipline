@echo off
REM ===================================================================
REM  Start_Collage_Builder.bat
REM
REM  Launches the Collage Builder GUI.
REM
REM  This tool needs ONLY Python and Pillow - no Fiji, no Cellpose, no
REM  conda. It therefore runs on an ordinary office PC. The script tries
REM  the analysis pipeline's own environment (.venv) first, then a plain
REM  Python, then a conda environment if one is present.
REM
REM  In the analysis pipeline the same builder is the "Collage" tab of
REM  "Cargo Analysis.bat" - this file is only for PCs without the pipeline.
REM ===================================================================

setlocal
cd /d "%~dp0"

REM --- 0. the pipeline's environment, if "Setup on this PC.bat" made one ---
if exist ".venv\Scripts\python.exe" (
    echo Using the pipeline environment .venv
    ".venv\Scripts\python.exe" collage_gui.py
    goto :end
)

REM --- 1. try whatever "python" is on PATH ---
python --version >nul 2>&1
if not errorlevel 1 (
    python -c "import PIL" >nul 2>&1
    if not errorlevel 1 (
        echo Using Python from PATH.
        python collage_gui.py
        goto :end
    )
    echo Python found, but Pillow is missing. Installing it now...
    python -m pip install --user pillow
    if not errorlevel 1 (
        python collage_gui.py
        goto :end
    )
)

REM --- 2. fall back to the py launcher (installed with python.org builds) ---
py -3 --version >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import PIL" >nul 2>&1
    if errorlevel 1 py -3 -m pip install --user pillow
    echo Using the py launcher.
    py -3 collage_gui.py
    goto :end
)

REM --- 3. fall back to a conda environment, if one exists ---
set CONDA_BAT=
for %%P in (
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\anaconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
    "%PROGRAMDATA%\miniconda3\condabin\conda.bat"
    "%PROGRAMDATA%\Anaconda3\condabin\conda.bat"
) do (
    if exist %%P set CONDA_BAT=%%P
)

if not "%CONDA_BAT%"=="" (
    echo Using conda: %CONDA_BAT%
    call %CONDA_BAT% activate cargo 2>nul || call %CONDA_BAT% activate cellpose
    python collage_gui.py
    goto :end
)

echo.
echo ============================================================
echo  No Python installation was found.
echo.
echo  Install Python from https://www.python.org/downloads/
echo  and TICK "Add Python to PATH" during setup.
echo.
echo  Then double-click this file again - it will install the one
echo  package it needs (Pillow) automatically.
echo ============================================================
echo.
pause

:end
if errorlevel 1 (
    echo.
    echo The program exited with an error - the message above should say why.
    pause
)
endlocal
