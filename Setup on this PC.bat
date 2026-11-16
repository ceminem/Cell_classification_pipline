@echo off
REM =====================================================================
REM  Setup on this PC.bat   -   run once on a new computer (e.g. a laptop)
REM =====================================================================
REM
REM  Installs everything the analysis needs into ONE environment, so that
REM  "Cargo Analysis.bat" can start it. Nothing here touches existing
REM  environments.
REM
REM   - conda installed (Anaconda / Miniconda / Miniforge):
REM         creates the conda environment  cargo  (Python 3.11)
REM   - no conda, but Python 3.10-3.12 installed:
REM         creates  .venv  inside this program folder
REM   - neither: offers to install Python 3.12 with winget (user only,
REM         no admin rights), then creates .venv
REM
REM  Then: pip installs Cellpose, PyTorch and the rest (requirements.txt).
REM  An NVIDIA card gets the CUDA build of PyTorch; every other computer
REM  (Intel / AMD laptop graphics) runs on the CPU, which works, only slower.
REM  Downloads about 1-3 GB; takes 5-20 minutes.
REM
REM  "Cargo Analysis.bat" offers to run this file itself when it finds no
REM  usable environment, so normally you never start it by hand.
REM =====================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "QUIET="
if /i "%~1"=="/fromlauncher" set "QUIET=1"

if not exist "requirements.txt" (
    echo ERROR: requirements.txt is not in %CD% - keep this .bat in the program folder.
    goto :fail
)

echo.
echo ============================================================
echo  Setting up the cargo analysis on this computer
echo ============================================================
echo.

call "%~dp0tools\find_conda.bat"
set "PY="

if defined CONDA_BAT (
    echo Found conda: !CONDA_BAT!
    call "!CONDA_BAT!" run -n cargo python -c "pass" >nul 2>&1
    if errorlevel 1 (
        echo Creating the conda environment "cargo" with Python 3.11 ...
        call "!CONDA_BAT!" create -y -n cargo python=3.11
        if errorlevel 1 goto :fail
    ) else (
        echo The conda environment "cargo" exists already - completing it.
    )
    for /f "usebackq delims=" %%P in (`call "!CONDA_BAT!" run -n cargo python -c "import sys; print(sys.executable)"`) do set "PY=%%P"
    goto :install
)

echo conda is not installed here - using a normal Python instead.
call :find_python
if not defined PYBASE (
    echo.
    echo No Python 3.10 - 3.12 found on this computer.
    where winget >nul 2>&1
    if errorlevel 1 goto :nopython
    choice /c YN /m "Install Python 3.12 now (for your user only, no admin rights needed)"
    if errorlevel 2 goto :nopython
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 goto :nopython
    REM this window does not see the new PATH yet - use the install folder directly
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYBASE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    if not defined PYBASE call :find_python
    if not defined PYBASE goto :nopython
)
echo Using Python: !PYBASE!
if not exist ".venv\Scripts\python.exe" (
    echo Creating the environment .venv in this folder ...
    "!PYBASE!" -m venv .venv
    if errorlevel 1 goto :fail
)
set "PY=%CD%\.venv\Scripts\python.exe"

:install
if not defined PY goto :fail
if not exist "!PY!" goto :fail
echo.
echo Installing the packages into  !PY!
echo (this downloads a lot the first time - please wait)
echo.
"!PY!" -m pip install --upgrade pip
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo NVIDIA graphics card found - installing the CUDA build of PyTorch.
    "!PY!" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 echo    CUDA build failed - continuing with the normal ^(CPU^) build.
)
"!PY!" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Checking the result ...
for /f "usebackq tokens=1-3 delims=|" %%A in (`call "!PY!" "%CD%\tools\env_probe.py"`) do (
    if "%%B"=="OK" (
        echo.
        echo ============================================================
        echo  Ready.  Compute: %%C
        echo  Start the program with  "Cargo Analysis.bat".
        echo ============================================================
        echo.
        if not defined QUIET pause
        endlocal & exit /b 0
    )
    if "%%B"=="MISSING" echo Still missing: %%C
)
goto :fail

REM ---------------------------------------------------------------------
:find_python
set "PYBASE="
REM the py launcher (python.org installs)
for %%V in (3.12 3.11 3.10) do (
    if not defined PYBASE (
        py -%%V -c "pass" >nul 2>&1
        if not errorlevel 1 for /f "usebackq delims=" %%E in (`py -%%V -c "import sys; print(sys.executable)"`) do set "PYBASE=%%E"
    )
)
REM python on the PATH (the Microsoft-Store placeholder fails this test)
if not defined PYBASE (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
    if not errorlevel 1 for /f "usebackq delims=" %%E in (`python -c "import sys; print(sys.executable)"`) do set "PYBASE=%%E"
)
REM usual python.org folders
if not defined PYBASE for %%V in (312 311 310) do (
    if not defined PYBASE if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PYBASE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
)
exit /b 0

:nopython
echo.
echo ============================================================
echo  Please install ONE of these, then run this setup again:
echo   - Python 3.12 from https://www.python.org/downloads/
echo     ^(tick "Add python.exe to PATH" in the installer^), or
echo   - Miniforge from https://conda-forge.org/download/
echo ============================================================
echo.
if not defined QUIET pause
endlocal & exit /b 1

:fail
echo.
echo ============================================================
echo  Setup did not finish - see the messages above.
echo  Internet connection needed; running it again continues
echo  where it stopped.
echo ============================================================
echo.
if not defined QUIET pause
endlocal & exit /b 1
