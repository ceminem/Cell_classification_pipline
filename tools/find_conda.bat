@echo off
REM =====================================================================
REM  tools\find_conda.bat  -  called by "Cargo Analysis.bat" and
REM  "Setup on this PC.bat". Sets CONDA_BAT to the full path of conda.bat,
REM  or leaves it empty when conda is not installed. Prints nothing.
REM
REM  Where it looks, in this order:
REM   1. conda_path.txt in the program folder (one line: the conda folder,
REM      e.g. C:\Users\me\miniconda3) - for unusual install locations
REM   2. CONDA_EXE (set when conda was initialised for this user)
REM   3. conda on the PATH
REM   4. the Windows registry (Anaconda / Miniconda installers register here)
REM   5. the usual install folders of Anaconda, Miniconda, Miniforge and
REM      Mambaforge, for one user or all users, on C: and D:
REM  No setlocal here on purpose: CONDA_BAT must survive the return.
REM =====================================================================
set "CONDA_BAT="
set "FC_PROG=%~dp0.."

REM 1. explicit path written by the user
REM    (one statement per line, so plain %-expansion sees each new value)
set "FC_LINE="
if exist "%FC_PROG%\conda_path.txt" set /p FC_LINE=<"%FC_PROG%\conda_path.txt"
if defined FC_LINE set "FC_LINE=%FC_LINE:"=%"
if defined FC_LINE call :fc_any "%FC_LINE%"

REM 2. CONDA_EXE  (...\Scripts\conda.exe)
if not defined CONDA_BAT if defined CONDA_EXE call :fc_file "%CONDA_EXE%"

REM 3. on the PATH
if not defined CONDA_BAT for /f "delims=" %%W in ('where conda.bat 2^>nul') do if not defined CONDA_BAT call :fc_file "%%W"
if not defined CONDA_BAT for /f "delims=" %%W in ('where conda.exe 2^>nul') do if not defined CONDA_BAT call :fc_file "%%W"

REM 4. registry: every InstallPath under Software\Python (Anaconda, Miniconda,
REM    python.org ...); only folders that contain condabin\conda.bat count
if not defined CONDA_BAT call :fc_reg HKCU
if not defined CONDA_BAT call :fc_reg HKLM

REM 5. usual folders
if not defined CONDA_BAT for %%R in ("%USERPROFILE%" "%LOCALAPPDATA%" "%LOCALAPPDATA%\Continuum" "%PROGRAMDATA%" "%ProgramFiles%" "C:" "D:" "C:\tools" "%USERPROFILE%\Documents") do (
    for %%N in (miniconda3 anaconda3 miniforge3 mambaforge Miniconda Anaconda Miniforge) do (
        if not defined CONDA_BAT if exist "%%~R\%%N\condabin\conda.bat" set "CONDA_BAT=%%~R\%%N\condabin\conda.bat"
    )
)
REM    scoop installs
if not defined CONDA_BAT for %%N in (miniconda3 anaconda3 miniforge) do (
    if not defined CONDA_BAT if exist "%USERPROFILE%\scoop\apps\%%N\current\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\scoop\apps\%%N\current\condabin\conda.bat"
)
set "FC_LINE="
set "FC_PROG="
exit /b 0

REM ---- a path that is either the conda folder or a file inside it ------
:fc_any
if exist "%~1\condabin\conda.bat" ( set "CONDA_BAT=%~1\condabin\conda.bat" & exit /b 0 )
if exist "%~1" if /i "%~x1"==".bat" ( set "CONDA_BAT=%~f1" & exit /b 0 )
if exist "%~1" if /i "%~x1"==".exe" call :fc_file "%~1"
exit /b 0

REM ---- a conda.bat / conda.exe somewhere inside the conda folder ---------
:fc_file
REM ...\condabin\conda.bat  or  ...\Scripts\conda.exe  ->  the folder above
for %%F in ("%~dp1..") do call :fc_root "%%~fF"
REM ...\Library\bin\conda.bat (older installs) -> two folders above
if not defined CONDA_BAT for %%F in ("%~dp1..\..") do call :fc_root "%%~fF"
exit /b 0

:fc_reg
for /f "tokens=2,*" %%A in ('reg query "%~1\Software\Python" /s /ve 2^>nul ^| findstr /c:"REG_SZ"') do (
    if not defined CONDA_BAT call :fc_root "%%B"
)
exit /b 0

:fc_root
if exist "%~1\condabin\conda.bat" set "CONDA_BAT=%~1\condabin\conda.bat"
exit /b 0
