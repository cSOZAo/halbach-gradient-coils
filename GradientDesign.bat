@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Always work from the repository root, even when launched by double-click.
cd /d "%~dp0"

set "APP_NAME=GradientDesign"
set "SETUP_VERSION=2026-08-20-2"
set "SUPPORTED_PYTHONS=3.14 3.13 3.12 3.11"
set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYTHONW=%VENV_DIR%\Scripts\pythonw.exe"
set "SETUP_STAMP=%VENV_DIR%\.gradientdesign-setup"
set "REQUIREMENTS=%CD%\requirements-project.txt"
set "GUI_SCRIPT=%CD%\halbach_coils\run_gui.py"

title %APP_NAME%

rem A matching setup stamp plus version and import checks makes later launches fast.
if not exist "%VENV_PYTHON%" goto setup
if not exist "%SETUP_STAMP%" goto setup

"%VENV_PYTHON%" -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 15) and not sys.version_info[:3] == (3, 14, 1)" >nul 2>&1
if errorlevel 1 goto setup

set "INSTALLED_SETUP_VERSION="
set /p INSTALLED_SETUP_VERSION=<"%SETUP_STAMP%"
if not "!INSTALLED_SETUP_VERSION!"=="%SETUP_VERSION%" goto setup

"%VENV_PYTHON%" -c "import tkinter, numpy, scipy, trimesh, sympy, PIL, matplotlib, manifold3d, networkx, rtree, skimage" >nul 2>&1
if errorlevel 1 goto setup
if /i "%GRADIENTDESIGN_SETUP_ONLY%"=="1" exit /b 0
if exist "%VENV_PYTHONW%" (
    start "" "%VENV_PYTHONW%" "%GUI_SCRIPT%"
) else (
    start "" "%VENV_PYTHON%" "%GUI_SCRIPT%"
)
exit /b 0

:setup
echo.
echo ============================================================
echo  First-time setup for %APP_NAME%
echo ============================================================
echo.
echo This process may take several minutes on the first run.
echo.

if not exist "%REQUIREMENTS%" goto missing_files
if not exist "%GUI_SCRIPT%" goto missing_files

rem Reuse a healthy environment when possible; replace only a broken one.
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 15) and not sys.version_info[:3] == (3, 14, 1)" >nul 2>&1
    if not errorlevel 1 goto install
)

if exist "%VENV_DIR%" (
    echo Removing an incomplete or incompatible virtual environment...
    rmdir /s /q "%VENV_DIR%"
    if exist "%VENV_DIR%" goto remove_failed
)

set "SELECTED_PYTHON="
set "PYTHON_COMMAND="
where py >nul 2>&1
if not errorlevel 1 (
    for %%V in (%SUPPORTED_PYTHONS%) do (
        if not defined SELECTED_PYTHON (
            py -%%V -c "import sys; assert f'{sys.version_info.major}.{sys.version_info.minor}' == '%%V' and not sys.version_info[:3] == (3, 14, 1)" >nul 2>&1
            if not errorlevel 1 (
                set "SELECTED_PYTHON=%%V"
                set "PYTHON_COMMAND=py -%%V"
            )
        )
    )
)

rem Some Python installations do not include the launcher; accept a supported
rem interpreter on PATH as a fallback.
if not defined PYTHON_COMMAND (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 15) and not sys.version_info[:3] == (3, 14, 1)" >nul 2>&1
        if not errorlevel 1 (
            set "SELECTED_PYTHON=compatible version on PATH"
            set "PYTHON_COMMAND=python"
        )
    )
)
if not defined SELECTED_PYTHON goto python_missing

echo Creating the virtual environment with Python !SELECTED_PYTHON!...
!PYTHON_COMMAND! -m venv "%VENV_DIR%"
if errorlevel 1 goto setup_failed

:install
echo Updating pip...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto setup_failed

echo Installing project dependencies...
"%VENV_PYTHON%" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 goto setup_failed

echo Verifying the installation...
"%VENV_PYTHON%" -m pip check
if errorlevel 1 goto setup_failed

"%VENV_PYTHON%" -c "import tkinter, numpy, scipy, trimesh, sympy, PIL, matplotlib, manifold3d, networkx, rtree, skimage"
if errorlevel 1 goto setup_failed

>"%SETUP_STAMP%" echo %SETUP_VERSION%
echo.
echo Setup completed successfully.
echo.
if /i "%GRADIENTDESIGN_SETUP_ONLY%"=="1" exit /b 0
if exist "%VENV_PYTHONW%" (
    start "" "%VENV_PYTHONW%" "%GUI_SCRIPT%"
) else (
    start "" "%VENV_PYTHON%" "%GUI_SCRIPT%"
)
exit /b 0

:python_missing
echo.
echo ERROR: A compatible Python version was not found.
echo Install 64-bit Python 3.11, 3.12, 3.13, or 3.14. Python 3.14.1 is not supported;
echo use the newest Python 3.14 patch release, then run this file again.
echo.
echo Descarga: https://www.python.org/downloads/windows/
goto wait_on_error

:missing_files
echo.
echo ERROR: requirements-project.txt or halbach_coils\run_gui.py is missing.
echo Run this file from a complete copy of the repository.
goto wait_on_error

:remove_failed
echo.
echo ERROR: "%VENV_DIR%" could not be replaced.
echo Close the GUI and any terminal using the environment, then try again.
goto wait_on_error

:setup_failed
echo.
echo ERROR: Setup did not complete successfully.
echo Review the message above. Running this file again will retry setup.

:wait_on_error
echo.
pause
exit /b 1
