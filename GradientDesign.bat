@echo off
setlocal EnableExtensions

rem Always work from the repository root, even when launched by double-click.
cd /d "%~dp0"

set "APP_NAME=GradientDesign"
set "SETUP_VERSION=2026-08-20-1"
set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYTHONW=%VENV_DIR%\Scripts\pythonw.exe"
set "SETUP_STAMP=%VENV_DIR%\.gradientdesign-setup"
set "REQUIREMENTS=%CD%\requirements-project.txt"
set "GUI_SCRIPT=%CD%\halbach_coils\run_gui.py"

title %APP_NAME%

rem A matching stamp plus a successful import check makes subsequent launches fast.
if not exist "%VENV_PYTHON%" goto setup
if not exist "%SETUP_STAMP%" goto setup

set "INSTALLED_SETUP_VERSION="
set /p INSTALLED_SETUP_VERSION=<"%SETUP_STAMP%"
if not "%INSTALLED_SETUP_VERSION%"=="%SETUP_VERSION%" goto setup

"%VENV_PYTHON%" -c "import tkinter, numpy, scipy, trimesh, sympy, PIL, matplotlib, manifold3d, networkx, pandas, rtree, skimage" >nul 2>&1
if errorlevel 1 goto setup
goto launch

:setup
echo.
echo ============================================================
echo  Configuracion inicial de %APP_NAME%
echo ============================================================
echo.
echo Este proceso puede tardar varios minutos la primera vez.
echo.

if not exist "%REQUIREMENTS%" goto missing_files
if not exist "%GUI_SCRIPT%" goto missing_files

rem Reuse a healthy environment when possible; replace only a broken one.
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
    if not errorlevel 1 goto install
)

if exist "%VENV_DIR%" (
    echo Eliminando un entorno virtual incompleto o incompatible...
    rmdir /s /q "%VENV_DIR%"
    if exist "%VENV_DIR%" goto remove_failed
)

where py >nul 2>&1
if errorlevel 1 goto python_missing

py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
if errorlevel 1 goto python_missing

echo Creando el entorno virtual con Python 3.11...
py -3.11 -m venv "%VENV_DIR%"
if errorlevel 1 goto setup_failed

:install
echo Actualizando pip...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto setup_failed

echo Instalando las dependencias del proyecto...
"%VENV_PYTHON%" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 goto setup_failed

echo Verificando la instalacion...
"%VENV_PYTHON%" -m pip check
if errorlevel 1 goto setup_failed

"%VENV_PYTHON%" -c "import tkinter, numpy, scipy, trimesh, sympy, PIL, matplotlib, manifold3d, networkx, pandas, rtree, skimage"
if errorlevel 1 goto setup_failed

>"%SETUP_STAMP%" echo %SETUP_VERSION%
echo.
echo Configuracion completada correctamente.
echo.

:launch
if not exist "%VENV_PYTHONW%" goto launch_console
start "" "%VENV_PYTHONW%" "%GUI_SCRIPT%"
exit /b 0

:launch_console
start "" "%VENV_PYTHON%" "%GUI_SCRIPT%"
exit /b 0

:python_missing
echo.
echo ERROR: No se encontro Python 3.11 mediante el comando "py -3.11".
echo Instala Python 3.11 de 64 bits y habilita el Python Launcher durante
echo la instalacion. Luego vuelve a ejecutar este archivo.
echo.
echo Descarga: https://www.python.org/downloads/release/python-3119/
goto wait_on_error

:missing_files
echo.
echo ERROR: Faltan requirements-project.txt o halbach_coils\run_gui.py.
echo Asegurate de ejecutar este archivo desde una copia completa del repositorio.
goto wait_on_error

:remove_failed
echo.
echo ERROR: No se pudo reemplazar "%VENV_DIR%".
echo Cierra la GUI y cualquier terminal que este usando el entorno e intenta de nuevo.
goto wait_on_error

:setup_failed
echo.
echo ERROR: La configuracion no termino correctamente.
echo Revisa el mensaje anterior. Al volver a ejecutar este archivo se reintentara.

:wait_on_error
echo.
pause
exit /b 1
