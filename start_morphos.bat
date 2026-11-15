@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "STUDIO_EXE=%~dp0MORPHOS_Studio.exe"
if not exist "%STUDIO_EXE%" set "STUDIO_EXE=%~dp0dist\MORPHOS_Studio.exe"
if exist "%STUDIO_EXE%" (
  start "" "%STUDIO_EXE%"
  exit /b 0
)

if not exist "%~dp0studio\dist\index.html" (
  echo MORPHOS Studio build is missing.
  echo Build it first with: npm --prefix studio install ^&^& npm --prefix studio run build
  pause
  exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 studio_launcher.py
  exit /b !errorlevel!
)
where python >nul 2>nul
if %errorlevel%==0 (
  python studio_launcher.py
  exit /b !errorlevel!
)

echo Python 3 is required. Install dependencies with: python -m pip install -r requirements.txt
pause
exit /b 1
