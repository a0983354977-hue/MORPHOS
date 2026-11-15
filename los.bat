@echo off
REM ===================================================================
REM  MORPHOS one-click launcher
REM
REM  Double-click this file, or run it from a terminal:
REM      los.bat                 interactive menu
REM      los.bat run             run a simulation
REM      los.bat studio          open the 3D studio
REM      los.bat live            run + stream live into the studio
REM      los.bat --help          every mode
REM
REM  It finds bin\los.exe, and builds it (plus the master CLI and the
REM  Rust engine) on first use if it is missing.
REM ===================================================================
setlocal
cd /d "%~dp0"

set "LOS=%~dp0bin\los.exe"

if not exist "%LOS%" goto build
goto launch

:build
echo.
echo   MORPHOS launcher not built yet - building now.
echo.
where nim >nul 2>nul
if errorlevel 1 (
  echo   [fail] Nim is not on PATH.
  echo          Install Nim from https://nim-lang.org/install.html and retry.
  echo          ^(The rest of MORPHOS still works: run bin\morphos.exe directly.^)
  pause
  exit /b 1
)
echo   building bin\los.exe ...
nim c -d:release --hints:off -o:"%LOS%" tools\los.nim
if errorlevel 1 (
  echo   [fail] build failed - see the messages above.
  pause
  exit /b 1
)
echo   [ok] built.
echo.

:launch
"%LOS%" %*
if errorlevel 2 (
  echo.
  pause
)
endlocal
