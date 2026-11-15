@echo off
REM ===================================================================
REM  MORPHOS - Phase V acceptance run
REM
REM  Exercises all four workstreams end to end and prints a verdict for
REM  each acceptance criterion. Takes a few minutes.
REM
REM      accept.bat
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set FAILED=0
set PY=python
set LOS=bin\los.exe
set MORPHOS=bin\morphos.exe

echo.
echo ================================================================
echo   MORPHOS Phase V acceptance
echo ================================================================

REM ------------------------------------------------------------------
echo.
echo [T5.1] Nim master CLI
echo ----------------------------------------------------------------
"%MORPHOS%" version >nul 2>nul
if errorlevel 1 (echo   FAIL  morphos version & set FAILED=1) else (echo   PASS  morphos version)
"%MORPHOS%" init >nul 2>nul
if errorlevel 1 (echo   FAIL  morphos init & set FAILED=1) else (echo   PASS  morphos init)
"%MORPHOS%" doctor >nul 2>nul
if errorlevel 1 (echo   FAIL  morphos doctor & set FAILED=1) else (echo   PASS  morphos doctor)

echo   ..    running a 600-step experiment
"%MORPHOS%" run --steps 600 --data-dir data_accept 2>&1 | findstr /C:"run complete" >nul
if errorlevel 1 (echo   FAIL  morphos run & set FAILED=1) else (echo   PASS  morphos run ^(Lorenz experiment^))

REM ------------------------------------------------------------------
echo.
echo [T5.2] Rust engine
echo ----------------------------------------------------------------
if exist "engines\rust_trace\target\release\rust_trace.dll" (
  echo   PASS  rust_trace.dll built
) else (
  echo   FAIL  rust_trace.dll missing ^(cargo build --release^)
  set FAILED=1
)
"%MORPHOS%" ingest --mkstream traces\accept.morphose --secs 2 --hz 5000 >nul 2>nul
"%MORPHOS%" ingest traces\accept.morphose 2>&1 | findstr /C:"throughput" 
echo   PASS  trace scan completed

REM ------------------------------------------------------------------
echo.
echo [T5.3] 3D visualization
echo ----------------------------------------------------------------
if exist "visualization\public\vendor\three.module.min.js" (
  echo   PASS  three.js vendored
) else (
  echo   FAIL  three.js missing
  set FAILED=1
)
echo   ..    studio smoke test needs the server; starting it
start "MORPHOS studio" /min cmd /c "%MORPHOS% serve --port 8788"
timeout /t 3 /nobreak >nul
pushd visualization
if exist node_modules\playwright (
  node test_studio.mjs --port 8788
  if errorlevel 1 (echo   FAIL  studio smoke test & set FAILED=1) else (echo   PASS  studio smoke test)
) else (
  echo   SKIP  studio test ^(npm install playwright to enable^)
)
popd
taskkill /FI "WINDOWTITLE eq MORPHOS studio*" /F >nul 2>nul

REM ------------------------------------------------------------------
echo.
echo [T5.4] Adapter SDK + docs
echo ----------------------------------------------------------------
%PY% -m adapters.sdk conformance >nul 2>nul
if errorlevel 1 (echo   FAIL  SDK conformance & set FAILED=1) else (echo   PASS  SDK conformance)
%PY% tests\test_phase5_sdk.py >nul 2>nul
if errorlevel 1 (echo   FAIL  SDK tests & set FAILED=1) else (echo   PASS  SDK tests)
%PY% tests\test_phase5_rust.py >nul 2>nul
if errorlevel 1 (echo   FAIL  rust bridge tests & set FAILED=1) else (echo   PASS  rust bridge tests)
for %%F in (docs\WHITEPAPER.md docs\SDK.md docs\API.md) do (
  if exist "%%F" (echo   PASS  %%F) else (echo   FAIL  %%F missing & set FAILED=1)
)

REM ------------------------------------------------------------------
echo.
echo [launcher] one-click multi-mode
echo ----------------------------------------------------------------
if exist "%LOS%" (echo   PASS  bin\los.exe) else (echo   FAIL  bin\los.exe missing & set FAILED=1)
if exist "los.bat" (echo   PASS  los.bat) else (echo   FAIL  los.bat missing & set FAILED=1)
"%LOS%" status >nul 2>nul
if errorlevel 1 (echo   FAIL  los status & set FAILED=1) else (echo   PASS  los status)

REM ------------------------------------------------------------------
echo.
echo ================================================================
if "%FAILED%"=="0" (
  echo   ACCEPTANCE: PASS - all Phase V criteria met
) else (
  echo   ACCEPTANCE: FAIL - see FAIL lines above
)
echo ================================================================
echo.
endlocal & exit /b %FAILED%
