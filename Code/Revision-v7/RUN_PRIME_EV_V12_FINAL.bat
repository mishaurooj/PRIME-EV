@echo off
setlocal EnableExtensions

rem Run from the folder that contains this BAT file and the Python script.
set "SCRIPT_DIR=%~dp0"
set "PY_SCRIPT=%SCRIPT_DIR%PRIME_EV_v12_final_corrected.py"
set "DATA_DIR=D:\other\prime-ev\Dataset"
set "MAIN_DATA=%DATA_DIR%\main-ev_charging_stations-dataset.csv"
set "EXTERNAL_STATIONS=%DATA_DIR%\ev_data.csv"
set "EXTERNAL_USAGE=%DATA_DIR%\EVChargingStationUsage.csv"
set "OUTPUT_DIR=%SCRIPT_DIR%prime_ev_v12_final"

if not exist "%PY_SCRIPT%" (
  echo ERROR: Python script not found:
  echo %PY_SCRIPT%
  pause
  exit /b 1
)
if not exist "%MAIN_DATA%" (
  echo ERROR: Main dataset not found:
  echo %MAIN_DATA%
  pause
  exit /b 1
)
if not exist "%EXTERNAL_STATIONS%" (
  echo ERROR: External station dataset not found:
  echo %EXTERNAL_STATIONS%
  pause
  exit /b 1
)
if not exist "%EXTERNAL_USAGE%" (
  echo ERROR: External usage dataset not found:
  echo %EXTERNAL_USAGE%
  pause
  exit /b 1
)

cd /d "%SCRIPT_DIR%"

echo Running PRIME-EV V12 final experiment...
echo Main dataset: %MAIN_DATA%
echo External station dataset: %EXTERNAL_STATIONS%
echo External usage dataset: %EXTERNAL_USAGE%
echo Output: %OUTPUT_DIR%
echo.

python "%PY_SCRIPT%" ^
  --data "%MAIN_DATA%" ^
  --external-us "%EXTERNAL_STATIONS%" ^
  --external-usage "%EXTERNAL_USAGE%" ^
  --output "%OUTPUT_DIR%" ^
  --epochs 30 ^
  --sensitivity-epochs 10 ^
  --regional-epochs 12 ^
  --multiseed-epochs 10 ^
  --operator-cv-epochs 10 ^
  --order-epochs 8 ^
  --order-permutations 5 ^
  --order-seeds "42,123,456" ^
  --review-seeds "42,123,456,789,2025,31415,27182,16180,57721,65537" ^
  --pair-loss-search-epochs 4 ^
  --pair-loss-search-steps 20 ^
  --pair-loss-search-pairs 20000 ^
  --deployment-selection-epochs 6 ^
  --deployment-selection-steps 30 ^
  --deployment-selection-pairs 20000 ^
  --cross-seeds "42,123,456,789,2025" ^
  --cross-bootstrap 1000 ^
  --temporal-seeds "42,123,456,789,2025" ^
  --temporal-epochs 40 ^
  --temporal-bootstrap 1000 ^
  --proxy-bootstrap 1000 ^
  --proxy-shuffles 1000 ^
  --target-shuffles 500 ^
  --torch-threads 1

if errorlevel 1 (
  echo.
  echo PRIME-EV V12 failed. Review the traceback above.
  pause
  exit /b 1
)

echo.
echo PRIME-EV V12 completed successfully.
echo Results are in:
echo %OUTPUT_DIR%
pause
