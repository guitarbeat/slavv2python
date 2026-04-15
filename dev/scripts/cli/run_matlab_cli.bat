@echo off
REM run_matlab_cli.bat
REM Windows batch script to invoke MATLAB R2019a from command line
REM
REM Usage:
REM   run_matlab_cli.bat "input_file.tif" "output_directory" [matlab_path] [params_json]
REM
REM Example:
REM   run_matlab_cli.bat "data\slavv_test_volume.tif" "comparison_output\matlab_results" "C:\Program Files\MATLAB\R2019a\bin\matlab.exe"

setlocal enabledelayedexpansion

REM Parse arguments
if "%~1"=="" (
    echo ERROR: Input file required
    echo Usage: run_matlab_cli.bat "input_file.tif" "output_directory" [matlab_path] [params_json]
    exit /b 1
)

if "%~2"=="" (
    echo ERROR: Output directory required
    echo Usage: run_matlab_cli.bat "input_file.tif" "output_directory" [matlab_path] [params_json]
    exit /b 1
)

set INPUT_FILE=%~1
set OUTPUT_DIR=%~2

REM Validate input file exists
if not exist "%INPUT_FILE%" (
    echo ERROR: Input file not found: %INPUT_FILE%
    exit /b 1
)

REM Default MATLAB path if not provided
if "%~3"=="" (
    set MATLAB_PATH=C:\Program Files\MATLAB\R2019a\bin\matlab.exe
) else (
    set MATLAB_PATH=%~3
)

REM Check if MATLAB exists
if not exist "%MATLAB_PATH%" (
    echo ERROR: MATLAB not found at: %MATLAB_PATH%
    echo Please provide correct MATLAB path as third argument
    exit /b 1
)

set PARAMS_FILE=
if not "%~4"=="" (
    set PARAMS_FILE=%~4
    if not exist "!PARAMS_FILE!" (
        echo ERROR: Parameters file not found: !PARAMS_FILE!
        exit /b 1
    )
)

REM Get absolute paths
for %%F in ("%INPUT_FILE%") do set INPUT_FILE_ABS=%%~fF
for %%D in ("%OUTPUT_DIR%") do set OUTPUT_DIR_ABS=%%~fD
if defined PARAMS_FILE (
    for %%P in ("%PARAMS_FILE%") do set PARAMS_FILE_ABS=%%~fP
)

set "ONEDRIVE_WARNING="
echo %OUTPUT_DIR_ABS% | findstr /I /C:"OneDrive" >nul && set "ONEDRIVE_WARNING=WARNING: Output directory appears to be under OneDrive sync; prefer a local non-synced drive for MATLAB outputs."

REM Get script directory (where this batch file is located)
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Get project root (parent of dev/scripts/cli)
for %%P in ("%SCRIPT_DIR%\..\..\..") do set PROJECT_ROOT=%%~fP

REM Change to Vectorization-Public directory for MATLAB
set VECTORIZATION_DIR=%PROJECT_ROOT%\external\Vectorization-Public

REM Thin shell-level safety checks. Keep policy in Python preflight.
if not exist "%OUTPUT_DIR_ABS%" (
    mkdir "%OUTPUT_DIR_ABS%" 2>nul
    if errorlevel 1 (
        echo ERROR: Could not create output directory: %OUTPUT_DIR_ABS%
        exit /b 1
    )
)
if not exist "%OUTPUT_DIR_ABS%" (
    echo ERROR: Output directory is not accessible: %OUTPUT_DIR_ABS%
    exit /b 1
)

set LOG_FILE=%OUTPUT_DIR_ABS%\matlab_run.log
> "%LOG_FILE%" (
    echo MATLAB CLI Run Log
    echo ===================
    echo Input file: %INPUT_FILE_ABS%
    echo Output directory: %OUTPUT_DIR_ABS%
    echo MATLAB path: %MATLAB_PATH%
    if defined PARAMS_FILE_ABS echo Parameters file: %PARAMS_FILE_ABS%
    echo Start time: %DATE% %TIME%
    echo.
)
if not exist "%LOG_FILE%" (
    echo ERROR: Could not create log file in output directory: %LOG_FILE%
    exit /b 1
)
if defined ONEDRIVE_WARNING (
    echo !ONEDRIVE_WARNING!
    >> "%LOG_FILE%" echo !ONEDRIVE_WARNING!
    >> "%LOG_FILE%" echo.
)

REM Build MATLAB command
REM Note: R2019a+ uses -batch flag, older versions may need -r
REM Convert Windows paths to MATLAB-friendly format (forward slashes or escaped backslashes)
REM Also escape single quotes for MATLAB string literals
set INPUT_FILE_ESC=%INPUT_FILE_ABS:'=''%
set OUTPUT_DIR_ESC=%OUTPUT_DIR_ABS:'=''%
set MATLAB_SCRIPT=cd('%VECTORIZATION_DIR:\=/%'); addpath('%SCRIPT_DIR:\=/%'); run_matlab_vectorization('%INPUT_FILE_ESC:\=/%', '%OUTPUT_DIR_ESC:\=/%'); exit
if defined PARAMS_FILE_ABS set PARAMS_FILE_ESC=%PARAMS_FILE_ABS:'=''%
if defined PARAMS_FILE_ABS set MATLAB_SCRIPT=cd('%VECTORIZATION_DIR:\=/%'); addpath('%SCRIPT_DIR:\=/%'); run_matlab_vectorization('%INPUT_FILE_ESC:\=/%', '%OUTPUT_DIR_ESC:\=/%', '%PARAMS_FILE_ESC:\=/%'); exit

echo Running MATLAB vectorization...
echo Command: "%MATLAB_PATH%" -wait -batch "%MATLAB_SCRIPT%"
echo.
>> "%LOG_FILE%" echo Command: "%MATLAB_PATH%" -wait -batch "%MATLAB_SCRIPT%"
>> "%LOG_FILE%" echo.

REM Run MATLAB and wait for the worker process so callers can time out or retry cleanly.
"%MATLAB_PATH%" -wait -batch "%MATLAB_SCRIPT%" >> "%LOG_FILE%" 2>&1
set MATLAB_EXIT_CODE=%ERRORLEVEL%

echo.
echo MATLAB execution completed with exit code: %MATLAB_EXIT_CODE%
echo Log file: %LOG_FILE%

if %MATLAB_EXIT_CODE% NEQ 0 (
    echo ERROR: MATLAB execution failed. Check log file for details.
    exit /b %MATLAB_EXIT_CODE%
)

exit /b 0

