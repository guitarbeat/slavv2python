@echo off
REM run_matlab_cli.bat
REM Windows batch script to invoke MATLAB R2019a from command line
REM
REM Usage:
REM   run_matlab_cli.bat "input_file.tif" "output_directory" [matlab_path]
REM
REM Example:
REM   run_matlab_cli.bat "data\slavv_test_volume.tif" "comparison_output\matlab_results" "C:\Program Files\MATLAB\R2019a\bin\matlab.exe"

setlocal enabledelayedexpansion

REM Parse arguments
if "%~1"=="" (
    echo ERROR: Input file required
    echo Usage: run_matlab_cli.bat "input_file.tif" "output_directory" [matlab_path]
    exit /b 1
)

if "%~2"=="" (
    echo ERROR: Output directory required
    echo Usage: run_matlab_cli.bat "input_file.tif" "output_directory" [matlab_path]
    exit /b 1
)

set INPUT_FILE=%~1
set OUTPUT_DIR=%~2

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

REM Get absolute paths
for %%F in ("%INPUT_FILE%") do set INPUT_FILE_ABS=%%~fF
for %%D in ("%OUTPUT_DIR%") do set OUTPUT_DIR_ABS=%%~fD

REM Get script directory (where this batch file is located)
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Get project root (parent of parent of this directory)
for %%P in ("%SCRIPT_DIR%\..\..") do set PROJECT_ROOT=%%~fP

REM Change to Vectorization-Public directory for MATLAB
set VECTORIZATION_DIR=%PROJECT_ROOT%\external\Vectorization-Public

REM Ensure output directory exists before creating log file
if not exist "%OUTPUT_DIR_ABS%" (
    mkdir "%OUTPUT_DIR_ABS%"
)

set LOG_FILE=%OUTPUT_DIR_ABS%\matlab_run.log
echo MATLAB CLI Run Log > "%LOG_FILE%"
echo =================== >> "%LOG_FILE%"
echo Input file: %INPUT_FILE_ABS% >> "%LOG_FILE%"
echo Output directory: %OUTPUT_DIR_ABS% >> "%LOG_FILE%"
echo MATLAB path: %MATLAB_PATH% >> "%LOG_FILE%"
echo Start time: %DATE% %TIME% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Build MATLAB command
REM Note: R2019a+ uses -batch flag, older versions may need -r
REM Convert Windows paths to MATLAB-friendly format (forward slashes or escaped backslashes)
set MATLAB_SCRIPT=cd('%VECTORIZATION_DIR:\=/%'); addpath('%SCRIPT_DIR:\=/%'); run_matlab_vectorization('%INPUT_FILE_ABS:\=/%', '%OUTPUT_DIR_ABS:\=/%'); exit

echo Running MATLAB vectorization...
echo Command: "%MATLAB_PATH%" -batch "%MATLAB_SCRIPT%"
echo.

REM Run MATLAB and capture output
"%MATLAB_PATH%" -batch "%MATLAB_SCRIPT%" >> "%LOG_FILE%" 2>&1
set MATLAB_EXIT_CODE=%ERRORLEVEL%

echo.
echo MATLAB execution completed with exit code: %MATLAB_EXIT_CODE%
echo Log file: %LOG_FILE%

if %MATLAB_EXIT_CODE% NEQ 0 (
    echo ERROR: MATLAB execution failed. Check log file for details.
    exit /b %MATLAB_EXIT_CODE%
)

exit /b 0
