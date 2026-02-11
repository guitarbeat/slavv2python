@echo off
set "INPUT_FILE=%~1"
set "OUTPUT_DIR=%~2"
set "MATLAB_EXE=%~3"

:: Replace backslashes with forward slashes for MATLAB string compatibility
set "INPUT_FILE=%INPUT_FILE:\=/%"
set "OUTPUT_DIR=%OUTPUT_DIR:\=/%"

echo Running MATLAB...
echo Input: %INPUT_FILE%
echo Output: %OUTPUT_DIR%

"%MATLAB_EXE%" -batch "addpath('slavv/scripts'); try, run_matlab_vectorization('%INPUT_FILE%', '%OUTPUT_DIR%'); catch ME, disp(ME.message); exit(1); end; exit(0);"
