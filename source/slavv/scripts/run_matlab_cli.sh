#!/bin/bash
INPUT_FILE="$1"
OUTPUT_DIR="$2"
MATLAB_EXE="$3"

echo "Running MATLAB..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"

"$MATLAB_EXE" -batch "addpath('slavv/scripts'); try, run_matlab_vectorization('$INPUT_FILE', '$OUTPUT_DIR'); catch ME, disp(ME.message); exit(1); end; exit(0);"
