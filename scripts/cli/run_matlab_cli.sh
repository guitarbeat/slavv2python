#!/bin/bash
# run_matlab_cli.sh
# Linux shell script to invoke MATLAB from command line

# Usage:
#   run_matlab_cli.sh "input_file.tif" "output_directory" [matlab_path]

set -e

if [ "$#" -lt 2 ]; then
    echo "ERROR: Input file and output directory required"
    echo "Usage: $0 \"input_file.tif\" \"output_directory\" [matlab_path]"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="$2"
MATLAB_PATH="${3:-matlab}"

# Get absolute paths
INPUT_FILE_ABS=$(realpath "$INPUT_FILE")
OUTPUT_DIR_ABS=$(realpath "$OUTPUT_DIR")

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assuming script is in scripts/cli/, root is ../..
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VECTORIZATION_DIR="$PROJECT_ROOT/external/Vectorization-Public"

mkdir -p "$OUTPUT_DIR_ABS"
LOG_FILE="$OUTPUT_DIR_ABS/matlab_run.log"

echo "MATLAB CLI Run Log" > "$LOG_FILE"
echo "===================" >> "$LOG_FILE"
echo "Input file: $INPUT_FILE_ABS" >> "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR_ABS" >> "$LOG_FILE"
echo "MATLAB path: $MATLAB_PATH" >> "$LOG_FILE"
date >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Escape single quotes for MATLAB string literals
# Replace ' with '' using Bash parameter expansion
INPUT_FILE_ESC="${INPUT_FILE_ABS//\'/\'\'}"
OUTPUT_DIR_ESC="${OUTPUT_DIR_ABS//\'/\'\'}"
VECTORIZATION_DIR_ESC="${VECTORIZATION_DIR//\'/\'\'}"
SCRIPT_DIR_ESC="${SCRIPT_DIR//\'/\'\'}"

# Build MATLAB command
# Note: we use -batch for R2019a+
MATLAB_SCRIPT="cd('$VECTORIZATION_DIR_ESC'); addpath('$SCRIPT_DIR_ESC'); run_matlab_vectorization('$INPUT_FILE_ESC', '$OUTPUT_DIR_ESC'); exit"

echo "Running MATLAB vectorization..."
# Don't echo the full script if it's long, but here it's fine.
echo "Command: \"$MATLAB_PATH\" -batch \"$MATLAB_SCRIPT\""
echo ""

# Run MATLAB
# Note: redirection to log file captures output
"$MATLAB_PATH" -batch "$MATLAB_SCRIPT" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo ""
echo "MATLAB execution completed with exit code: $EXIT_CODE"
echo "Log file: $LOG_FILE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: MATLAB execution failed. Check log file for details."
    exit $EXIT_CODE
fi

exit 0
