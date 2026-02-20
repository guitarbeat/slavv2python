#!/bin/bash
# run_matlab_cli.sh
# Bash script to invoke MATLAB from command line
#
# Usage:
#   ./run_matlab_cli.sh "input_file.tif" "output_directory" [matlab_path]

set -e

# Arguments
INPUT_FILE="$1"
OUTPUT_DIR="$2"
MATLAB_PATH="$3"

# Validation
if [ -z "$INPUT_FILE" ]; then
    echo "ERROR: Input file required"
    echo "Usage: ./run_matlab_cli.sh \"input_file.tif\" \"output_directory\" [matlab_path]"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory required"
    echo "Usage: ./run_matlab_cli.sh \"input_file.tif\" \"output_directory\" [matlab_path]"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Default MATLAB path if not provided
if [ -z "$MATLAB_PATH" ]; then
    # Try to find matlab in PATH
    if command -v matlab &> /dev/null; then
        MATLAB_PATH=$(command -v matlab)
    else
        MATLAB_PATH="/usr/local/bin/matlab"
    fi
fi

if [ ! -x "$MATLAB_PATH" ] && ! command -v "$MATLAB_PATH" &> /dev/null; then
    echo "ERROR: MATLAB executable not found or not executable at: $MATLAB_PATH"
    echo "Please provide correct MATLAB path as third argument"
    exit 1
fi

# Get absolute paths
INPUT_FILE_ABS=$(readlink -f "$INPUT_FILE")
OUTPUT_DIR_ABS=$(readlink -f "$OUTPUT_DIR") || OUTPUT_DIR_ABS="$OUTPUT_DIR" # Fallback if dir doesn't exist yet

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get project root (parent of parent of script dir)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VECTORIZATION_DIR="$PROJECT_ROOT/external/Vectorization-Public"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR_ABS"
# Re-resolve absolute path now that it exists
OUTPUT_DIR_ABS=$(readlink -f "$OUTPUT_DIR_ABS")

LOG_FILE="$OUTPUT_DIR_ABS/matlab_run.log"
{
    echo "MATLAB CLI Run Log"
    echo "==================="
    echo "Input file: $INPUT_FILE_ABS"
    echo "Output directory: $OUTPUT_DIR_ABS"
    echo "MATLAB path: $MATLAB_PATH"
    echo "Start time: $(date)"
    echo ""
} > "$LOG_FILE"

# Prepare MATLAB command
# Escape single quotes in paths for MATLAB string literals
# Replace ' with '' (standard SQL/MATLAB escaping)
INPUT_FILE_ESC="${INPUT_FILE_ABS//\'/''}"
OUTPUT_DIR_ESC="${OUTPUT_DIR_ABS//\'/''}"
VECTORIZATION_DIR_ESC="${VECTORIZATION_DIR//\'/''}"
SCRIPT_DIR_ESC="${SCRIPT_DIR//\'/''}"

MATLAB_SCRIPT="cd('$VECTORIZATION_DIR_ESC'); addpath('$SCRIPT_DIR_ESC'); run_matlab_vectorization('$INPUT_FILE_ESC', '$OUTPUT_DIR_ESC'); exit"

echo "Running MATLAB vectorization..."
echo "Command: $MATLAB_PATH -batch \"$MATLAB_SCRIPT\""

# Run MATLAB
# Note: -batch is available in R2019a+. For older versions, use -nodesktop -nosplash -r "..."
# We assume R2019a+ as per .bat file
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
