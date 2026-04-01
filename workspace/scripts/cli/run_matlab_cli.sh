#!/bin/bash
# run_matlab_cli.sh
# Bash script to invoke MATLAB from command line
#
# Usage:
#   ./run_matlab_cli.sh "input_file.tif" "output_directory" [matlab_path] [params_json]

set -e

# Arguments
INPUT_FILE="$1"
OUTPUT_DIR="$2"
MATLAB_PATH="$3"
PARAMS_FILE="$4"

# Validation
if [ -z "$INPUT_FILE" ]; then
    echo "ERROR: Input file required"
    echo "Usage: ./run_matlab_cli.sh \"input_file.tif\" \"output_directory\" [matlab_path] [params_json]"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory required"
    echo "Usage: ./run_matlab_cli.sh \"input_file.tif\" \"output_directory\" [matlab_path] [params_json]"
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

if [ -n "$PARAMS_FILE" ] && [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameters file not found: $PARAMS_FILE"
    exit 1
fi

# Get absolute paths
INPUT_FILE_ABS="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$(basename "$INPUT_FILE")"
if [ -n "$PARAMS_FILE" ]; then
    PARAMS_FILE_ABS="$(cd "$(dirname "$PARAMS_FILE")" && pwd)/$(basename "$PARAMS_FILE")"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get project root (parent of workspace/scripts/cli)
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
VECTORIZATION_DIR="$PROJECT_ROOT/external/Vectorization-Public"

# Thin shell-level safety checks. Keep policy in Python preflight.
if ! mkdir -p "$OUTPUT_DIR"; then
    echo "ERROR: Could not create output directory: $OUTPUT_DIR"
    exit 1
fi
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

ONEDRIVE_WARNING=""
if printf '%s' "$OUTPUT_DIR_ABS" | grep -qi "onedrive"; then
    ONEDRIVE_WARNING="WARNING: Output directory appears to be under OneDrive sync; prefer a local non-synced drive for MATLAB outputs."
fi

LOG_FILE="$OUTPUT_DIR_ABS/matlab_run.log"
if ! {
    echo "MATLAB CLI Run Log"
    echo "==================="
    echo "Input file: $INPUT_FILE_ABS"
    echo "Output directory: $OUTPUT_DIR_ABS"
    echo "MATLAB path: $MATLAB_PATH"
    if [ -n "$PARAMS_FILE_ABS" ]; then
        echo "Parameters file: $PARAMS_FILE_ABS"
    fi
    echo "Start time: $(date)"
    echo ""
} > "$LOG_FILE"; then
    echo "ERROR: Could not create log file in output directory: $LOG_FILE"
    exit 1
fi

if [ -n "$ONEDRIVE_WARNING" ]; then
    echo "$ONEDRIVE_WARNING"
    echo "$ONEDRIVE_WARNING" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
fi

# Prepare MATLAB command
# Escape single quotes in paths for MATLAB string literals
INPUT_FILE_ESC="${INPUT_FILE_ABS//\'/\'\'}"
OUTPUT_DIR_ESC="${OUTPUT_DIR_ABS//\'/\'\'}"
VECTORIZATION_DIR_ESC="${VECTORIZATION_DIR//\'/\'\'}"
SCRIPT_DIR_ESC="${SCRIPT_DIR//\'/\'\'}"

if [ -n "$PARAMS_FILE_ABS" ]; then
    PARAMS_FILE_ESC="${PARAMS_FILE_ABS//\'/\'\'}"
    MATLAB_SCRIPT="cd('$VECTORIZATION_DIR_ESC'); addpath('$SCRIPT_DIR_ESC'); run_matlab_vectorization('$INPUT_FILE_ESC', '$OUTPUT_DIR_ESC', '$PARAMS_FILE_ESC'); exit"
else
    MATLAB_SCRIPT="cd('$VECTORIZATION_DIR_ESC'); addpath('$SCRIPT_DIR_ESC'); run_matlab_vectorization('$INPUT_FILE_ESC', '$OUTPUT_DIR_ESC'); exit"
fi

echo "Running MATLAB vectorization..."
echo "Command: $MATLAB_PATH -batch \"$MATLAB_SCRIPT\""
echo "Command: $MATLAB_PATH -batch \"$MATLAB_SCRIPT\"" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Run MATLAB
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
