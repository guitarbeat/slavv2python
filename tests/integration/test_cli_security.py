import os
import subprocess
import pytest
from pathlib import Path
import sys

# Assume script is at <repo_root>/scripts/cli/run_matlab_cli.sh
REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts" / "cli"

@pytest.mark.skipif(sys.platform == 'win32', reason="Shell script only runs on Linux/Unix")
def test_run_matlab_cli_injection_sh(tmp_path):
    # Create a mock matlab script
    mock_matlab = tmp_path / "mock_matlab.sh"
    # The mock script just exits successfully
    mock_matlab.write_text("#!/bin/bash\necho \"MOCK MATLAB CALLED WITH: $@\"\nexit 0")
    mock_matlab.chmod(0o755)

    # Create a file with a single quote in its name
    input_file_name = "input'injection.tif"
    input_file = tmp_path / input_file_name
    input_file.touch()

    output_dir = tmp_path / "output_dir"

    script = SCRIPT_DIR / "run_matlab_cli.sh"

    # Verify script exists
    if not script.exists():
        pytest.fail(f"Script not found at {script}")

    # Make script executable
    script.chmod(0o755)

    # Run script
    # Arguments: input_file output_dir matlab_path
    result = subprocess.run(
        [str(script), str(input_file), str(output_dir), str(mock_matlab)],
        capture_output=True,
        text=True
    )

    # Check that the script ran successfully
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0

    # Check the log file for the command
    log_file = output_dir / "matlab_run.log"
    assert log_file.exists()
    content = log_file.read_text()

    # Check that the single quote was escaped in the MATLAB command string
    # We expect the path to be replaced: ' -> ''
    # So "input'injection.tif" -> "input''injection.tif"
    # And the whole thing wrapped in single quotes: '...input''injection.tif...'

    # Get absolute path used by script (realpath)
    # Python's resolve() is similar to realpath
    input_abs = input_file.resolve()
    expected_escaped = str(input_abs).replace("'", "''")

    print(f"Checking for escaped path: {expected_escaped}")
    print(f"Log content:\n{content}")

    # Verify the escaped path is present in the log (inside the command string)
    assert expected_escaped in content

    # Verify the specific MATLAB call structure
    # run_matlab_vectorization('...input''injection.tif', ...
    assert f"run_matlab_vectorization('{expected_escaped}'," in content
