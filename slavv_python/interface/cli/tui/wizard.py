"""Clack-style interactive setup wizard using questionary."""

from __future__ import annotations

from typing import Any

import questionary


def run_setup_wizard() -> dict[str, Any]:
    """Runs a step-by-step interactive CLI setup wizard for a SLAVV run."""
    questionary.print("\n🩸 Welcome to the SLAVV Setup Wizard\n", style="bold fg:ansired")

    # 1. Select/Input 3D TIFF Volume
    default_input = "volume.tif"
    input_path = questionary.path(
        "Enter or select the path to the input 3D TIFF volume file:",
        default=default_input,
        only_directories=False,
    ).ask()

    # If the user cancels the prompt, input_path will be None
    if input_path is None:
        return {}

    # 2. Choose Output Directory
    default_output = "slavv_output"
    output_dir = questionary.text(
        "Enter the destination directory for exported artifacts:",
        default=default_output,
    ).ask()

    if output_dir is None:
        return {}

    # 3. Choose Execution Profile
    profile = questionary.select(
        "Select the pipeline execution profile:",
        choices=[
            questionary.Choice("Native Paper Profile (Recommended)", value="paper"),
            questionary.Choice(
                "MATLAB Compatibility Profile (Exact Parity)", value="matlab_compat"
            ),
        ],
        default="paper",
    ).ask()

    if profile is None:
        return {}

    # 4. Select Export Formats (Checkbox spacebar multi-select)
    exports = questionary.checkbox(
        "Select artifacts to export upon completion (use space to select):",
        choices=[
            questionary.Choice("JSON Network Structure", value="json", checked=True),
            questionary.Choice("CSV Node/Edge Lists", value="csv", checked=True),
            questionary.Choice("Interactive HTML Plot", value="plot", checked=False),
        ],
    ).ask()

    if exports is None:
        return {}

    # 5. Confirm and Run
    confirm = questionary.confirm(
        "Would you like to start the vectorization run now with these settings?",
        default=True,
    ).ask()

    return {
        "input_path": input_path,
        "output_dir": output_dir,
        "profile": profile,
        "exports": exports,
        "execute_now": bool(confirm),
    }
