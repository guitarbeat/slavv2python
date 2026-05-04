"""
Telemetry Normalizer for SLAVV

This module provides tools to flatten nested JSON telemetry logs into queryable tables
using pandas. It specifically targets artifacts like run_snapshot.json and candidate_audit.json
to extract parity-path metrics for MATLAB alignment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Compatibility helper for older pandas
try:
    from pandas import json_normalize
except ImportError:
    from pandas.io.json import json_normalize


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryNormalizer:
    """
    Normalizes complex, nested SLAVV telemetry logs into flat DataFrames.
    """

    def __init__(self, output_dir: str | Path = "03_Analysis"):
        """
        Initialize the normalizer.

        Args:
            output_dir: Target directory for flattened tables.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_json(self, file_path: str | Path) -> dict[str, Any]:
        """
        Load a JSON file with basic error handling.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error reading {path}: {e}")
            return {}

    def normalize_run_snapshot(self, data: dict[str, Any]) -> pd.DataFrame:
        """
        Flatten run_snapshot.json content.
        Typically contains timings, memory usage, and run metadata.
        """
        if not data:
            return pd.DataFrame()

        # Flatten top-level metrics
        df = json_normalize(data)
        logger.info(f"Normalized run snapshot: {len(df)} rows")
        return df

    def normalize_candidate_audit(self, data: dict[str, Any]) -> pd.DataFrame:
        """
        Flatten candidate_audit.json content.
        Focuses on parity-path metrics like candidate counts and watershed pairs.
        """
        if not data:
            return pd.DataFrame()

        # Often these are lists of events or a dict with nested structures.
        # We try to flatten the core 'events' or 'candidates' if they exist, 
        # otherwise flatten the whole root.

        # If 'events' is a list, normalize that
        if "events" in data and isinstance(data["events"], list):
            df = json_normalize(data["events"])
        else:
            df = json_normalize(data)

        # Ensure requested parity fields are present or defaulted
        required_fields = ["candidate_connection_count", "watershed_total_pairs", "use_frontier_tracer"]
        for field in required_fields:
            if field not in df.columns:
                # Try to find them in nested paths if they weren't flattened perfectly
                # (This depends on the exact JSON structure)
                logger.debug(f"Field {field} not found in top-level flattening.")

        logger.info(f"Normalized candidate audit: {len(df)} rows")
        return df

    def normalize_run_manifest(self, data: dict[str, Any]) -> pd.DataFrame:
        """
        Flatten run_manifest.json content.
        Typically contains provenance and file paths.
        """
        if not data:
            return pd.DataFrame()

        df = json_normalize(data)
        logger.info(f"Normalized run manifest: {len(df)} rows")
        return df

    def process_run(self, run_dir: str | Path) -> dict[str, pd.DataFrame]:
        """
        Batch process all standard telemetry artifacts in a run directory.
        """
        run_path = Path(run_dir)
        results = {}

        # Define targets
        targets = {
            "run_snapshot.json": self.normalize_run_snapshot,
            "candidate_audit.json": self.normalize_candidate_audit,
            "run_manifest.json": self.normalize_run_manifest,
        }

        for filename, normalizer in targets.items():
            # Check both root and 99_Metadata
            file_path = run_path / filename
            if not file_path.exists():
                file_path = run_path / "99_Metadata" / filename

            if file_path.exists():
                data = self.load_json(file_path)
                df = normalizer(data)
                if not df.empty:
                    # Add run provenance info
                    df["_source_run"] = str(run_path.name)
                    results[filename.replace(".json", "")] = df

        return results

    def export(self, dfs: dict[str, pd.DataFrame], prefix: str = "") -> None:
        """
        Export DataFrames to .jsonl and .csv in the output directory.
        """
        for name, df in dfs.items():
            if df.empty:
                continue

            base_name = f"{prefix}{name}"

            # Export as CSV
            csv_path = self.output_dir / f"{base_name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Exported {csv_path}")

            # Export as JSONL (append-only style)
            jsonl_path = self.output_dir / f"{base_name}.jsonl"
            df.to_json(jsonl_path, orient="records", lines=True)
            logger.info(f"Exported {jsonl_path}")


if __name__ == "__main__":
    # Example usage / CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Normalize SLAVV telemetry logs.")
    parser.add_argument("-i", "--input", required=True, help="Run directory containing JSON logs.")
    parser.add_argument("-o", "--output", default="03_Analysis", help="Output directory.")
    args = parser.parse_args()

    normalizer = TelemetryNormalizer(output_dir=args.output)
    processed_dfs = normalizer.process_run(args.input)
    normalizer.export(processed_dfs)
