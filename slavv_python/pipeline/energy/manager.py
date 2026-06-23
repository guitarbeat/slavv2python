"""Consolidated energy field manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slavv_python.pipeline.energy.chunking import (
    _calculate_energy_field_chunked,
    _compute_direct_energy_outputs,
    _energy_lattice,
    _energy_result_payload,
)
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.resumable import calculate_energy_field_resumable

if TYPE_CHECKING:
    from slavv_python.engine.state import StageController
    from slavv_python.schema.results import EnergyResult


class EnergyManager:
    """Deep facade for multi-scale energy computation (ephemeral and resumable)."""

    @classmethod
    def run(
        cls,
        image: np.ndarray,
        params: dict[str, Any],
        get_chunking_lattice_func=None,
    ) -> EnergyResult:
        """Calculate energy without run-directory checkpointing.

        Args:
            image: Input 3D image array.
            params: Pipeline parameters containing scales and thresholds.
            get_chunking_lattice_func: Optional function to override lattice generation.

        Returns:
            EnergyResult: The computed energy field and associated metadata.
        """
        return cls._run(
            image,
            params,
            stage_controller=None,
            get_chunking_lattice_func=get_chunking_lattice_func,
        )

    @classmethod
    def run_resumable(
        cls,
        image: np.ndarray,
        params: dict[str, Any],
        stage_controller: StageController,
        get_chunking_lattice_func=None,
    ) -> EnergyResult:
        """Calculate energy with persisted chunk/scale units.

        Args:
            image: Input 3D image array.
            params: Pipeline parameters.
            stage_controller: Controller for managing stage state and artifacts.
            get_chunking_lattice_func: Optional function to override lattice generation.

        Returns:
            EnergyResult: The computed energy field and associated metadata.
        """
        return cls._run(
            image,
            params,
            stage_controller=stage_controller,
            get_chunking_lattice_func=get_chunking_lattice_func,
        )

    @classmethod
    def _run(
        cls,
        image: np.ndarray,
        params: dict[str, Any],
        *,
        stage_controller: StageController | None,
        get_chunking_lattice_func=None,
    ) -> EnergyResult:
        """Internal dispatcher for energy calculation.

        Args:
            image: Input 3D image array.
            params: Pipeline parameters.
            stage_controller: Optional stage controller for resumable execution.
            get_chunking_lattice_func: Optional function to override lattice generation.

        Returns:
            EnergyResult: The computed energy field and associated metadata.
        """
        if params.get("comparison_exact_network"):
            image = image.astype(np.float64, copy=False)
        else:
            image = image.astype(np.float32, copy=False)
        if stage_controller is not None:
            return calculate_energy_field_resumable(
                image,
                params,
                stage_controller,
                get_chunking_lattice_func=get_chunking_lattice_func,
            )

        config = _prepare_energy_config(image, params)
        lattice = _energy_lattice(
            image.shape,
            int(config["max_voxels"]),
            int(config["margin"]),
            get_chunking_lattice_func,
        )
        if len(lattice) > 1:
            return cast(
                "EnergyResult",
                _calculate_energy_field_chunked(
                    image,
                    params,
                    config,
                    lattice,
                    get_chunking_lattice_func,
                    cls.run,
                ),
            )
        energy_3d, scale_indices, energy_4d = _compute_direct_energy_outputs(image, config)
        return _energy_result_payload(config, image.shape, energy_3d, scale_indices, energy_4d)


__all__ = ["EnergyManager"]
