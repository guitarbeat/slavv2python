from __future__ import annotations

from source.core._energy import backends as legacy_energy_backends
from source.core._energy.config import _prepare_energy_config as legacy_prepare_energy_config
from source.core._energy.provenance import (
    energy_origin_for_method as legacy_energy_origin_for_method,
)
from source.core.energy_internal import energy_backends
from source.core.energy_internal.energy_config import (
    _prepare_energy_config as internal_prepare_energy_config,
)
from source.core.energy_internal.energy_provenance import (
    energy_origin_for_method as internal_energy_origin_for_method,
)


def test_grouped_energy_import_surfaces_resolve_consistently():
    assert legacy_energy_backends._require_zarr_backend is energy_backends._require_zarr_backend
    assert legacy_prepare_energy_config is internal_prepare_energy_config
    assert legacy_energy_origin_for_method is internal_energy_origin_for_method
