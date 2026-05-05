from __future__ import annotations

import numpy as np
import pytest
from source.core._edge_candidates.global_watershed import _matlab_global_watershed_insert_available_location

@pytest.mark.unit
def test_insert_available_location_seed_1_ordering_with_ties():
    # MATLAB's seed_idx == 1 (primary seed) worst-to-best list (best is at the end)
    # The list is [worst, ..., best] (highest energy to lowest energy)
    # Insertion should maintain this order.
    
    energy_lookup = {0: -10.0, 1: -5.0, 2: -5.0, 3: -1.0}
    # Initial list (worst to best): [3, 1, 0]
    available_locations = [3, 1, 0]
    
    updated = _matlab_global_watershed_insert_available_location(
        available_locations,
        next_location=2,
        next_energy=-5.0,
        energy_lookup=energy_lookup,
        seed_idx=1,
        is_current_location_clear=True
    )
    
    energies = [energy_lookup[loc] for loc in updated]
    # energies should be worst-to-best descending: [-1.0, -5.0, -5.0, -10.0]
    
    assert energies == sorted(energies, reverse=True), f"List not sorted descending: {energies}"

@pytest.mark.unit
def test_insert_available_location_seed_gt_1_ordering():
    energy_lookup = {0: -10.0, 1: -5.0, 2: -7.0, 3: -1.0}
    # Initial list (worst to best): [3, 1, 0]
    available_locations = [3, 1, 0]
    
    updated = _matlab_global_watershed_insert_available_location(
        available_locations,
        next_location=2,
        next_energy=-7.0,
        energy_lookup=energy_lookup,
        seed_idx=2,
        is_current_location_clear=True
    )
    
    energies = [energy_lookup[loc] for loc in updated]
    # energies should be worst-to-best descending: [-1.0, -5.0, -7.0, -10.0]
    
    assert energies == sorted(energies, reverse=True), f"List not sorted descending: {energies}"

