from __future__ import annotations

import numpy as np
import pytest
from source.core._edge_candidates.global_watershed import _matlab_global_watershed_insert_available_location

@pytest.mark.unit
def test_insert_available_location_seed_1_ordering_with_ties():
    # MATLAB's seed_idx == 1 (primary seed) worst-to-best list (best is at the end)
    # The list is [worst, ..., best]
    # Insertion should maintain this order.
    # If energy is EQUAL, where does MATLAB insert it?
    
    energy_lookup = {0: -10.0, 1: -5.0, 2: -5.0, 3: -1.0}
    # Initial list (worst to best): [0, 1, 3]
    available_locations = [0, 1, 3]
    
    # Insert new location 2 with energy -5.0 (tie with 1)
    # Current implementation for seed_idx == 1:
    # if _energy_for(available_locations[0]) <= float(next_energy):
    #     insert_at = 0
    # else:
    #     insert_at = 0
    #     for idx, linear_index in enumerate(available_locations):
    #         if _energy_for(linear_index) > float(next_energy):
    #             insert_at = idx + 1
    
    # For available_locations=[0, 1, 3], next_energy=-5.0, seed_idx=1:
    # _energy_for(0) is -10.0. -10.0 <= -5.0 is True. insert_at = 0.
    # Result: [2, 0, 1, 3] -> 2 is now the WORST. But its energy is -5.0, which is BETTER than -10.0.
    # This looks WRONG if the list is supposed to be worst-to-best.
    
    updated = _matlab_global_watershed_insert_available_location(
        available_locations,
        next_location=2,
        next_energy=-5.0,
        energy_lookup=energy_lookup,
        seed_idx=1,
        is_current_location_clear=True
    )
    
    # Current result will be [2, 0, 1, 3]
    # Expected if sorted (worst to best): [0, 1, 2, 3] OR [0, 2, 1, 3]
    
    energies = [energy_lookup[loc] for loc in updated]
    # energies will be [-5.0, -10.0, -5.0, -1.0] -> NOT SORTED
    
    assert energies == sorted(energies), f"List not sorted: {energies}"

@pytest.mark.unit
def test_insert_available_location_seed_gt_1_ordering():
    energy_lookup = {0: -10.0, 1: -5.0, 2: -7.0, 3: -1.0}
    available_locations = [0, 1, 3]
    
    # Insert new location 2 with energy -7.0 (seed_idx > 1)
    # Current implementation for seed_idx > 1:
    # if _energy_for(available_locations[-1]) >= float(next_energy):
    #     insert_at = len(available_locations) if is_current_location_clear else len(available_locations) - 1
    # else:
    #     insert_at = next(idx for idx, linear_index in enumerate(available_locations) if _energy_for(linear_index) < float(next_energy))
    
    # For available_locations=[0, 1, 3], next_energy=-7.0, seed_idx=2, clear=True:
    # _energy_for(3) is -1.0. -1.0 >= -7.0 is True. insert_at = 3.
    # Result: [0, 1, 3, 2] -> 2 is now the BEST. But -7.0 is WORSE than -1.0.
    # This also looks WRONG.
    
    updated = _matlab_global_watershed_insert_available_location(
        available_locations,
        next_location=2,
        next_energy=-7.0,
        energy_lookup=energy_lookup,
        seed_idx=2,
        is_current_location_clear=True
    )
    
    energies = [energy_lookup[loc] for loc in updated]
    # energies will be [-10.0, -5.0, -1.0, -7.0] -> NOT SORTED
    
    assert energies == sorted(energies), f"List not sorted: {energies}"
