import bisect

# Energy values (Best is -20, Worst is -5)
# We want to maintain [Worst, ..., Best] so pop() is efficient for Best.
energy_map = {
    101: -5,
    102: -10,
    103: -20
}

available_locations = [101, 102, 103] # [Worst, ..., Best]

def find_insert_point(new_loc, new_energy):
    # We want to insert new_loc into available_locations such that energies remain descending
    # i.e. available_locations[i] energies: [-5, -10, -20]
    # Use bisect_right on negated energies: [5, 10, 20] (Ascending)
    return bisect.bisect_right(
        available_locations,
        -new_energy,
        key=lambda loc: -energy_map[loc]
    )

# Test inserting -15
next_energy = -15
next_loc = 104
energy_map[104] = -15

idx = find_insert_point(next_loc, next_energy)
print(f"Insertion point for -15 into [-5, -10, -20]: {idx}")
available_locations.insert(idx, next_loc)
print(f"Resulting list: {available_locations}")
print(f"Resulting energies: {[energy_map[l] for l in available_locations]}")

# Test inserting -25 (new Best)
next_energy = -25
next_loc = 105
energy_map[105] = -25
idx = find_insert_point(next_loc, next_energy)
available_locations.insert(idx, next_loc)
print(f"Resulting energies after -25: {[energy_map[l] for l in available_locations]}")

# Test inserting -2 (new Worst)
next_energy = -2
next_loc = 106
energy_map[106] = -2
idx = find_insert_point(next_loc, next_energy)
available_locations.insert(idx, next_loc)
print(f"Resulting energies after -2: {[energy_map[l] for l in available_locations]}")
