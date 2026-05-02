"""Energy provenance helpers."""

from __future__ import annotations

CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN = "python_native_hessian"
EXACT_COMPATIBLE_ENERGY_ORIGINS = frozenset({CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN})


def energy_origin_for_method(energy_method: str) -> str:
    """Return the persisted provenance label for one energy backend."""
    if energy_method == "hessian":
        return CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN
    return f"python_{energy_method}"


def is_exact_compatible_energy_origin(origin: object) -> bool:
    """Return whether an energy provenance is accepted on the exact route."""
    return isinstance(origin, str) and origin in EXACT_COMPATIBLE_ENERGY_ORIGINS


def exact_route_gate_description() -> str:
    """Return the maintained summary of the exact-route gate."""
    return "comparison_exact_network + python_native_hessian energy provenance"


def exact_compatible_energy_origins_text() -> str:
    """Render accepted exact-route energy origins for user-facing errors."""
    return ", ".join(sorted(EXACT_COMPATIBLE_ENERGY_ORIGINS))


__all__ = [
    "CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN",
    "EXACT_COMPATIBLE_ENERGY_ORIGINS",
    "energy_origin_for_method",
    "exact_compatible_energy_origins_text",
    "exact_route_gate_description",
    "is_exact_compatible_energy_origin",
]
