"""Energy provenance helpers."""

from __future__ import annotations

CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN = "python_native_hessian"
# Stretch Approach A: MATLAB-engine float path under Python orchestration.
STRETCH_ENGINE_ENERGY_ORIGIN = "matlab_engine_hessian"
# "hessian" is a legacy provenance label stored in older checkpoints; treat as equivalent.
EXACT_COMPATIBLE_ENERGY_ORIGINS = frozenset({CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN, "hessian"})
STRETCH_COMPATIBLE_ENERGY_ORIGINS = frozenset({STRETCH_ENGINE_ENERGY_ORIGIN})

ENERGY_FLOAT_BACKEND_NUMPY = "numpy"
ENERGY_FLOAT_BACKEND_MATLAB_ENGINE = "matlab_engine"
ENERGY_FLOAT_BACKENDS = frozenset({ENERGY_FLOAT_BACKEND_NUMPY, ENERGY_FLOAT_BACKEND_MATLAB_ENGINE})


def energy_origin_for_method(
    energy_method: str,
    *,
    energy_float_backend: str = ENERGY_FLOAT_BACKEND_NUMPY,
) -> str:
    """Return the persisted provenance label for one energy backend."""
    backend = str(energy_float_backend or ENERGY_FLOAT_BACKEND_NUMPY).strip().lower()
    if backend == ENERGY_FLOAT_BACKEND_MATLAB_ENGINE:
        if energy_method != "hessian":
            raise ValueError("energy_float_backend=matlab_engine requires energy_method=hessian")
        return STRETCH_ENGINE_ENERGY_ORIGIN
    if energy_method == "hessian":
        return CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN
    return f"python_{energy_method}"


def is_exact_compatible_energy_origin(origin: object) -> bool:
    """Return whether an energy provenance is accepted on the exact route."""
    return isinstance(origin, str) and origin in EXACT_COMPATIBLE_ENERGY_ORIGINS


def is_stretch_compatible_energy_origin(origin: object) -> bool:
    """Return whether an energy provenance is accepted for stretch Watershed."""
    return isinstance(origin, str) and origin in STRETCH_COMPATIBLE_ENERGY_ORIGINS


def is_watershed_allowed_energy_origin(
    origin: object,
    *,
    stretch_mode: bool = False,
) -> bool:
    """Exact-route Watershed allowlist; stretch mode adds engine origin only."""
    if is_exact_compatible_energy_origin(origin):
        return True
    return bool(stretch_mode) and is_stretch_compatible_energy_origin(origin)


def refuse_mixed_stretch_energy_origins(origins: set[str] | frozenset[str]) -> None:
    """Stretch proofs reject mixed NumPy/engine Energy provenance (KTD7)."""
    cleaned = {str(item) for item in origins if item}
    has_native = bool(cleaned & EXACT_COMPATIBLE_ENERGY_ORIGINS)
    has_engine = bool(cleaned & STRETCH_COMPATIBLE_ENERGY_ORIGINS)
    if has_native and has_engine:
        raise ValueError(
            "mixed Energy origins in stretch proof: "
            f"{sorted(cleaned)}; refuse NumPy+engine mix (KTD7)"
        )
    if has_native and not has_engine:
        raise ValueError(
            f"stretch proof requires matlab_engine_hessian origin; found {sorted(cleaned)}"
        )


def exact_route_gate_description() -> str:
    """Return the maintained summary of the exact-route gate."""
    return (
        "comparison_exact_network + python_native_hessian; "
        "energy.energy ULP≤48 + strict scale_indices (ADR 0011)"
    )


def exact_compatible_energy_origins_text() -> str:
    """Render accepted exact-route energy origins for user-facing errors."""
    return ", ".join(sorted(EXACT_COMPATIBLE_ENERGY_ORIGINS))


__all__ = [
    "CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN",
    "ENERGY_FLOAT_BACKENDS",
    "ENERGY_FLOAT_BACKEND_MATLAB_ENGINE",
    "ENERGY_FLOAT_BACKEND_NUMPY",
    "EXACT_COMPATIBLE_ENERGY_ORIGINS",
    "STRETCH_COMPATIBLE_ENERGY_ORIGINS",
    "STRETCH_ENGINE_ENERGY_ORIGIN",
    "energy_origin_for_method",
    "exact_compatible_energy_origins_text",
    "exact_route_gate_description",
    "is_exact_compatible_energy_origin",
    "is_stretch_compatible_energy_origin",
    "is_watershed_allowed_energy_origin",
    "refuse_mixed_stretch_energy_origins",
]
