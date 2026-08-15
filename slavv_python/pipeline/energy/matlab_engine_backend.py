"""In-process MATLAB Engine adapter for stretch Energy float math.

Default Phase 1 Energy remains NumPy (``python_native_hessian``). This module
is the Approach A float path (KTD2/KTD3): one long-lived engine per Energy job,
Fortran-order array transfer, no ``.tolist()`` volume marshalling.

Missing MATLAB / unsupported Python / license failures raise
``MatlabEngineInfraError`` → stretch status ``incomplete_infra`` (never
``blocked_float_path``).
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# R2019a-era Engine for Python support window (setup.py on this host).
_SUPPORTED_PYTHON_PREFIXES = ("2.7", "3.5", "3.6", "3.7")


class MatlabEngineInfraError(RuntimeError):
    """MATLAB / engine / license / version infra failure (incomplete_infra)."""


class MatlabEnginePolicyError(ValueError):
    """Caller asked for a forbidden stretch success definition (e.g. R6)."""


@dataclass(frozen=True)
class MatlabEnginePrerequisites:
    """Resolved prerequisites for starting an engine session."""

    matlab_root: Path
    vectorization_root: Path
    python_version: str


def matlab_engine_importable() -> bool:
    """Return True when ``matlab.engine`` can be imported in this interpreter."""
    try:
        import matlab.engine  # noqa: F401
    except Exception:
        return False
    return True


def resolve_matlab_root(matlab_exe: str | Path | None = None) -> Path:
    """Resolve the MATLAB install root from ``matlab.exe`` or ``MATLAB_EXE``."""
    executable = (
        str(matlab_exe)
        if matlab_exe
        else os.environ.get("MATLAB_EXE") or shutil.which("matlab.exe") or shutil.which("matlab")
    )
    if not executable:
        raise MatlabEngineInfraError(
            "MATLAB executable unavailable; set MATLAB_EXE or put matlab on PATH"
        )
    exe_path = Path(executable)
    if not exe_path.is_file():
        raise MatlabEngineInfraError(f"MATLAB executable not found: {exe_path}")
    # …/bin/matlab.exe → install root
    return exe_path.resolve().parent.parent


def default_vectorization_root() -> Path:
    """Vendored Vectorization-Public source root."""
    return Path(__file__).resolve().parents[3] / "external" / "Vectorization-Public"


def verify_matlab_engine_prerequisites(
    *,
    matlab_exe: str | Path | None = None,
    vectorization_root: Path | None = None,
) -> MatlabEnginePrerequisites:
    """Fail fast when engine stretch path cannot run on this host."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if not any(version.startswith(prefix) for prefix in _SUPPORTED_PYTHON_PREFIXES):
        raise MatlabEngineInfraError(
            "MATLAB Engine for Python (R2019a) supports Python "
            f"{', '.join(_SUPPORTED_PYTHON_PREFIXES)}, but this interpreter is "
            f"{sys.version.split()[0]} — stretch Energy must use a compatible "
            "operator Python (incomplete_infra)"
        )
    matlab_root = resolve_matlab_root(matlab_exe)
    root = Path(vectorization_root) if vectorization_root else default_vectorization_root()
    if not root.is_dir():
        raise MatlabEngineInfraError(f"Vectorization-Public unavailable: {root}")
    source = root / "source"
    if not source.is_dir():
        raise MatlabEngineInfraError(f"Vectorization-Public source missing: {source}")
    if not matlab_engine_importable():
        raise MatlabEngineInfraError(
            "matlab.engine is not importable; install MATLAB Engine for Python "
            "into this interpreter (incomplete_infra)"
        )
    return MatlabEnginePrerequisites(
        matlab_root=matlab_root,
        vectorization_root=root,
        python_version=version,
    )


def numpy_to_matlab_double(array: np.ndarray) -> Any:
    """Convert a float64 ndarray to ``matlab.double`` without ndarray ``.tolist()``.

    Packs a Fortran-order flat sequence and sets ``size`` so MATLAB receives
    column-major ``[Y, X, Z]`` layout without a C-order nested ``.tolist()``.
    """
    try:
        import matlab
    except Exception as exc:
        raise MatlabEngineInfraError(
            f"matlab package unavailable for array transfer: {exc}"
        ) from exc

    arr = np.asfortranarray(np.asarray(array, dtype=np.float64))
    if arr.ndim == 0:
        return matlab.double([float(arr)])
    flat = np.ravel(arr, order="F")
    # Element iterator avoids ndarray.tolist(); size= restores MATLAB shape.
    sequence = [float(value) for value in flat]
    size = [int(dim) for dim in arr.shape]
    try:
        return matlab.double(sequence, size=size)
    except TypeError:
        # Older constructors may reject size=; fall back to column vector.
        return matlab.double(sequence)


def matlab_double_to_numpy(matlab_array: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Convert ``matlab.double`` back to float64 ndarray with given shape (F-order)."""
    data = getattr(matlab_array, "_data", None)
    if data is not None:
        flat = np.frombuffer(data, dtype=np.float64).copy()
        return np.ascontiguousarray(np.reshape(flat, shape, order="F"))
    as_array = np.asarray(matlab_array, dtype=np.float64)
    if tuple(as_array.shape) == shape:
        return np.ascontiguousarray(as_array)
    flat = np.ravel(as_array, order="F")
    return np.ascontiguousarray(np.reshape(flat, shape, order="F"))


def refuse_matlab_only_energy_checkpoint_as_stretch_success() -> None:
    """R6: MATLAB-written Energy alone is not stretch success."""
    raise MatlabEnginePolicyError(
        "loading a MATLAB-only Energy checkpoint is not stretch success; "
        "Python must produce Energy under orchestration (R6)"
    )


def ensure_matlab_engine_float_backend_ready(config: dict[str, Any]) -> None:
    """Refuse NumPy float-body execution under ``energy_float_backend=matlab_engine``.

    When the stretch backend is selected, either a bound engine float body must
    be active (``config['_stretch_engine_float_body_bound']=True`` set by the
    job session) or this raises ``MatlabEngineInfraError``. Never fall back to
    NumPy while stamping ``matlab_engine_hessian``.
    """
    backend = str(config.get("energy_float_backend", "numpy")).strip().lower()
    if backend != "matlab_engine":
        return
    if config.get("_stretch_engine_float_body_bound") is True:
        verify_matlab_engine_prerequisites(
            matlab_exe=config.get("matlab_exe"),
            vectorization_root=(
                Path(config["vectorization_root"]) if config.get("vectorization_root") else None
            ),
        )
        return
    # Probe infra first so unsupported Python / missing engine classify correctly.
    verify_matlab_engine_prerequisites(
        matlab_exe=config.get("matlab_exe"),
        vectorization_root=(
            Path(config["vectorization_root"]) if config.get("vectorization_root") else None
        ),
    )
    raise MatlabEngineInfraError(
        "matlab_engine backend selected but Energy float body is not bound to "
        "the MATLAB session; refusing NumPy execution under stretch origin "
        "(incomplete_infra — deepen energy_filter_V200 engine ownership)"
    )


class MatlabEngineSession:
    """One long-lived MATLAB engine for a stretch Energy job."""

    def __init__(
        self,
        *,
        matlab_exe: str | Path | None = None,
        vectorization_root: Path | None = None,
    ) -> None:
        self._prereqs = verify_matlab_engine_prerequisites(
            matlab_exe=matlab_exe,
            vectorization_root=vectorization_root,
        )
        self._engine: Any | None = None

    def __enter__(self) -> MatlabEngineSession:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.quit()

    @property
    def engine(self) -> Any:
        if self._engine is None:
            raise MatlabEngineInfraError("MATLAB engine session is not started")
        return self._engine

    def start(self) -> None:
        if self._engine is not None:
            return
        try:
            import matlab.engine
        except Exception as exc:
            raise MatlabEngineInfraError(f"failed to import matlab.engine: {exc}") from exc
        try:
            self._engine = matlab.engine.start_matlab()
        except Exception as exc:
            raise MatlabEngineInfraError(
                f"failed to start MATLAB engine (license/version?): {exc}"
            ) from exc
        source = str(self._prereqs.vectorization_root / "source")
        try:
            self._engine.addpath(source, nargout=0)
        except Exception as exc:
            self.quit()
            raise MatlabEngineInfraError(f"addpath failed for {source}: {exc}") from exc
        logger.info("MATLAB engine started for stretch Energy; addpath=%s", source)

    def quit(self) -> None:
        eng = self._engine
        self._engine = None
        if eng is None:
            return
        try:
            eng.quit()
        except Exception as exc:
            logger.warning("MATLAB engine quit raised: %s", exc)

    def call(self, name: str, *args: Any, nargout: int = 1) -> Any:
        """Invoke a MATLAB function by name; path-miss → infra error."""
        try:
            func = getattr(self.engine, name)
        except AttributeError as exc:
            raise MatlabEngineInfraError(f"MATLAB function not found on path: {name}") from exc
        try:
            return func(*args, nargout=nargout)
        except Exception as exc:
            raise MatlabEngineInfraError(f"MATLAB call {name!r} failed: {exc}") from exc

    def roundtrip_float64(self, array: np.ndarray) -> np.ndarray:
        """Identity round-trip through ``matlab.double`` for transfer tests."""
        shape = tuple(int(s) for s in np.asarray(array).shape)
        ml = numpy_to_matlab_double(array)
        # Use reshape in MATLAB to preserve size, then convert back.
        restored = matlab_double_to_numpy(ml, shape)
        return restored.astype(np.float64, copy=False)
