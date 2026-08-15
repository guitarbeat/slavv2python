"""Python 3.7 MATLAB Engine host for stretch Energy (R2019a ABI).

Repo slavv runs on Python >=3.11; R2019a ``matlab.engine`` supports <=3.7.
This module starts one long-lived 3.7 worker (or in-process engine when the
current interpreter is 3.7) and binds ``energy_filter_V200`` as the float body.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    MatlabEngineSession,
    current_python_supports_r2019a_engine,
    default_vectorization_root,
    matlab_double_to_numpy,
    numpy_to_matlab_double,
    resolve_matlab_root,
    verify_matlab_engine_prerequisites,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

STRETCH_PY37_ENV = "STRETCH_PY37_PYTHON"
DEFAULT_MATCHING_KERNEL = "3D gaussian conv annular pulse"
WORKER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "stretch_matlab_engine_worker.py"
HELPER_DIR = Path(__file__).resolve().parents[3] / "scripts"


def default_isolated_python37() -> Path:
    """Repo-local isolated 3.7 prefix (conda or venv), if present."""
    root = Path(__file__).resolve().parents[3]
    scratch = root / "workspace" / "scratch"
    for candidate in (
        scratch / "conda_py37_stretch" / "python.exe",
        scratch / "conda_py37_stretch" / "Scripts" / "python.exe",
        scratch / "venv_py37_stretch" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return candidate
    return scratch / "conda_py37_stretch" / "python.exe"


def resolve_python37_executable() -> Path | None:
    """Locate a Python 3.7 interpreter without using the repo 3.12 venv."""
    env_path = os.environ.get(STRETCH_PY37_ENV)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    isolated = default_isolated_python37()
    if isolated.is_file():
        candidates.append(isolated)
    which_37 = shutil.which("python3.7")
    if which_37:
        candidates.append(Path(which_37))
    anaconda = Path(r"C:\ProgramData\Anaconda3\python.exe")
    if anaconda.is_file():
        candidates.append(anaconda)
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve()) if candidate.is_file() else ""
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        if _python_is_37(candidate):
            return candidate.resolve()
    py_launcher = shutil.which("py")
    if py_launcher:
        try:
            completed = subprocess.run(
                [py_launcher, "-3.7", "-c", "import sys; print(sys.executable)"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            path = Path(completed.stdout.strip())
            if path.is_file() and _python_is_37(path):
                return path.resolve()
    return None


def _python_is_37(executable: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(executable), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "3.7"


class MatlabEnginePy37Worker:
    """Long-lived Python 3.7 subprocess hosting ``matlab.engine``."""

    def __init__(
        self,
        python37: Path,
        *,
        vectorization_root: Path,
        helper_dir: Path = HELPER_DIR,
    ) -> None:
        self._python37 = python37
        self._vectorization_root = vectorization_root
        self._helper_dir = helper_dir
        self._proc: subprocess.Popen[str] | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        if not WORKER_SCRIPT.is_file():
            raise MatlabEngineInfraError(f"stretch engine worker missing: {WORKER_SCRIPT}")
        self._tmpdir = tempfile.TemporaryDirectory(prefix="slavv_stretch_engine_")
        matlab_bin = resolve_matlab_root() / "bin" / "win64"
        env = os.environ.copy()
        env["PATH"] = str(matlab_bin) + os.pathsep + env.get("PATH", "")
        try:
            self._proc = subprocess.Popen(
                [str(self._python37), "-u", str(WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            self._cleanup_tmpdir()
            raise MatlabEngineInfraError(f"failed to spawn Python 3.7 worker: {exc}") from exc
        source = self._vectorization_root / "source"
        reply = self._request(
            {
                "op": "start",
                "vectorization_source": str(source),
                "helper_dir": str(self._helper_dir),
            }
        )
        if not reply.get("ok"):
            self.quit()
            raise MatlabEngineInfraError(
                f"Python 3.7 matlab.engine start failed: {reply.get('error')}"
            )
        logger.info("stretch Energy Python 3.7 MATLAB worker started: %s", self._python37)

    def quit(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                proc.stdin.flush()
            except OSError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._cleanup_tmpdir()

    def _cleanup_tmpdir(self) -> None:
        tmp = self._tmpdir
        self._tmpdir = None
        if tmp is not None:
            tmp.cleanup()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise MatlabEngineInfraError("Python 3.7 MATLAB worker is not started")
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except OSError as exc:
            err = ""
            if proc.stderr is not None:
                err = proc.stderr.read()
            raise MatlabEngineInfraError(
                f"Python 3.7 MATLAB worker I/O failed: {exc}; stderr={err!r}"
            ) from exc
        if not line:
            err = ""
            if proc.stderr is not None:
                err = proc.stderr.read()
            raise MatlabEngineInfraError(f"Python 3.7 MATLAB worker exited; stderr={err!r}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MatlabEngineInfraError(
                f"Python 3.7 MATLAB worker returned non-JSON: {line!r}"
            ) from exc
        if not isinstance(reply, dict):
            raise MatlabEngineInfraError("Python 3.7 MATLAB worker reply must be an object")
        return reply

    def energy_filter_v200_from_spatial(
        self,
        chunk: np.ndarray,
        *,
        matching_kernel_string: str,
        radius: float,
        vessel_wall: float,
        microns_per_pixel: np.ndarray,
        pixels_per_sigma_psf: np.ndarray,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        z0: int,
        z1: int,
        gaussian_to_ideal_ratio: float,
        spherical_to_annular_ratio: float,
        scales_per_octave: float,
    ) -> np.ndarray:
        if self._tmpdir is None:
            raise MatlabEngineInfraError("Python 3.7 MATLAB worker tempdir missing")
        work = Path(self._tmpdir.name)
        chunk_path = work / "chunk.npy"
        out_path = work / "energy.npy"
        np.save(chunk_path, np.asfortranarray(np.asarray(chunk, dtype=np.float64)))
        reply = self._request(
            {
                "op": "energy_filter",
                "chunk_npy": str(chunk_path),
                "out_npy": str(out_path),
                "matching_kernel_string": matching_kernel_string,
                "radius": float(radius),
                "vessel_wall": float(vessel_wall),
                "microns_per_pixel": np.asarray(microns_per_pixel, dtype=np.float64).tolist(),
                "pixels_per_sigma_psf": np.asarray(pixels_per_sigma_psf, dtype=np.float64).tolist(),
                "y0": int(y0),
                "y1": int(y1),
                "x0": int(x0),
                "x1": int(x1),
                "z0": int(z0),
                "z1": int(z1),
                "gaussian_to_ideal_ratio": float(gaussian_to_ideal_ratio),
                "spherical_to_annular_ratio": float(spherical_to_annular_ratio),
                "scales_per_octave": float(scales_per_octave),
            }
        )
        if not reply.get("ok"):
            raise MatlabEngineInfraError(
                f"energy_filter_V200 worker call failed: {reply.get('error')}"
            )
        energy = np.load(out_path)
        return cast("np.ndarray", np.ascontiguousarray(np.asarray(energy, dtype=np.float64)))


def energy_filter_v200_from_spatial(
    session: MatlabEngineSession | MatlabEnginePy37Worker,
    chunk: np.ndarray,
    *,
    matching_kernel_string: str = DEFAULT_MATCHING_KERNEL,
    radius: float,
    vessel_wall: float = 0.0,
    microns_per_pixel: np.ndarray,
    pixels_per_sigma_psf: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    z0: int,
    z1: int,
    gaussian_to_ideal_ratio: float,
    spherical_to_annular_ratio: float,
    scales_per_octave: float,
) -> np.ndarray:
    """Call ``stretch_energy_filter_v200`` (FFT + ``energy_filter_V200``)."""
    if isinstance(session, MatlabEnginePy37Worker):
        return session.energy_filter_v200_from_spatial(
            chunk,
            matching_kernel_string=matching_kernel_string,
            radius=radius,
            vessel_wall=vessel_wall,
            microns_per_pixel=microns_per_pixel,
            pixels_per_sigma_psf=pixels_per_sigma_psf,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            z0=z0,
            z1=z1,
            gaussian_to_ideal_ratio=gaussian_to_ideal_ratio,
            spherical_to_annular_ratio=spherical_to_annular_ratio,
            scales_per_octave=scales_per_octave,
        )
    ml_chunk = numpy_to_matlab_double(chunk)
    microns = numpy_to_matlab_double(np.asarray(microns_per_pixel, dtype=np.float64).reshape(-1))
    psf = numpy_to_matlab_double(np.asarray(pixels_per_sigma_psf, dtype=np.float64).reshape(-1))
    energy = session.call(
        "stretch_energy_filter_v200",
        ml_chunk,
        matching_kernel_string,
        float(radius),
        float(vessel_wall),
        microns,
        psf,
        float(y0),
        float(y1),
        float(x0),
        float(x1),
        float(z0),
        float(z1),
        float(gaussian_to_ideal_ratio),
        float(spherical_to_annular_ratio),
        float(scales_per_octave),
    )
    shape = (int(y1) - int(y0) + 1, int(x1) - int(x0) + 1, int(z1) - int(z0) + 1)
    return matlab_double_to_numpy(energy, shape)


@contextmanager
def stretch_engine_float_body_session(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Bind Energy float body for one job; no-op when backend is NumPy."""
    backend = str(config.get("energy_float_backend", "numpy")).strip().lower()
    if backend != "matlab_engine":
        yield config
        return
    bound = dict(config)
    bound["n_jobs"] = 1
    matlab_exe = bound.get("matlab_exe")
    vectorization_root = (
        Path(bound["vectorization_root"])
        if bound.get("vectorization_root")
        else default_vectorization_root()
    )
    helper_dir = HELPER_DIR
    session: MatlabEngineSession | MatlabEnginePy37Worker
    if current_python_supports_r2019a_engine():
        verify_matlab_engine_prerequisites(
            matlab_exe=matlab_exe,
            vectorization_root=vectorization_root,
        )
        session = MatlabEngineSession(
            matlab_exe=matlab_exe,
            vectorization_root=vectorization_root,
        )
        session.start()
        session.engine.addpath(str(helper_dir), nargout=0)
    else:
        resolve_matlab_root(matlab_exe)
        if not (vectorization_root / "source").is_dir():
            raise MatlabEngineInfraError(
                f"Vectorization-Public source missing: {vectorization_root / 'source'}"
            )
        python37 = resolve_python37_executable()
        if python37 is None:
            raise MatlabEngineInfraError(
                "MATLAB Engine for Python (R2019a) needs Python 3.7; set "
                f"{STRETCH_PY37_ENV} to an isolated 3.7 interpreter with matlab.engine "
                "(incomplete_infra)"
            )
        session = MatlabEnginePy37Worker(
            python37,
            vectorization_root=vectorization_root,
            helper_dir=helper_dir,
        )
        session.start()
    bound["_stretch_engine_float_body_bound"] = True
    bound["_stretch_engine_session"] = session
    try:
        yield bound
    finally:
        session.quit()
        bound["_stretch_engine_float_body_bound"] = False
        bound.pop("_stretch_engine_session", None)
