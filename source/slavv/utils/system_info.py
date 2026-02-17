"""System information utilities for performance benchmarking."""

import platform
from datetime import datetime
from typing import Dict, Any, Optional


def get_system_info() -> Dict[str, Any]:
    """
    Capture comprehensive system information for benchmarking.
    
    Returns
    -------
    dict
        System information including CPU, memory, GPU, OS, and software versions.
    """
    info = {
        "cpu": _get_cpu_info(),
        "memory": _get_memory_info(),
        "gpu": _get_gpu_info(),
        "os": _get_os_info(),
        "python": _get_python_info(),
        "timestamp": datetime.now().isoformat()
    }
    
    return info


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    cpu_info = {
        "model": platform.processor() or "Unknown",
        "cores": None,
        "threads": None,
        "frequency_mhz": None
    }
    
    # Try to get detailed CPU info using psutil
    try:
        import psutil
        cpu_info["cores"] = psutil.cpu_count(logical=False)
        cpu_info["threads"] = psutil.cpu_count(logical=True)
        
        # Get CPU frequency if available
        freq = psutil.cpu_freq()
        if freq:
            cpu_info["frequency_mhz"] = round(freq.current, 2)
    except ImportError:
        pass
    
    # Try py-cpuinfo for more detailed model name
    try:
        import cpuinfo
        cpu_data = cpuinfo.get_cpu_info()
        if cpu_data and "brand_raw" in cpu_data:
            cpu_info["model"] = cpu_data["brand_raw"]
    except ImportError:
        pass
    
    return cpu_info


def _get_memory_info() -> Dict[str, Any]:
    """Get memory information."""
    memory_info = {
        "total_gb": None,
        "available_gb": None
    }
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        memory_info["total_gb"] = round(mem.total / (1024**3), 2)
        memory_info["available_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        pass
    
    return memory_info


def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    gpu_info = {
        "available": False,
        "devices": []
    }
    
    # Check for CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["devices"] = [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ]
            return gpu_info
    except ImportError:
        pass
    
    # Check for CuPy (NVIDIA)
    try:
        import cupy as cp
        if cp.cuda.is_available():
            gpu_info["available"] = True
            # CuPy doesn't provide easy device name access, so just note it's available
            gpu_info["devices"] = [f"CUDA Device {i}" for i in range(cp.cuda.runtime.getDeviceCount())]
            return gpu_info
    except ImportError:
        pass
    
    return gpu_info


def _get_os_info() -> Dict[str, Any]:
    """Get operating system information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine()
    }


def _get_python_info() -> Dict[str, Any]:
    """Get Python version information."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler()
    }


def get_matlab_info(matlab_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get MATLAB version information.
    
    Parameters
    ----------
    matlab_path : str, optional
        Path to MATLAB executable. If provided, attempts to extract version.
    
    Returns
    -------
    dict
        MATLAB version information if available.
    """
    matlab_info = {
        "version": None,
        "path": matlab_path
    }
    
    if matlab_path:
        # Try to extract version from path (e.g., "R2019a" from path)
        import re
        match = re.search(r'R\d{4}[ab]', matlab_path)
        if match:
            matlab_info["version"] = match.group(0)
    
    return matlab_info
