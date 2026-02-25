"""
Safe Unpickler for SLAVV.

This module provides a secure unpickler that restricts loaded classes to a
predefined allowlist, preventing arbitrary code execution from malicious pickles.
"""
import pickle
import logging
import joblib.numpy_pickle

logger = logging.getLogger(__name__)

# Allowed modules
SAFE_MODULE_PREFIXES = {
    'numpy',
    'sklearn',
    'pandas',
    'scipy',
    'joblib.numpy_pickle',
    '_codecs',
    'copyreg',
    'collections',
    'types'
}

class SafeNumpyUnpickler(joblib.numpy_pickle.NumpyUnpickler):
    """
    Restricted NumpyUnpickler that enforces an allowlist of modules.
    """

    def find_class(self, module, name):
        """
        Check if the module is in the allowlist.
        """
        # Allow primitives and basic types
        if module == "builtins":
            if name in {"dict", "list", "set", "tuple", "object", "int", "float", "str",
                        "bool", "bytes", "bytearray", "complex", "range", "slice"}:
                 return super().find_class(module, name)

        # Check against safe prefixes
        for prefix in SAFE_MODULE_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                 return super().find_class(module, name)

        # Log and raise error
        msg = f"Forbidden pickle module: {module}.{name}"
        logger.error(msg)
        raise pickle.UnpicklingError(msg)

    def __init__(self, filename, file_handle, ensure_native_byte_order=False, mmap_mode=None):
        """
        Initialize SafeNumpyUnpickler.
        Matches signature of joblib.numpy_pickle.NumpyUnpickler (v1.5.3).
        """
        super().__init__(filename, file_handle, ensure_native_byte_order=ensure_native_byte_order, mmap_mode=mmap_mode)

def safe_load(filename, mmap_mode=None):
    """
    Safely load a pickle file using SafeNumpyUnpickler.

    Args:
        filename: Path to the file to load.
        mmap_mode: Memory mapping mode (optional).

    Returns:
        The unpickled object.
    """
    import gzip
    import bz2
    import lzma

    filename = str(filename)

    # Simple magic number detection
    try:
        with open(filename, 'rb') as f_peek:
            header = f_peek.read(6)
    except OSError as e:
        raise OSError(f"Could not open file {filename}: {e}")

    if header.startswith(b'\x1f\x8b'):
        f = gzip.open(filename, 'rb')
    elif header.startswith(b'BZh'):
        f = bz2.open(filename, 'rb')
    elif header.startswith(b'\xfd7zXZ'):
        f = lzma.open(filename, 'rb')
    else:
        f = open(filename, 'rb')

    try:
        # Create unpickler
        unpickler = SafeNumpyUnpickler(filename, f, ensure_native_byte_order=False, mmap_mode=mmap_mode)
        return unpickler.load()
    finally:
        f.close()
