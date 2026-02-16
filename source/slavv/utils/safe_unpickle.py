import os
import pickle
import io
import logging
import joblib
from joblib.numpy_pickle import NumpyUnpickler

logger = logging.getLogger(__name__)

SAFE_BUILTINS = {
    'range',
    'complex',
    'set',
    'frozenset',
    'slice',
    'dict',
    'list',
    'tuple',
    'str',
    'bytes',
    'bytearray',
    'bool',
    'int',
    'float',
    'NoneType',
}

SAFE_MODULE_PREFIXES = [
    'numpy',
    'sklearn',
    'scipy',
    'pandas',
    'joblib',
    'collections',
    'copyreg',
    'datetime',
    '_codecs',
    'slavv',
]

class SafeUnpickler(NumpyUnpickler):
    """
    A safer unpickler that restricts allowable classes to a whitelist.
    Inherits from joblib.numpy_pickle.NumpyUnpickler to support numpy arrays.
    """
    def find_class(self, module, name):
        # Only allow safe builtins
        if module == 'builtins':
            if name in SAFE_BUILTINS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"Security: Forbidden builtin '{name}'")

        # Check against allowed prefixes
        for prefix in SAFE_MODULE_PREFIXES:
            if module == prefix or module.startswith(prefix + '.'):
                return super().find_class(module, name)

        # Log and raise error for anything else
        logger.warning(f"Blocked unpickling of forbidden module: {module}.{name}")
        raise pickle.UnpicklingError(f"Security: Forbidden module '{module}'")

def safe_load(file_or_path):
    """
    Safely load a joblib/pickle file by restricting allowable classes.

    Args:
        file_or_path: Path to file or file-like object.

    Returns:
        The unpickled object.

    Raises:
        pickle.UnpicklingError: If the file contains forbidden classes or is invalid.
    """
    if isinstance(file_or_path, (str, bytes, os.PathLike)):
        filename = str(file_or_path)
        with open(filename, 'rb') as f:
            return _safe_load_stream(f, filename=filename)
    else:
        # Assume file-like object
        filename = getattr(file_or_path, 'name', None)
        if filename is None:
            # NumpyUnpickler requires a filename, provide a placeholder
            filename = "<stream>"
        return _safe_load_stream(file_or_path, filename=filename)

def _safe_load_stream(f, filename):
    try:
        # NumpyUnpickler signature varies by version, but typically requires:
        # (self, filename, file_handle, ensure_native_byte_order, mmap_mode=None)
        # We use keyword arguments to be explicit and robust.
        unpickler = SafeUnpickler(filename=filename, file_handle=f,
                                 ensure_native_byte_order=False, mmap_mode=None)
        return unpickler.load()
    except TypeError as e:
        logger.warning(f"SafeUnpickler kwarg init failed: {e}. Trying positional args.")
        try:
             # Fallback for older joblib versions or different signatures
             # Try (filename, file_handle)
             unpickler = SafeUnpickler(filename, f)
             return unpickler.load()
        except TypeError:
             # Try (filename, file_handle, mmap_mode) - unlikely but possible
             unpickler = SafeUnpickler(filename, f, None)
             return unpickler.load()
    except Exception as e:
        logger.error(f"Failed to load model safely: {e}")
        raise
