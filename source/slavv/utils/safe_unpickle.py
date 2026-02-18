import pickle
import joblib
import os
import builtins
import collections
import types
from joblib.numpy_pickle import NumpyUnpickler
from joblib.numpy_pickle_utils import _detect_compressor, _COMPRESSORS

class SafeNumpyUnpickler(NumpyUnpickler):
    """
    A safer version of NumpyUnpickler that restricts which classes can be loaded.
    This prevents arbitrary code execution from malicious pickle files.
    """
    def find_class(self, module, name):
        # Allow only safe modules and types
        if not self._is_allowed(module, name):
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")

        return super().find_class(module, name)

    def _is_allowed(self, module, name):
        # Allow specific safe types from builtins
        if module == 'builtins':
            return name in {
                'dict', 'list', 'set', 'frozenset', 'tuple', 'int', 'float', 'complex',
                'bool', 'str', 'bytes', 'bytearray', 'range', 'slice', 'NoneType',
                'Ellipsis', 'NotImplemented', 'object'
            }

        allowed_modules = {
            'copyreg',
            'collections',
            'types',
            'numpy',
            'sklearn',
            'joblib',
            '_codecs',
            'pandas',
            'scipy',
        }

        # Check if module is exactly one of the allowed modules
        if module in allowed_modules:
            return True

        # Check if module is a submodule of an allowed module
        for allowed in allowed_modules:
            if module.startswith(allowed + '.'):
                return True

        return False

def safe_load(filename, mmap_mode=None):
    """
    Safely load a joblib file using a restricted unpickler.

    Parameters
    ----------
    filename: str, os.PathLike, or file object
        The file object or path of the file to be loaded.
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk.

    Returns
    -------
    result: any
        The object stored in the file.
    """
    file_handle = None
    own_file_handle = False

    try:
        # Open file if path provided
        if isinstance(filename, (str, os.PathLike)):
            filename = str(filename)
            file_handle = open(filename, 'rb')
            own_file_handle = True
        elif hasattr(filename, 'read'):
            file_handle = filename
            filename = getattr(file_handle, 'name', None)
        else:
            raise ValueError("filename must be a string, PathLike, or a file-like object")

        # Handle compression
        compressor = _detect_compressor(file_handle)

        if compressor == 'compat':
            # 'compat' indicates zlib stream or older format, handle as is or wrap?
            # joblib usually handles this. NumpyUnpickler handles uncompressed streams.
            # If it's really zlib compressed, we might need zlib.
            pass
        elif compressor != 'not-compressed':
             # Use joblib's compressor wrapper to get a decompressed stream
             if compressor in _COMPRESSORS:
                 wrapper = _COMPRESSORS[compressor]
                 # wrapper.decompressor_file returns a file-like object
                 file_handle = wrapper.decompressor_file(file_handle)

        # Create unpickler
        # Note: We pass original filename (if any) or None for mmap_mode reference
        unpickler = SafeNumpyUnpickler(filename, file_handle, ensure_native_byte_order=False, mmap_mode=mmap_mode)
        return unpickler.load()

    finally:
        if own_file_handle and file_handle is not None:
            file_handle.close()
