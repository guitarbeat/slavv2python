import joblib.numpy_pickle
import pickle
import io
import os
import builtins
import numpy
import pandas
import collections

class SafeNumpyUnpickler(joblib.numpy_pickle.NumpyUnpickler):
    """
    A safer unpickler that restricts which global objects can be loaded.
    This helps mitigate arbitrary code execution vulnerabilities in untrusted pickle data.
    """
    def find_class(self, module, name):
        # Whitelist of safe modules
        # We allow submodules of these packages
        ALLOWED_PACKAGES = {
            'numpy',
            'sklearn',
            'scipy',
            'joblib',
            'collections',
            'pandas',
            'slavv',
            '_codecs',
            'copy_reg'
        }

        # Check if the module belongs to an allowed package
        root_module = module.split('.')[0]
        if root_module in ALLOWED_PACKAGES:
             return super().find_class(module, name)

        # Strict checking for builtins
        if module == 'builtins' or module == '__builtin__':
            ALLOWED_BUILTINS = {
                'dict', 'list', 'set', 'frozenset', 'tuple',
                'float', 'int', 'str', 'bool', 'slice', 'range', 'complex',
                'bytearray', 'bytes', 'NoneType', 'object'
            }
            if name in ALLOWED_BUILTINS:
                return super().find_class(module, name)

        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")

def safe_load(filename, mmap_mode=None):
    """
    Safely load a pickle file using a whitelist of allowed modules.
    Replaces joblib.load for untrusted inputs.

    Args:
        filename: str or file-like object. The file to load.
        mmap_mode: str, optional. The memory-mapping mode to use for numpy arrays.

    Returns:
        The deserialized object.

    Raises:
        pickle.UnpicklingError: If the input contains forbidden classes.
    """
    if hasattr(filename, 'read'):
        fobj = filename
        # Joblib's NumpyUnpickler expects a filename for mmap,
        # but if we pass a file handle, it reads from it.
        # We pass a dummy name or the name attribute if available.
        fname = getattr(fobj, 'name', None)
        # ensure_native_byte_order is typically False for loading unless specific checks needed
        # We pass False as the 3rd positional argument
        unpickler = SafeNumpyUnpickler(fname, fobj, False, mmap_mode=mmap_mode)
        return unpickler.load()
    else:
        with open(filename, 'rb') as f:
            unpickler = SafeNumpyUnpickler(filename, f, False, mmap_mode=mmap_mode)
            return unpickler.load()
