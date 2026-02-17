import joblib
import joblib.numpy_pickle
import inspect
import sys

print(f"Joblib version: {joblib.__version__}")
try:
    sig = inspect.signature(joblib.numpy_pickle.NumpyUnpickler.__init__)
    print(f"NumpyUnpickler.__init__ signature: {sig}")
except Exception as e:
    print(f"Could not get signature: {e}")

# Check if ensure_native_byte_order is in parameters
if 'ensure_native_byte_order' in sig.parameters:
    print("ensure_native_byte_order IS present.")
else:
    print("ensure_native_byte_order IS NOT present.")
