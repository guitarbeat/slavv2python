import sys
import bisect
print(sys.version)
try:
    bisect.bisect_right([1, 2, 3], 2, key=lambda x: x)
    print("bisect key supported")
except TypeError as e:
    print(f"bisect key error: {e}")
except Exception as e:
    print(f"Other error: {e}")
