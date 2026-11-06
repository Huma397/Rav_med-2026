"""Logic self-test of the ERS arithmetic and bands (no model download needed)."""
import numpy as np
from rav_med import ers_from_parts, band
e = ers_from_parts([.9, .6, .3], [.0, .5, .9]); assert np.allclose(e, [.9, .1, -.6])
assert [band(x) for x in e] == ["SAFE", "RISKY", "UNSAFE"] and band(0.2) == "SAFE" and band(0.0) == "RISKY" and band(-0.0001) == "UNSAFE"
print("selftest OK")
