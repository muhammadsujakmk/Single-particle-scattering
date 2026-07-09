import numpy as np
from scipy import interpolate

def interpol(x, y, f):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)   # <- critical line

    # Ensure x is increasing (Fitpack expects sorted x)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    tck = interpolate.splrep(x, y, s=0)
    return interpolate.splev(f, tck)

def mat_cal(wvl, material):
    path = r'C:\My workspace\Personal Project\Mie Theory with Python\Mie Theory with Python\Single Sphere\Code for Single sphere\material library'
    data = np.loadtxt(path + material, comments="#")  # explicit

    x = np.asarray(data[:, 0], dtype=np.float64)  # nm
    n = np.asarray(data[:, 1], dtype=np.float64)
    k = np.asarray(data[:, 2], dtype=np.float64)

    wvl = np.asarray(wvl, dtype=np.float64)  # <- critical line (again)

    n_new = interpol(x, n, wvl)
    k_new = interpol(x, k, wvl)
    return n_new + 1j * k_new
