import numpy as np

def si_sdr(y, x):
    alpha = np.dot(y, x) / (np.linalg.norm(y) ** 2)
    frac = (np.linalg.norm(alpha * y) ** 2) /  (np.linalg.norm(alpha * y - x) ** 2)
    return 10 * np.log10(frac)