import numpy as np
from numba import jit

@jit(nopython=True)
def rs(rho):
    if rho < 1e-100:
        return 1e100
    return (3/(4*np.pi*rho))**(1/3.)


def cmpKey(x):
    return x[1] + x[0]/10000.