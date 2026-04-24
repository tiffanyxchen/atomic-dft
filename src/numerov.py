import numpy as np
from numba import jit

@jit(nopython=True)
def Numerov(f, x0, dx, dh):
    x = np.zeros(len(f))
    x[0] = x0
    x[1] = x0 + dh * dx
    h2  = dh**2
    h12 = h2 / 12.

    w0 = x0   * (1 - h12 * f[0])
    w1 = x[1] * (1 - h12 * f[1])
    xi = x[1]
    fi = f[1]

    for i in range(2, len(f)):
        w2  = 2 * w1 - w0 + h2 * fi * xi
        fi  = f[i]
        xi  = w2 / (1 - h12 * fi)
        x[i]= xi
        w0, w1 = w1, w2

    return x


@jit(nopython=True)
def NumerovUP(U, x0, dx, dh):
    x = np.zeros(len(U))
    x[0] = x0
    x[1] = dx*dh + x0

    h2 = dh*dh
    h12 = h2/12

    w0 = x[0] - h12*U[0]
    w1 = x[1] - h12*U[1]
    Ui = U[1]

    for i in range(2, len(U)):
        w2 = 2*w1 - w0 + h2*Ui
        Ui = U[i]
        xi = w2 + h12*Ui
        x[i] = xi
        w0, w1 = w1, w2

    return x