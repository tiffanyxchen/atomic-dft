import numpy as np
from src.numerov import NumerovUP

def FuncForHartree(y, r, rhoSpline):
    return [y[1], -8 * np.pi * r * rhoSpline(r)]


def HartreeU(R, rho, Zatom):
    U2 = NumerovUP(-8*np.pi*R*rho, 0.0, 0.5, R[1]-R[0])
    alpha2 = (2*Zatom - U2[-1]) / R[-1]
    U2 += alpha2 * R
    return U2