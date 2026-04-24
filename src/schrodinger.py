import numpy as np
from scipy import integrate, optimize
from numpy.polynomial import Polynomial
from src.numerov import Numerov

def fShrod(En, l, R, Uks):
    return (l*(l+1.)/R + Uks)/R - En


def ComputeSchrod(En, R, l, Uks):
    f = fShrod(En, l, R, Uks)
    ur = Numerov(f[::-1], 0.0, -1e-10, R[0]-R[1])[::-1]
    norm = integrate.simpson(ur**2, x=R)
    return ur / np.sqrt(abs(norm))


def Shoot(En, R, l, Uks):
    ur = ComputeSchrod(En, R, l, Uks)
    ur *= 1 / R**l
    poly = Polynomial.fit(R[:4], ur[:4], deg=3)
    return poly(0.0)


def FindBoundStates(R, l, nmax, Esearch, Uks):
    Ebnd = []
    u0 = Shoot(Esearch[0], R, l, Uks)

    for i in range(1, len(Esearch)):
        u1 = Shoot(Esearch[i], R, l, Uks)

        if u0 * u1 < 0:
            Ebound = optimize.brentq(
                Shoot, Esearch[i-1], Esearch[i],
                xtol=1e-15, args=(R, l, Uks)
            )
            Ebnd.append((l, Ebound))

            if len(Ebnd) > nmax:
                break

        u0 = u1

    return Ebnd