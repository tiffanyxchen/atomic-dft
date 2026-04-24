import numpy as np
from src.schrodinger import ComputeSchrod

def ChargeDensity(bst, R, Zatom, Uks):
    rho = np.zeros(len(R))
    N   = 0.
    Ebs = 0.

    for (l, En) in bst:
        ur = ComputeSchrod(En, R, l, Uks)
        dN = 2*(2*l+1)

        if N + dN <= Zatom:
            ferm = 1.
        else:
            ferm = (Zatom-N)/float(dN)

        drho = ur**2 * ferm * dN/(4*np.pi*R**2)
        rho += drho

        N   += dN
        Ebs += En * dN * ferm

        if N >= Zatom:
            break

    return rho, Ebs