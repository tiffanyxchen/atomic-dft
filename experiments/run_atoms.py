import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from src.hartree import HartreeU
from src.schrodinger import FindBoundStates
from src.density import ChargeDensity
from src.utils import rs, cmpKey
from src.excor import ExchangeCorrelation


def main():
    Zatom = 8
    mixr = 0.5
    nmax = 4

    R = np.linspace(1e-8, 10, 2**14+1)

    E0 = -1.2 * Zatom**2
    Eshift = 0.1
    Esearch = -np.logspace(-4, np.log10(-E0+Eshift), 500)[::-1] + Eshift

    exc = ExchangeCorrelation()
    Uks = -2*Zatom*np.ones(len(R))

    Eold = 0
    Etol = 1e-7

    for itt in range(100):
        Bnd = []
        for l in range(nmax-1):
            Bnd += FindBoundStates(R, l, nmax-l, Esearch, Uks)

        Bnd = sorted(Bnd, key=cmpKey)

        rho_new, Ebs = ChargeDensity(Bnd, R, Zatom, Uks)

        if itt == 0:
            rho = rho_new
        else:
            rho = rho_new * mixr + (1-mixr)*rho_old

        rho_old = rho.copy()

        U2 = HartreeU(R, rho, Zatom)

        Vxc = np.array([2*exc.Vc(rs(rh)) + 2*exc.Vx(rs(rh)) for rh in rho])
        Uks = U2 - 2*Zatom + Vxc*R

        ExcVxc = np.array([2*exc.EcVc(rs(rh)) + 2*exc.ExVx(rs(rh)) for rh in rho])

        pot = (ExcVxc*R - 0.5*U2)*R*rho*4*np.pi
        epot = integrate.simpson(pot, x=R)

        Etot = epot + Ebs

        print(f"Iteration {itt}, Etot = {Etot/2:.6f} Hartree")

        if itt > 0 and abs(Etot - Eold) < Etol:
            break

        Eold = Etot

    plt.plot(R, rho * (4*np.pi*R**2))
    plt.xlabel("r")
    plt.ylabel("Radial density")
    plt.show()


if __name__ == "__main__":
    main()