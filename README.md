# Atomic DFT Solver (Kohn–Sham)

This project implements a numerical solver for the radial Kohn–Sham equations using:

- Numerov method for ODE integration
- Shooting method for eigenvalue search
- Self-consistent Hartree and exchange-correlation potentials

## Features
- Bound state search via Brent root finding
- Hartree potential via Poisson equation
- Exchange-correlation (LDA-style)
- Self-consistent field (SCF) loop

## Run

python experiments/run_atom.py

## Example Output
- Electron density
- Total energy convergence

## Physics Background
Solves:
- Radial Schrödinger equation
- Poisson equation for Hartree potential
- Exchange-correlation potential (LDA)

## Author
Tiffany Chen