# Iterative block solvers for PorePy using PETSc
This repository implements linear solvers for the open-source software
[PorePy](https://github.com/pmgbergen/porepy), using [PETSc](https://petsc.org/) as the
linear algebra backend.

The repository is a fork of https://github.com/pmgbergen/FTHM-Solver. 
Whereas the upstream repository is used for prototyping of solvers and production of
papers, this repository aims to make the solvers robust and easily applicable in
general PorePy models.

# Installation
This package can be installed with

    pip install -e .

It is assumed that working installations of PorePy and PETSc are available.


# Understand the code

See the [tutorial](tutorial.ipynb).
