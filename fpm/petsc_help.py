import sys
import petsc4py

petsc4py.init(['-pc_type', 'hypre', '-help'])

from petsc4py import PETSc

ksp = PETSc.KSP().create()
ksp.setFromOptions()
# ksp.view()
