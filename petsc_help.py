import sys
import petsc4py
import scipy.sparse
import numpy as np


args = '-help'
# args = '-pc_type hypre -pc_hypre_type pilut -help'
# args = '-pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_levels 0 -sub_ksp_type preonly'
# args = '-pc_type gamg -pc_gamg_threshold 0.01 -mg_levels_ksp_max_it 5 -pc_gamg_agg_nsmooths 1'

petsc4py.init(args)

from petsc4py import PETSc

ksp = PETSc.KSP().create()

mat = scipy.sparse.csr_matrix(np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]))

petsc_mat = PETSc.Mat().createAIJ(
            size=mat.shape,
            csr=(mat.indptr, mat.indices, mat.data),
            bsize=1,
        )

ksp.setFromOptions()
ksp.setOperators(petsc_mat)
ksp.setUp()
# ksp.view()
