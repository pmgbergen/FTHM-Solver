import sys
import time

import numpy as np
import petsc4py
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

petsc4py.init(sys.argv)

from petsc4py import PETSc


class OmegaInv:
    def __init__(self, solve_momentum, solve_mass, C1, C2):
        self.B_inv = solve_momentum
        self.S_A_inv = solve_mass
        self.C1 = C1
        self.C2 = C2
        self.sep = solve_momentum.shape[0]
        shape = solve_momentum.shape[0] + solve_mass.shape[0]
        self.shape = shape, shape

    def dot(self, x):
        x_B, x_A = x[: self.sep], x[self.sep :]
        tmp_B = self.B_inv.dot(x_B)
        tmp_A = x_A - self.C1.dot(tmp_B)
        y_A = self.S_A_inv.dot(tmp_A)
        y = np.zeros_like(x)
        y[self.sep :] = y_A
        y[: self.sep] = self.B_inv.dot(x_B - self.C2.dot(y_A))
        return y


def cond(mat):
    return np.linalg.cond(mat.A)


def inv(mat):
    return scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(mat))


def pinv(mat):
    return scipy.sparse.csr_matrix(np.linalg.pinv(mat.A))


def condest(mat):
    mat = mat.tocsr()
    data = abs(mat.data)
    return data.max() / data.min()


class UpperBlockPreconditioner:

    def __init__(self, F_inv, Omega_inv, Phi):
        self.F_inv = F_inv
        self.Omega_inv = Omega_inv
        self.Phi = Phi
        shape = F_inv.shape[0] + Omega_inv.shape[0]
        self.shape = shape, shape
        self.sep = F_inv.shape[0]

    def dot(self, x):
        x_K, x_Omega = x[: self.sep], x[self.sep :]
        y = np.zeros_like(x)
        y_Omega = self.Omega_inv.dot(x_Omega)
        tmp = x_K - self.Phi.dot(y_Omega)
        y[: self.sep] = self.F_inv.dot(tmp)
        y[self.sep :] = y_Omega
        return y


def make_permutations(row_dof, order):
    indices = np.concatenate([row_dof[i] for i in order])
    perm = scipy.sparse.eye(indices.size).tocsr()
    perm.indices[:] = indices
    return perm


class TimerContext:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time


class PetscPC:
    def __init__(self, mat=None, block_size=1) -> None:
        self.pc = PETSc.PC().create()

        self.petsc_mat = PETSc.Mat()
        self.petsc_x = PETSc.Vec()
        self.petsc_b = PETSc.Vec()
        self.pc.setFromOptions()

        self.shape: tuple[int, int]
        if mat is not None:
            self.set_operator(mat, block_size=block_size)

    def set_operator(self, mat, block_size=1):
        self.shape = mat.shape
        self.petsc_mat.destroy()
        self.petsc_x.destroy()
        self.petsc_b.destroy()
        self.petsc_mat.createAIJ(
            size=mat.shape, csr=(mat.indptr, mat.indices, mat.data), bsize=block_size
        )
        self.petsc_b = self.petsc_mat.createVecLeft()
        self.petsc_x = self.petsc_mat.createVecLeft()
        self.pc.setOperators(self.petsc_mat)
        self.pc.setUp()

    def __del__(self):
        self.pc.destroy()
        self.petsc_mat.destroy()
        self.petsc_b.destroy()
        self.petsc_x.destroy()

    def dot(self, b):
        self.petsc_x.set(0.0)
        self.petsc_b.setArray(b)
        self.pc.apply(self.petsc_b, self.petsc_x)
        res = self.petsc_x.getArray()
        return res

    def get_matrix(self):
        indptr, indices, data = self.petsc_mat.getValuesCSR()
        return scipy.sparse.csr_matrix((data, indices, indptr))


class PetscAMGMechanics(PetscPC):
    def __init__(self, dim: int, mat=None) -> None:
        options = PETSc.Options()
        # options["pc_type"] = "gamg"
        # options['pc_gamg_agg_nsmooths'] = 1
        # options["mg_levels_ksp_type"] = "chebyshev"
        # options["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        # options["mg_levels_pc_type"] = "jacobi"

        options["pc_type"] = "hypre"
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_max_iter"] = 1
        options["pc_hypre_boomeramg_cycle_type"] = "W"
        options["pc_hypre_boomeramg_truncfactor"] = 0.3
        # options.setValue('pc_hypre_boomeramg_relax_type_all', 'Chebyshev')
        # options.setValue('pc_hypre_boomeramg_smooth_type', 'Pilut')

        # options["pc_hypre_boomeramg_strong_threshold"] = 0.7
        # options["pc_hypre_boomeramg_agg_nl"] = 4
        # options["pc_hypre_boomeramg_agg_num_paths"] = 5
        # options["pc_hypre_boomeramg_max_levels"] = 25
        # options["pc_hypre_boomeramg_coarsen_type"] = "HMIS"
        # options["pc_hypre_boomeramg_interp_type"] = "ext+i"
        # options["pc_hypre_boomeramg_P_max"] = 2

        super().__init__(mat=mat, block_size=dim)


class PetscAMGFlow(PetscPC):
    def __init__(self, mat=None) -> None:
        options = PETSc.Options()

        options["pc_type"] = "hypre"
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_max_iter"] = 1
        options["pc_hypre_boomeramg_cycle_type"] = "W"
        options["pc_hypre_boomeramg_truncfactor"] = 0.3

        super().__init__(mat=mat, block_size=1)


class PetscILU(PetscPC):
    def __init__(self, mat=None, factor_levels: int = 0) -> None:
        options = PETSc.Options()
        options.setValue("pc_type", "ilu")
        options.setValue("pc_factor_levels", factor_levels)
        options.setValue("pc_factor_diagonal_fill", None)  # Doesn't affect
        options.setValue("pc_factor_mat_ordering_type", "rcm")
        options.setValue("pc_factor_nonzeros_along_diagonal", None)
        super().__init__(mat=mat)


class PetscPythonPC:

    def __init__(self, pc):
        self.pc = pc

    def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
        """Apply the preconditioner on vector b, return in x."""
        result = self.pc.dot(b.getArray())
        x.setArray(result)


class PetscGMRES:
    def __init__(self, mat, pc: PETSc.PC | None = None) -> None:
        self.shape = mat.shape
        restart = 50

        self.ksp = PETSc.KSP().create()
        options = PETSc.Options()
        options.setValue("ksp_type", "gmres")
        # options.setValue("ksp_type", "bcgs")
        options.setValue("ksp_rtol", 1e-10)
        options.setValue("ksp_max_it", 20 * restart)
        options.setValue("ksp_norm_type", "unpreconditioned")
        options.setValue("ksp_gmres_restart", restart)
        # options.setValue("ksp_pc_side", "left")
        if pc is None:
            options.setValue("pc_type", "none")

        self.ksp.setFromOptions()
        self.ksp.setComputeEigenvalues(True)
        self.ksp.setConvergenceHistory()
        self.pc = PETSc.PC()
        if pc is not None:
            self.pc.createPython(PetscPythonPC(pc))
            self.ksp.setPC(self.pc)

        self.petsc_mat = PETSc.Mat().createAIJ(
            size=mat.shape, csr=(mat.indptr, mat.indices, mat.data)
        )
        self.ksp.setOperators(self.petsc_mat)
        self.ksp.setUp()

        self.petsc_x = self.petsc_mat.createVecLeft()
        self.petsc_b = self.petsc_mat.createVecLeft()

    def __del__(self):
        self.ksp.destroy()
        self.pc.destroy()
        self.petsc_mat.destroy()
        self.petsc_x.destroy()
        self.petsc_b.destroy()

    def solve(self, b):
        self.petsc_b.setArray(b)
        self.petsc_x.set(0.0)
        self.ksp.solve(self.petsc_b, self.petsc_x)
        res = self.petsc_x.getArray()
        return res

    def dot(self, b):
        return self.solve(b)

    def get_residuals(self):
        return self.ksp.getConvergenceHistory()


class PetscJacobi(PetscPC):
    def __init__(self, mat=None) -> None:
        options = PETSc.Options()
        options["pc_type"] = "jacobi"
        super().__init__(mat=mat)


class PetscSOR(PetscPC):
    def __init__(self, mat=None) -> None:
        options = PETSc.Options()
        options["pc_type"] = "sor"
        options["pc_type_symmetric"] = True
        super().__init__(mat=mat)


def extract_diag_inv(mat):
    diag = mat.diagonal()
    ones = scipy.sparse.eye(mat.shape[0], format="csr")
    diag_inv = 1 / diag
    ones.data[:] = diag_inv
    return ones


def inv_block_diag(mat, nd: int):
    if nd == 1:
        return extract_diag_inv(mat)
    if nd == 2:
        return inv_block_diag_2x2(mat)
    if nd == 3:
        diag = diag_nd(mat, nd=3)
        if diag.nnz != mat.nnz:
            print('Matrix contained nondiagonal elements')
        return inv(diag)
    raise ValueError
    # print(f"{nd = } not implemented, using direct inverse")
    # return inv(mat)


def inv_block_diag_2x2(mat):
    ad = mat.diagonal()
    a = ad[::2]
    d = ad[1::2]
    b = mat.diagonal(k=1)[::2]
    c = mat.diagonal(k=-1)[::2]

    det = a * d - b * c

    assert abs(det).min() > 0

    diag = np.zeros_like(ad)
    diag[::2] = d / det
    diag[1::2] = a / det
    lower = np.zeros(ad.size - 1)
    lower[::2] = -c / det
    upper = np.zeros(ad.size - 1)
    upper[::2] = -b / det

    return scipy.sparse.diags([lower, diag, upper], offsets=[-1, 0, 1]).tocsr()


def lump_nd(mat, nd: int):
    result = scipy.sparse.lil_matrix(mat.shape)
    indices = np.arange(0, mat.shape[0], nd)
    for i in range(nd):
        for j in range(nd):
            indices_i = indices + i
            indices_j = indices + j
            I, J = np.meshgrid(
                indices_i, indices_j, copy=False, sparse=True, indexing="ij"
            )
            submat = mat[I, J]
            lump = np.array(submat.sum(axis=1)).ravel()
            result[indices_i, indices_j] = lump
    return result.tocsr()


def diag_nd(mat, nd: int):
    result = scipy.sparse.lil_matrix(mat.shape)
    indices = np.arange(0, mat.shape[0], nd)
    for i in range(nd):
        for j in range(nd):
            indices_i = indices + i
            indices_j = indices + j
            result[indices_i, indices_j] = mat[indices_i, indices_j]
    return result.tocsr()


def extract_rowsum_inv(mat):
    rowsum = np.array((mat).sum(axis=1)).squeeze()
    ones = scipy.sparse.eye(mat.shape[0], format="csr")
    diag_inv = 1 / rowsum
    ones.data[:] = diag_inv
    return ones


def reverse_cuthill_mckee(mat):
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    reorder = reverse_cuthill_mckee(mat)
    return mat[reorder][:, reorder]
