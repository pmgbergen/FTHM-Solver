import sys
from typing import Literal, TYPE_CHECKING

import numpy as np
from numba import njit
import petsc4py
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

if TYPE_CHECKING:
    from block_matrix import BlockMatrixStorage

petsc4py.init(sys.argv)

from petsc4py import PETSc


def assert_finite(vals, groups):
    pass
    # if not np.all(np.isfinite(vals)) or np.any(abs(vals).max() > 1e30):
    #     print("Divergence", groups)


class FieldSplit:
    def __init__(
        self,
        solve_momentum,
        solve_mass,
        C1,
        C2,
        groups_0=None,
        groups_1=None,
        factorization_type: Literal["full", "upper", "lower"] = "full",
    ):
        self.groups_0 = groups_0
        self.groups_1 = groups_1
        self.J00_inv = solve_momentum
        self.S11_inv = solve_mass
        self.J01 = C1
        self.J10 = C2
        self.sep = solve_momentum.shape[0]
        self.factorization_type: Literal["full", "upper", "lower"] = factorization_type
        shape = solve_momentum.shape[0] + solve_mass.shape[0]
        self.shape = shape, shape

    def dot(self, x):
        x_0, x_1 = x[: self.sep], x[self.sep :]
        if self.factorization_type != "upper":
            tmp_0 = self.J00_inv.dot(x_0)
            assert_finite(tmp_0, groups=self.groups_0)  # 1e+32
            tmp_1 = x_1 - self.J01.dot(tmp_0)
        else:
            tmp_0 = x_0
            tmp_1 = x_1
        y_1 = self.S11_inv.dot(tmp_1)
        assert_finite(y_1, groups=self.groups_1)
        y = np.zeros_like(x)
        y[self.sep :] = y_1
        if self.factorization_type != "lower":
            tmp_2 = self.J00_inv.dot(x_0 - self.J10.dot(y_1))
            assert_finite(tmp_2, groups=self.groups_0)
        else:
            tmp_2 = tmp_0
        y[: self.sep] = tmp_2
        return y


class RestrictedOperator:

    def __init__(self, mat: "BlockMatrixStorage", to_groups: list, prec):
        self.R = mat.make_restriction_matrix(to_groups).mat
        self.prec = prec(mat[to_groups])
        self.shape = mat.shape

    def dot(self, x: np.ndarray) -> np.ndarray:
        x_local = self.R @ x
        y_local = self.prec.dot(x_local)
        return self.R.T @ y_local


class TwoStagePreconditioner:

    def __init__(self, mat: "BlockMatrixStorage", stages: list):
        assert len(stages) == 2
        self.mat: "BlockMatrixStorage" = mat
        self.shape = mat.shape
        self.stages: list = stages

    def dot(self, x: np.ndarray) -> np.ndarray:
        y1 = self.stages[0].dot(x)
        r1 = x - self.mat.mat.dot(y1)
        y2 = self.stages[1].dot(r1)
        return y1 + y2


class BlockJacobi:
    def __init__(self, bmat, solve_A, solve_B, groups_0=None, groups_1=None):
        self.bmat = bmat
        self.solve_A = solve_A
        self.solve_B = solve_B
        self.sep = solve_A.shape[0]
        self.shape = bmat.shape
        self.groups_0 = groups_0
        self.groups_1 = groups_1

    def dot(self, x):
        x_0, x_1 = x[: self.sep], x[self.sep :]
        tmp_0 = self.solve_A.dot(x_0)
        assert_finite(tmp_0, groups=self.groups_0)  # 1e+32
        tmp_1 = self.solve_B.dot(x_1)
        assert_finite(tmp_1, groups=self.groups_1)
        return np.concatenate([tmp_0, tmp_1])


class BlockGS:
    def __init__(self, bmat, solve_A, solve_B, groups_0=None, groups_1=None):
        self.bmat = bmat
        self.solve_A = solve_A
        self.solve_B = solve_B
        self.A10 = bmat[groups_1, groups_0].mat
        self.A01 = bmat[groups_0, groups_1].mat
        self.sep = solve_A.shape[0]
        self.shape = bmat.shape
        self.groups_0 = groups_0
        self.groups_1 = groups_1

    def dot(self, x):
        x_0, x_1 = x[: self.sep], x[self.sep :]
        # tmp_0 = self.solve_A.dot(x_0)
        # tmp_1 = self.solve_B.dot(x_1 - self.A10.dot(tmp_0))
        tmp_1 = self.solve_B.dot(x_1)
        tmp_0 = self.solve_A.dot(x_0 - self.A01.dot(tmp_1))
        return np.concatenate([tmp_0, tmp_1])


def cond(mat):
    try:
        mat = mat.todense()
    except AttributeError:
        pass
    return np.linalg.cond(mat)


def eigs(mat):
    try:
        mat = mat.toarray()
    except AttributeError:
        pass
    return np.linalg.eigvals(mat)


def inv(mat):
    return scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(mat))


def pinv(mat):
    return scipy.sparse.csr_matrix(np.linalg.pinv(mat.toarray()))


def csr_zeros(n, m=None) -> scipy.sparse.csr_matrix:
    if m is None:
        m = n
    return scipy.sparse.csr_matrix((n, m))


def csr_ones(n) -> scipy.sparse.csr_matrix:
    return scipy.sparse.eye(n, format="csr")


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


class PetscPC:
    def __init__(self, mat=None, block_size=1, null_space: np.ndarray = None) -> None:
        self.pc = PETSc.PC().create()

        self.petsc_mat = PETSc.Mat()
        self.petsc_x = PETSc.Vec()
        self.petsc_b = PETSc.Vec()
        self.pc.setFromOptions()

        self.null_space_vectors = []
        if null_space is not None:
            for b in null_space:
                null_space_vec_petsc = PETSc.Vec().create()
                null_space_vec_petsc.setSizes(b.shape[0], block_size)
                null_space_vec_petsc.setUp()
                null_space_vec_petsc.setArray(b)
                self.null_space_vectors.append(null_space_vec_petsc)
            self.null_space_petsc = PETSc.NullSpace().create(
                True, self.null_space_vectors
            )
        else:
            self.null_space_petsc = None

        self.block_size = block_size

        self.shape: tuple[int, int]
        if mat is not None:
            self.set_operator(mat)

    def set_operator(self, mat):
        self.shape = mat.shape
        self.petsc_mat.destroy()
        self.petsc_x.destroy()
        self.petsc_b.destroy()
        self.petsc_mat.createAIJ(
            size=mat.shape,
            csr=(mat.indptr, mat.indices, mat.data),
            bsize=self.block_size,
        )
        if self.null_space_petsc is not None:
            self.petsc_mat.setNearNullSpace(self.null_space_petsc)
        self.petsc_b = self.petsc_mat.createVecLeft()
        self.petsc_x = self.petsc_mat.createVecLeft()
        self.pc.setOperators(self.petsc_mat)
        self.pc.setUp()

    def __del__(self):
        self.pc.destroy()
        self.petsc_mat.destroy()
        self.petsc_b.destroy()
        self.petsc_x.destroy()
        for vec in self.null_space_vectors:
            vec.destroy()
        if self.null_space_petsc is not None:
            self.null_space_petsc.destroy()

    def dot(self, b: np.ndarray) -> np.ndarray:
        self.petsc_x.set(0.0)
        self.petsc_b.setArray(b)
        self.pc.apply(self.petsc_b, self.petsc_x)
        res = self.petsc_x.getArray()
        return res

    def get_matrix(self):
        indptr, indices, data = self.petsc_mat.getValuesCSR()
        return scipy.sparse.csr_matrix((data, indices, indptr))


class PetscAMGMechanics(PetscPC):
    def __init__(self, dim: int, mat=None, null_space: np.ndarray = None) -> None:
        options = PETSc.Options()

        for key in options.getAll():
            options.delValue(key)

        options["pc_type"] = "gamg"
        options["mg_levels_ksp_type"] = "richardson"
        options["mg_levels_ksp_max_it"] = 1
        options["mg_levels_pc_type"] = "ilu"
        if dim == 3:
            options["mg_levels_pc_factor_levels"] = 0
        else:
            options["mg_levels_pc_factor_levels"] = 5

        # options["pc_type"] = "gamg"
        # options["pc_gamg_type"] = "agg"
        # options["pc_gamg_threshold"] = "0.03"
        # options["pc_gamg_square_graph"] = "1"
        # options["pc_gamg_sym_graph"] = None
        # options["mg_levels_ksp_type"] = "richardson"
        # options["mg_levels_pc_type"] = "sor"
        # options['pc_gamg_agg_nsmooths'] = 2

        super().__init__(mat=mat, block_size=dim, null_space=null_space)

        for key in options.getAll():
            options.delValue(key)


class PetscAMGFlow(PetscPC):
    def __init__(self, mat=None) -> None:
        options = PETSc.Options()

        options["pc_type"] = "hypre"
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_max_iter"] = 1
        # options["pc_hypre_boomeramg_cycle_type"] = "W"
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


class PetscKrylovSolver:

    def __init__(
        self,
        mat,
        pc: PETSc.PC | None = None,
        tol=1e-10,
        atol=1e-10,
    ) -> None:
        options = PETSc.Options()
        options.setValue("ksp_divtol", 1e10)
        options.setValue("ksp_atol", atol)
        options.setValue("ksp_rtol", tol)
        if pc is None:
            PETSc.Options().setValue("pc_type", "none")

        self.shape = mat.shape
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()

        self.pc = PETSc.PC()
        if pc is not None:
            self.pc.createPython(PetscPythonPC(pc))
            self.ksp.setPC(self.pc)

        self.ksp.setComputeEigenvalues(True)
        self.ksp.setConvergenceHistory()

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


class PetscGMRES(PetscKrylovSolver):

    def __init__(
        self,
        mat,
        pc: PETSc.PC | None = None,
        tol=1e-10,
        pc_side: Literal["left", "right"] = "right",
    ) -> None:
        restart = 20

        options = PETSc.Options()
        options.setValue("ksp_type", "gmres")
        # options.setValue("ksp_type", "bcgs")
        # options.setValue("ksp_type", "richardson")

        # options.setValue('ksp_gmres_modifiedgramschmidt', None)
        # options.setValue('ksp_gmres_cgs_refinement_type', 'refine_always')

        options.setValue("ksp_max_it", 3 * restart)
        options.setValue("ksp_gmres_restart", restart)

        if pc_side == "left":
            options.setValue("ksp_pc_side", "left")
            options.setValue("ksp_norm_type", "preconditioned")
        elif pc_side == "right":
            options.setValue("ksp_pc_side", "right")
            options.setValue("ksp_norm_type", "unpreconditioned")
        else:
            raise ValueError(pc_side)

        super().__init__(mat, pc, tol, atol=1e-15)


class PetscRichardson(PetscKrylovSolver):

    def __init__(
        self,
        mat,
        pc: PETSc.PC | None = None,
        tol=1e-10,
        atol=1e-10,
        pc_side: Literal["left"] = "left",
    ) -> None:
        assert pc_side == "left"

        options = PETSc.Options()
        options.setValue("ksp_type", "richardson")
        # options.setValue('ksp_type', 'gmres')
        options.setValue("ksp_max_it", 150)

        if pc_side == "left":
            options.setValue("ksp_pc_side", "left")
            options.setValue("ksp_norm_type", "preconditioned")
        else:
            raise ValueError(pc_side)

        # Absolute tolerances are different for Richardson and GMRES because the latter
        # checks the unpreconditioned residual.
        super().__init__(mat, pc, tol, atol=atol)


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


def inv_block_diag(mat, nd: int, lump: bool = False):
    if lump:
        mat = lump_nd(mat, nd)
    if nd == 1:
        return extract_diag_inv(mat)
    if nd == 2:
        return inv_block_diag_2x2(mat)
    if nd == 3:
        return inv_block_diag_3x3(mat)
    print(f"Using inefficient invert block diag, {nd = }")
    return inv(diag_nd(mat, nd=nd))


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


def pinv_left(A):
    return inv(A.T @ A) @ A.T


def pinv_right(A):
    return A.T @ inv(A @ A.T)


@njit
def inv_list_of_matrices(mats):
    results = np.zeros_like(mats)
    for i, mat in enumerate(mats):
        results[i] = np.linalg.inv(mat)
    return results


def inv_block_diag_3x3(mat):
    assert (mat.shape[0] % 3) == 0
    diag = mat.diagonal()
    a00 = diag[0::3]
    a11 = diag[1::3]
    a22 = diag[2::3]

    diag_m1 = mat.diagonal(k=-1)
    a10 = diag_m1[0::3]
    a21 = diag_m1[1::3]

    diag_m2 = mat.diagonal(k=-2)
    a20 = diag_m2[0::3]

    diag_p1 = mat.diagonal(k=1)
    a01 = diag_p1[0::3]
    a12 = diag_p1[1::3]

    diag_p2 = mat.diagonal(k=2)
    a02 = diag_p2[0::3]

    mats_3x3 = np.array(
        [
            [a00, a01, a02],
            [a10, a11, a12],
            [a20, a21, a22],
        ]
    ).transpose(2, 0, 1)
    mats_3x3_inv = inv_list_of_matrices(mats_3x3)
    return scipy.sparse.block_diag(mats_3x3_inv, format=mat.format)
