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
    return scipy.sparse.linalg.inv(mat.tocsc())


def condest(mat):
    data = abs(mat.data)
    return data.max() / data.min()


def slice_matrix(mat, row_dofs, col_dofs, row_id, col_id):
    rows = row_dofs[row_id]
    cols = col_dofs[col_id]
    rows, cols = np.meshgrid(rows, cols, sparse=True, copy=True, indexing="ij")
    return mat[rows, cols]


def concatenate_blocks(block_matrix, rows=None, cols=None):
    if rows is None:
        rows = range(block_matrix.shape[0])
    if cols is None:
        cols = range(block_matrix.shape[1])
    result = []
    for i in rows:
        res_row = []
        for j in cols:
            res_row.append(block_matrix[i][j])
        result.append(res_row)
    return scipy.sparse.bmat(result)


def make_row_col_dofs(model):
    eq_info = []
    eq_dofs = []
    offset = 0
    for (
        eq_name,
        data,
    ) in model.equation_system._equation_image_space_composition.items():
        local_offset = 0
        for grid, dofs in data.items():
            eq_dofs.append(dofs + offset)
            eq_info.append((eq_name, grid))
            local_offset += len(dofs)
        offset += local_offset

    var_info = []
    var_dofs = []
    for var in model.equation_system.variables:
        var_info.append((var.name, var.domain))
        var_dofs.append(model.equation_system.dofs_of([var]))
    return eq_dofs, var_dofs


def make_block_mat(model, mat):
    eq_dofs, var_dofs = make_row_col_dofs(model)
    return _make_block_mat(mat=mat, row_dofs=eq_dofs, col_dofs=var_dofs)


def _make_block_mat(mat, row_dofs, col_dofs):
    block_matrix = []
    for i in range(len(row_dofs)):
        block_row = []
        for j in range(len(col_dofs)):
            block_row.append(slice_matrix(mat, row_dofs, col_dofs, i, j))
        block_matrix.append(block_row)

    return np.array(block_matrix)


def get_fixed_stress_stabilization(model, l_factor: float = 0.6):
    mu_lame = model.solid.shear_modulus()
    lambda_lame = model.solid.lame_lambda()
    alpha_biot = model.solid.biot_coefficient()
    dim = model.nd

    l_phys = alpha_biot**2 / (2 * mu_lame / dim + lambda_lame)
    l_min = alpha_biot**2 / (4 * mu_lame + 2 * lambda_lame)

    val = l_min * (l_phys / l_min) ** l_factor

    diagonal_approx = val
    subdomains = model.mdg.subdomains(dim=dim)
    cell_volumes = subdomains[0].cell_volumes
    diagonal_approx *= cell_volumes

    density = model.fluid_density(subdomains).value(model.equation_system)
    diagonal_approx *= density

    dt = model.time_manager.dt
    diagonal_approx /= dt

    return scipy.sparse.diags(diagonal_approx)


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
    # order = [1, 2, 4, 5, 6, 3, 0]
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
    def __init__(self, mat=None) -> None:
        options = PETSc.Options()
        options.setValue("pc_type", "ilu")
        options.setValue("pc_factor_levels", 0)
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

        self.ksp = PETSc.KSP().create()
        options = PETSc.Options()
        options.setValue("ksp_type", "gmres")
        # options.setValue("ksp_type", "bcgs")
        options.setValue("ksp_rtol", 1e-10)
        options.setValue("ksp_max_it", 20 * 50)
        options.setValue('ksp_norm_type', 'unpreconditioned')
        options.setValue("ksp_gmres_restart", 50)
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

    def get_residuals(self):
        return self.ksp.getConvergenceHistory()


class PetscJacobi(PetscPC):
    def __init__(self, mat=None) -> None:
        options = PETSc.Options()
        options["pc_type"] = "jacobi"
        super().__init__(mat=mat)


def make_variable_to_idx(model):
    return {var: i for i, var in enumerate(model.equation_system.variables)}


def get_variables_indices(variable_to_idx, md_variables_groups):
    indices = []
    for md_var_group in md_variables_groups:
        group_idx = []
        for md_var in md_var_group:
            group_idx.extend([variable_to_idx[var] for var in md_var.sub_vars])
        indices.append(group_idx)
    return indices


def make_equation_to_idx(model):
    equation_to_idx = {}
    idx = 0
    for (
        eq_name,
        domains,
    ) in model.equation_system._equation_image_space_composition.items():
        for domain in domains:
            equation_to_idx[(eq_name, domain)] = idx
            idx += 1
    return equation_to_idx


def get_equations_indices(equation_to_idx, equations_group_order):
    indices = []
    for group in equations_group_order:
        group_idx = []
        for eq_name, domains in group:
            for domain in domains:
                if (eq_name, domain) in equation_to_idx:
                    group_idx.append(equation_to_idx[(eq_name, domain)])
        indices.append(group_idx)
    return indices


def extract_diag_inv(mat):
    diag = mat.diagonal()
    # diag = np.array(mat.sum(axis=1)).squeeze()
    ones = scipy.sparse.eye(mat.shape[0], format="csr")
    diag_inv = 1 / diag
    ones.data[:] = diag_inv
    return ones


def reverse_cuthill_mckee(mat):
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    reorder = reverse_cuthill_mckee(mat)
    return mat[reorder][:, reorder]
