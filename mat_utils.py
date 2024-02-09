import numpy as np
import scipy.linalg
import porepy as pp
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import gmres, LinearOperator, inv
from numpy.linalg import norm
from typing import Literal
import time
from dataclasses import dataclass
import petsc4py

petsc4py.init()

from petsc4py import PETSc


def trim_label(label: str) -> str:
    trim = 15
    if len(label) <= trim:
        return label
    return label[:trim] + "..."


def spy(mat, show=True):
    marker = "+"
    if max(*mat.shape) > 300:
        marker = ","
    plt.spy(mat, marker=marker, markersize=4, color="black")
    if show:
        plt.show()


def plot_jacobian(model, equations=None):
    if equations is None:
        equations = list(model.equation_system.equations.values())
    try:
        equations[0]
    except IndexError:
        equations = list(equations)

    ax = plt.gca()

    eq_labels = []
    eq_labels_pos = []
    y_offset = 0
    jac_list = []
    for i, eq in enumerate(equations):
        jac = eq.value_and_jacobian(model.equation_system).jac
        jac_list.append([jac])
        eq_labels.append(trim_label(eq.name))
        eq_labels_pos.append(y_offset + jac.shape[0] / 2)
        plt.axhspan(
            y_offset - 0.5, y_offset + jac.shape[0] - 0.5, facecolor=f"C{i}", alpha=0.3
        )
        y_offset += jac.shape[0]

    jac = bmat(jac_list)
    spy(jac, show=False)
    if len(eq_labels) == 1:
        ax.set_title(eq_labels[0])
    else:
        ax.yaxis.set_ticks(eq_labels_pos)
        ax.set_yticklabels(eq_labels, rotation=0)

    labels = []
    labels_pos = []
    for i, var in enumerate(model.equation_system.variables):
        dofs = model.equation_system.dofs_of([var])
        plt.axvspan(dofs[0] - 0.5, dofs[-1] + 0.5, facecolor=f"C{i}", alpha=0.3)
        labels_pos.append(np.average(dofs))
        labels.append(trim_label(var.name))

    ax.xaxis.set_ticks(labels_pos)
    ax.set_xticklabels(labels, rotation=45, ha="left")


def plot_mat(mat, log=True):
    mat = mat.A
    if log:
        mat = np.log10(abs(mat))
    plt.matshow(mat)
    plt.colorbar()


def plot_eigs(mat, label="", logx=False):
    eigs, _ = scipy.linalg.eig(mat.A)
    if logx:
        eigs.real = abs(eigs.real)
    plt.scatter(eigs.real, eigs.imag, label=label, marker="$\lambda$", alpha=0.5)
    plt.xlabel(r"Re($\lambda)$")
    plt.ylabel(r"Im($\lambda$)")
    plt.legend()
    plt.grid(True)
    if logx:
        plt.xscale("log")


def solve(
    mat,
    prec=None,
    label="",
    callback_type: Literal["pr_norm", "x"] = "pr_norm",
    tol=1e-10,
):
    residuals = []
    rhs = np.ones(mat.shape[0])

    def callback(x):
        if callback_type == "pr_norm":
            residuals.append(float(x))
        else:
            residuals.append(norm(mat.dot(x) - rhs))

    if prec is not None:
        prec = LinearOperator(shape=prec.shape, matvec=prec.dot)

    restart = 50
    t0 = time.time()
    res, info = gmres(
        mat,
        rhs,
        M=prec,
        tol=tol,
        restart=restart,
        callback=callback,
        callback_type=callback_type,
        maxiter=20,
    )
    print("Solve", label, "took:", round(time.time() - t0, 2))

    inner = np.arange(len(residuals))
    if callback_type == "x":
        inner *= restart

    linestyle = "-"
    if info != 0:
        linestyle = "--"

    plt.plot(inner, residuals, label=label, marker=".", linestyle=linestyle)
    plt.yscale("log")
    plt.ylabel("pr. residual")
    plt.xlabel("gmres iter.")
    plt.grid(True)


class OmegaInv:
    def __init__(self, solve_momentum, solve_mass, C1, C2):
        self.D_inv = solve_momentum
        self.S_A_inv = solve_mass
        self.B = C1
        self.C = C2
        self.sep = solve_momentum.shape[0]
        shape = solve_momentum.shape[0] + solve_mass.shape[0]
        self.shape = shape, shape

    def dot(self, x):
        x_D, x_A = x[: self.sep], x[self.sep :]
        tmp_D = self.D_inv.dot(x_D)
        tmp_A = x_A - self.B.dot(tmp_D)
        y_A = self.S_A_inv.dot(tmp_A)
        y = np.zeros_like(x)
        y[self.sep :] = y_A
        y[: self.sep] = self.D_inv.dot(x_D - self.C.dot(y_A))
        return y


def cond(mat):
    return np.linalg.cond(mat.A)


def slice_matrix(mat, row_dofs, col_dofs, row_id, col_id):
    rows = row_dofs[row_id]
    cols = col_dofs[col_id]
    rows, cols = np.meshgrid(rows, cols, sparse=True, copy=True, indexing="ij")
    return mat[rows, cols]


def concatenate_blocks(block_matrix, rows, cols):
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
    dim = 2

    l_phys = alpha_biot**2 / (2 * mu_lame / dim + lambda_lame)
    l_min = alpha_biot**2 / (4 * mu_lame + 2 * lambda_lame)

    val = l_min * (l_phys / l_min) ** l_factor

    diagonal_approx = val
    subdomains = model.mdg.subdomains(dim=2)
    cell_volumes = subdomains[0].cell_volumes
    diagonal_approx *= cell_volumes

    density = model.fluid_density(subdomains).value(model.equation_system)
    diagonal_approx *= density

    dt = model.time_manager.dt
    diagonal_approx /= dt

    return scipy.sparse.diags(diagonal_approx)


class UpperBlockPreconditioner:

    def __init__(self, F_inv, Omega_inv, Phi):
        self.K_inv = F_inv
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
        y[: self.sep] = self.K_inv.dot(tmp)
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
    def __init__(self, mat=None) -> None:
        self.pc = PETSc.PC().create()

        self.pc.setFromOptions()
        self.petsc_mat = PETSc.Mat()
        self.petsc_x = PETSc.Vec()
        self.petsc_b = PETSc.Vec()

        self.shape: tuple[int, int]
        if mat is not None:
            self.set_operator(mat)

    def set_operator(self, mat):
        self.shape = mat.shape
        self.petsc_mat.destroy()
        self.petsc_x.destroy()
        self.petsc_b.destroy()
        self.petsc_mat.createAIJ(
            size=mat.shape, csr=(mat.indptr, mat.indices, mat.data)
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


class PetscAMG(PetscPC):
    def __init__(self, mat=None) -> None:
        options = PETSc.Options()
        # options["pc_type"] = "gamg"
        # options['pc_gamg_agg_nsmooths'] = 1
        # options["mg_levels_ksp_type"] = "chebyshev"
        # options["mg_levels_pc_type"] = "jacobi"
        # options["mg_levels_ksp_chebyshev_esteig_steps"] = 10

        options["pc_type"] = "hypre"
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_max_iter"] = 1
        options["pc_hypre_boomeramg_cycle_type"] = "V"
        # options.setValue('pc_hypre_boomeramg_relax_type_all', 'Chebyshev')
        # options.setValue('pc_hypre_boomeramg_smooth_type', 'Pilut')
        super().__init__(mat=mat)


@dataclass
class SolverStats:

    time_invert_F: float = -1
    time_prepare_mass: float = -1
    time_prepare_momentum: float = -1
    time_prepare_solver: float = -1
    gmres_iters: int = -1
    time_solve_linear_system: float = -1


class MyAwesomeSolver:

    _solver_initialized = False

    def sticking_sliding_open(self):
        subdomains = self.mdg.subdomains(dim=1)
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # The complimentarity condition
        b = pp.ad.Scalar(-1.0) * t_n - self.contact_mechanics_numerical_constant(
            subdomains
        ) * (u_n - self.fracture_gap(subdomains))
        b = b.value(self.equation_system)
        open_cells = b <= 0

        nd_vec_to_tangential = self.tangential_component(subdomains)
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )

        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)
        c_num = pp.ad.sum_operator_list(
            [e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis]
        )
        tangential_sum = t_t + c_num @ u_t_increment
        norm_tangential_sum = f_norm(tangential_sum)
        norm = norm_tangential_sum.value(self.equation_system)
        sticking_cells = (b > norm) & np.logical_not(open_cells)

        sliding_cells = True ^ (sticking_cells | open_cells)
        return sticking_cells, sliding_cells, open_cells

    def _initialize_solver(self):
        self.eq_dofs, self.var_dofs = make_row_col_dofs(self)
        self._variables_indices = self.make_variables_indices()
        self._equations_indices = self.make_equations_indices()
        eq_idx = self._equations_indices
        self.permutation = make_permutations(
            self.eq_dofs, order=eq_idx[2] + eq_idx[1] + eq_idx[0]
        )
        self.solver_stats = []
        self._solver_initialized = True

    def make_variables_indices(self):
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_lower = [
            k for i in reversed(range(0, dim_max)) for k in self.mdg.subdomains(dim=i)
        ]
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        intf = self.mdg.interfaces()
        intf_frac = self.mdg.interfaces(dim=dim_max - 1)

        return get_variables_indices(
            variable_to_idx=make_variable_to_idx(self),
            md_variables_groups=[
                [
                    self.pressure(sd_ambient),
                ],
                [
                    self.displacement(sd_ambient),
                ],
                [
                    self.pressure(sd_lower),
                    self.interface_darcy_flux(intf),
                    self.contact_traction(sd_frac),
                    self.interface_displacement(intf_frac),
                ],
            ],
        )

    def make_equations_indices(self):
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_lower = [
            k for i in reversed(range(0, dim_max)) for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()
        return get_equations_indices(
            equation_to_idx=make_equation_to_idx(self),
            equations_group_order=[
                [("mass_balance_equation", sd_ambient)],
                [("momentum_balance_equation", sd_ambient)],
                [
                    ("mass_balance_equation", sd_lower),
                    ("interface_darcy_flux_equation", intf),
                    ("normal_fracture_deformation_equation", sd_lower),
                    ("tangential_fracture_deformation_equation", sd_lower),
                    ("interface_force_balance_equation", intf),
                ],
            ],
        )

    def invert_F(self, F):
        return inv(F)

    def prepare_solve_mass(self, S_A):
        return PetscAMG(S_A)

    def prepare_solve_momentum(self, D):
        return PetscAMG(D)

    def _prepare_solver(self):
        with TimerContext() as t_prepare_solver:
            if not self._solver_initialized:
                self._initialize_solver()
            self._stats = SolverStats()
            mat, _ = self.linear_system
            block_matrix = _make_block_mat(
                mat, row_dofs=self.eq_dofs, col_dofs=self.var_dofs
            )

            eq_blocks = self._equations_indices
            var_blocks = self._variables_indices

            A = concatenate_blocks(block_matrix, eq_blocks[0], var_blocks[0])
            C1 = concatenate_blocks(block_matrix, eq_blocks[0], var_blocks[1])
            C2 = concatenate_blocks(block_matrix, eq_blocks[1], var_blocks[0])
            B = concatenate_blocks(block_matrix, eq_blocks[1], var_blocks[1])
            D1 = concatenate_blocks(block_matrix, eq_blocks[0], var_blocks[2])
            E1 = concatenate_blocks(block_matrix, eq_blocks[1], var_blocks[2])
            D2 = concatenate_blocks(block_matrix, eq_blocks[2], var_blocks[0])
            E2 = concatenate_blocks(block_matrix, eq_blocks[2], var_blocks[1])
            F = concatenate_blocks(block_matrix, eq_blocks[2], var_blocks[2])
            Phi = bmat([[E2, D2]])

            with TimerContext() as t:
                F_inv = self.invert_F(F)
            self._stats.time_invert_F = t.elapsed_time

            D1_Finv_D2 = D1 @ F_inv @ D2
            E1_Finv_D2 = E1 @ F_inv @ D2
            D1_Finv_E2 = D1 @ F_inv @ E2
            E1_Finv_E2 = E1 @ F_inv @ E2

            Ap = A - D1_Finv_D2
            Bp = B - E1_Finv_E2
            C1p = C1 - D1_Finv_E2
            C2p = C2 - E1_Finv_D2

            with TimerContext() as t:
                Bp_inv = self.prepare_solve_momentum(Bp)
            self._stats.time_prepare_momentum = t.elapsed_time

            S_Ap_fs = Ap + get_fixed_stress_stabilization(self)

            with TimerContext() as t:
                S_Ap_fs_inv = self.prepare_solve_mass(S_Ap_fs)
            self._stats.time_prepare_mass = t.elapsed_time

            Omega_p_inv_fstress = OmegaInv(
                solve_momentum=Bp_inv, solve_mass=S_Ap_fs_inv, C1=C1p, C2=C2p
            )

            preconditioner = UpperBlockPreconditioner(
                F_inv=F_inv, Omega_inv=Omega_p_inv_fstress, Phi=Phi
            )
            reordered_mat = concatenate_blocks(
                block_matrix,
                eq_blocks[2] + eq_blocks[1] + eq_blocks[0],
                var_blocks[2] + var_blocks[1] + var_blocks[0],
            )

            permuted_mat = self.permutation @ mat @ self.permutation.T
            assert (reordered_mat - permuted_mat).data.size == 0

        self._stats.time_prepare_solver = t_prepare_solver.elapsed_time
        return reordered_mat, preconditioner

    def solve_linear_system(self) -> np.ndarray:
        if not self.params.get('iterative_solver', True):
            return super().solve_linear_system()
        with TimerContext() as t_solve:
            mat, rhs = self.linear_system
            mat_permuted, prec = self._prepare_solver()

            rhs_permuted = self.permutation @ rhs

            residuals = []

            def callback(x):
                residuals.append(float(x))

            restart = 50
            res_permuted, info = gmres(
                mat_permuted,
                rhs_permuted,
                M=LinearOperator(shape=prec.shape, matvec=prec.dot),
                tol=1e-10,
                restart=restart,
                callback=callback,
                callback_type="pr_norm",
                maxiter=20,
            )

            if info < 0:
                print(f"GMRES failed, {info=}")
            if info > 0:
                print(f"GMRES did not converge, {info=}")

            res = self.permutation.T.dot(res_permuted)

        self._stats.time_solve_linear_system = t_solve.elapsed_time
        self._stats.gmres_iters = len(residuals)
        self.solver_stats.append(self._stats)
        return np.atleast_1d(res)


class PetscILU(PetscPC):
    def __init__(self, mat=None) -> None:
        options = PETSc.Options()
        options.setValue("pc_type", "ilu")
        options.setValue("pc_factor_levels", 0)
        options.setValue("pc_factor_diagonal_fill", None)  # Doesn't affect
        options.setValue("pc_factor_mat_ordering_type", "rcm")
        options.setValue("pc_factor_nonzeros_along_diagonal", None)
        super().__init__(mat=mat)


def color_spy(block_mat, row_idx=None, col_idx=None, row_names=None, col_names=None):
    if row_idx is None:
        row_idx = list(range(block_mat.shape[0]))
    if col_idx is None:
        col_idx = list(range(block_mat.shape[1]))
    if row_names is None:
        row_names = row_idx
    if col_names is None:
        col_names = col_idx
    row_sep = [0]
    col_sep = [0]
    active_submatrices = []
    for i in row_idx:
        active_row = []
        for j in col_idx:
            submat = block_mat[i, j]
            active_row.append(submat)
            if i == row_idx[0]:
                col_sep.append(col_sep[-1] + submat.shape[1])
        row_sep.append(row_sep[-1] + submat.shape[0])
        active_submatrices.append(active_row)
    spy(bmat(active_submatrices), show=False)

    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_idx)):
        ystart, yend = row_sep[i : i + 2]
        row_label_pos.append(ystart + (yend - ystart) / 2)
        plt.axhspan(ystart - 0.5, yend - 0.5, facecolor=f"C{i}", alpha=0.3)
    ax.yaxis.set_ticks(row_label_pos)
    ax.set_yticklabels(row_names, rotation=0)

    col_label_pos = []
    for i in range(len(col_idx)):
        xstart, xend = col_sep[i : i + 2]
        col_label_pos.append(xstart + (xend - xstart) / 2)
        plt.axvspan(xstart - 0.5, xend - 0.5, facecolor=f"C{i}", alpha=0.3)
    ax.xaxis.set_ticks(col_label_pos)
    ax.set_xticklabels(col_names, rotation=0)


from petsc4py.PETSc import KSP
from petsc4py.PETSc import PC
from petsc4py.PETSc import Mat
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


class PetscPythonPC:

    def __init__(self, pc):
        self.pc = pc

    def apply(self, pc: PC, b: Vec, x: Vec) -> None:
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
        # options.setValue('ksp_norm_type', 'unpreconditioned')
        options.setValue("ksp_gmres_restart", 50)
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


import itertools

MARKERS = itertools.cycle(
    [
        "x",
        "+",
        # "o",
        # "v",
        # "<",
        # ">",
        # "^",
        "1",
        "2",
        "3",
        "4",
    ]
)


def solve_petsc(
    mat,
    prec=None,
    label="",
    logx_eigs=False,
):
    rhs = np.ones(mat.shape[0])
    gmres = PetscGMRES(mat, pc=prec)

    t0 = time.time()
    _ = gmres.solve(rhs)
    print("Solve", label, "took:", round(time.time() - t0, 2))
    residuals = gmres.get_residuals()
    info = gmres.ksp.getConvergedReason()
    linestyle = "-"
    if info <= 0:
        linestyle = "--"

    plt.gcf().set_size_inches(14, 4)

    ax = plt.subplot(1, 2, 1)
    ax.plot(residuals / residuals[0], label=label, marker=".", linestyle=linestyle)
    ax.set_yscale("log")
    ax.set_ylabel("pr. residual")
    ax.set_xlabel("gmres iter.")
    ax.grid(True)
    if label != "":
        ax.legend()

    eigs = gmres.ksp.computeEigenvalues()
    ax = plt.subplot(1, 2, 2)
    if logx_eigs:
        eigs.real = abs(eigs.real)
    # ax.scatter(eigs.real, eigs.imag, label=label, marker="$\lambda$", alpha=0.9)
    ax.scatter(eigs.real, eigs.imag, label=label, alpha=1, s=300, marker=next(MARKERS))
    ax.set_xlabel(r"Re($\lambda)$")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.grid(True)
    if label != '':
        ax.legend()
    if logx_eigs:
        plt.xscale("log")


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
