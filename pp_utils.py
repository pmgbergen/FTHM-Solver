import sys
from dataclasses import asdict, dataclass, field
from functools import cached_property, partial
from pathlib import Path
import time
import json
import numpy as np
import porepy as pp
import scipy.sparse
from scipy.sparse import bmat
from porepy.applications.md_grids import fracture_sets
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.examples.flow_benchmark_2d_case_3 import Permeability
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
from porepy.models.momentum_balance import BoundaryConditionsMomentumBalance
from porepy.models.poromechanics import Poromechanics
from porepy.viz.diagnostics_mixin import DiagnosticsMixin
from scipy.sparse.linalg import inv

from mat_utils import (
    OmegaInv,
    PetscAMGFlow,
    PetscAMGMechanics,
    PetscGMRES,
    TimerContext,
    UpperBlockPreconditioner,
    _make_block_mat,
    concatenate_blocks,
    get_equations_indices,
    get_fixed_stress_stabilization,
    get_variables_indices,
    make_equation_to_idx,
    make_permutations,
    make_row_col_dofs,
    make_variable_to_idx,
    cond,
    extract_diag_inv,
)


class BCMechanics(BoundaryConditionsMomentumBalance):
    def bc_type_mechanics(self, sd):
        boundary_faces = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, boundary_faces.south + boundary_faces.west + boundary_faces.east, "dir"
        )
        bc.internal_to_dirichlet(sd)
        return bc


class BCMechanicsSticking(BCMechanics):

    bc_name = "sticking"

    def bc_values_stress(self, boundary_grid):
        stress = np.zeros((self.nd, boundary_grid.num_cells))
        boundary_faces = self.domain_boundary_sides(boundary_grid)
        stress[1, boundary_faces.north] = -self.solid.convert_units(
            1000, "kg * m^-1 * s^-2"
        )
        return stress.ravel("F")


class BCMechanicsOpen(BCMechanics):

    bc_name = "open"

    def bc_values_stress(self, boundary_grid):
        stress = np.zeros((self.nd, boundary_grid.num_cells))
        boundary_faces = self.domain_boundary_sides(boundary_grid)
        stress[1, boundary_faces.north] = self.solid.convert_units(
            1000, "kg * m^-1 * s^-2"
        )
        return stress.ravel("F")


class BCMechanicsSliding(BCMechanics):

    bc_name = "sliding"

    def bc_values_stress(self, boundary_grid):
        stress = np.zeros((self.nd, boundary_grid.num_cells))
        boundary_faces = self.domain_boundary_sides(boundary_grid)
        stress[0, boundary_faces.north] = self.solid.convert_units(
            1000, "kg * m^-1 * s^-2"
        )
        return stress.ravel("F")


class BCFlow(BoundaryConditionsSinglePhaseFlow):
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.north + bounds.south, "dir")
        return bc

    # def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
    #     bounds = self.domain_boundary_sides(boundary_grid)
    #     values = np.zeros(boundary_grid.num_cells)
    #     values[bounds.north] = self.fluid.convert_units(4e4, "Pa")
    #     # values[bounds.south] = self.fluid.convert_units(0.01, "Pa")
    #     return values
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[bounds.north] = self.fluid.convert_units(2e5, "Pa")
        return values


class TimeStepping:
    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        self.time_manager.compute_time_step(iteration_counter, recompute_solution=False)

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.time_manager.compute_time_step(recompute_solution=True)


@dataclass
class LinearSolveStats:
    time_invert_F: float = -1
    time_prepare_mass: float = -1
    time_prepare_momentum: float = -1
    time_prepare_solver: float = -1
    time_gmres: float = -1
    gmres_iters: int = -1
    time_solve_linear_system: float = -1
    matrix_id: str = ""
    rhs_id: str = ""
    petsc_converged_reason: int = -100
    num_sticking_sliding_open: tuple[int, int, int] = (-1, -1, -1)


@dataclass
class TimeStepStats:
    linear_solves: list[LinearSolveStats] = field(default_factory=list)

    @classmethod
    def from_json(cls, json: str):
        data = cls(**json)
        tmp = []
        for x in data.linear_solves:
            payload = {
                k: v for k, v in x.items() if k in LinearSolveStats.__dataclass_fields__
            }
            tmp.append(LinearSolveStats(**payload))
        data.linear_solves = tmp
        return data


def dump_json(name, data):
    save_path = Path("./stats")
    save_path.mkdir(exist_ok=True)
    dict_data = [asdict(x) for x in data]
    json_data = json.dumps(dict_data)
    with open(save_path / name, "w") as file:
        file.write(json_data)


def make_right_scaling(model: pp.SolutionStrategy, scales=(1e9, 1e-9)):
    eq_dofs, var_dofs = make_row_col_dofs(model)

    subdomains = model.mdg.subdomains()
    intf = model.mdg.interfaces()

    var_idx = get_variables_indices(
        variable_to_idx=make_variable_to_idx(model),
        md_variables_groups=[
            [
                model.pressure(subdomains),
            ],
            [
                model.interface_darcy_flux(intf),
            ],
        ],
    )

    diag = np.ones(model.equation_system.num_dofs())

    if model.params.get("rprec", False):
        for i in range(len(var_idx)):
            for idx in var_idx[i]:
                diag[var_dofs[idx]] = scales[i]

    rprec = scipy.sparse.diags(diag).tocsr()
    rprec_inv = scipy.sparse.diags(1 / diag).tocsr()
    return rprec, rprec_inv


@dataclass
class SlicedMatrix:
    #  A  C1 E1
    #  C1 B  E2
    #  D1 E1 F
    A: scipy.sparse.csr_matrix  # mass nd
    B: scipy.sparse.csr_matrix  # momentum nd
    F: scipy.sparse.csr_matrix  # everything else
    C1: scipy.sparse.csr_matrix
    C2: scipy.sparse.csr_matrix
    D1: scipy.sparse.csr_matrix
    D2: scipy.sparse.csr_matrix
    E1: scipy.sparse.csr_matrix
    E2: scipy.sparse.csr_matrix


@dataclass
class SlicedOmega:
    Ap: scipy.sparse.csr_matrix
    Bp: scipy.sparse.csr_matrix
    C1p: scipy.sparse.csr_matrix
    C2p: scipy.sparse.csr_matrix
    F_inv: scipy.sparse.csr_matrix


class MyPetscSolver(pp.SolutionStrategy):

    _solver_initialized = False
    _linear_solve_stats: LinearSolveStats
    _time_step_stats: TimeStepStats

    @cached_property
    def statistics(self) -> list[TimeStepStats]:
        return []

    @property
    def simulation_name(self) -> str:
        sim_name = self.params.get("simulation_name", "simulation")
        sim_name = f"{sim_name}_{self.bc_name}"
        if self.params.get("rprec", False):
            sim_name = f"{sim_name}_rprec"
        return sim_name

    def before_nonlinear_loop(self) -> None:
        self._time_step_stats = TimeStepStats()
        return super().before_nonlinear_loop()

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.statistics.append(self._time_step_stats)
        dump_json(self.simulation_name + ".json", self.statistics)
        super().after_nonlinear_convergence(solution, errors, iteration_counter)

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.statistics.append(self._time_step_stats)
        dump_json(self.simulation_name + ".json", self.statistics)
        super().after_nonlinear_failure(solution, errors, iteration_counter)

    def before_nonlinear_iteration(self) -> None:
        self._linear_solve_stats = LinearSolveStats()
        super().before_nonlinear_iteration()

        data = self.sticking_sliding_open()
        self._linear_solve_stats.num_sticking_sliding_open = tuple(
            int(sum(x)) for x in data
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        save_path = Path("./matrices")
        save_path.mkdir(exist_ok=True)
        mat, rhs = self.linear_system
        name = f"{self.simulation_name}_{int(time.time() * 1000)}"
        mat_id = f"{name}.npz"
        rhs_id = f"{name}_rhs.npy"
        scipy.sparse.save_npz(save_path / mat_id, mat)
        np.save(save_path / rhs_id, rhs)
        self._linear_solve_stats.matrix_id = mat_id
        self._linear_solve_stats.rhs_id = rhs_id

        self._time_step_stats.linear_solves.append(self._linear_solve_stats)
        super().after_nonlinear_iteration(solution_vector)

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params,
    ) -> tuple[float, bool, bool]:
        return super().check_convergence(
            solution=solution,
            prev_solution=prev_solution,
            init_solution=init_solution,
            nl_params=nl_params,
        )

    def sticking_sliding_open(self):
        frac_dim = self.nd - 1
        subdomains = self.mdg.subdomains(dim=frac_dim)
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
            subdomains, dim=frac_dim  # type: ignore[call-arg]
        )

        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, frac_dim), "norm_function")
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
        self._rprec, self._rprec_inv = make_right_scaling(self)
        eq_idx = self._equations_indices
        self.permutation = make_permutations(
            self.eq_dofs, order=eq_idx[2] + eq_idx[1] + eq_idx[0]
        )
        self._solver_initialized = True

    def slice_jacobian(self, mat) -> SlicedMatrix:
        if not self._solver_initialized:
            self._initialize_solver()

        block_matrix = _make_block_mat(
            mat, row_dofs=self.eq_dofs, col_dofs=self.var_dofs
        )

        eq_blocks = self._equations_indices
        var_blocks = self._variables_indices

        return SlicedMatrix(
            A=concatenate_blocks(block_matrix, eq_blocks[0], var_blocks[0]),
            C1=concatenate_blocks(block_matrix, eq_blocks[0], var_blocks[1]),
            C2=concatenate_blocks(block_matrix, eq_blocks[1], var_blocks[0]),
            B=concatenate_blocks(block_matrix, eq_blocks[1], var_blocks[1]),
            D1=concatenate_blocks(block_matrix, eq_blocks[0], var_blocks[2]),
            E1=concatenate_blocks(block_matrix, eq_blocks[1], var_blocks[2]),
            D2=concatenate_blocks(block_matrix, eq_blocks[2], var_blocks[0]),
            E2=concatenate_blocks(block_matrix, eq_blocks[2], var_blocks[1]),
            F=concatenate_blocks(block_matrix, eq_blocks[2], var_blocks[2]),
        )

    def slice_omega(self, sliced_mat: SlicedMatrix) -> SlicedOmega:
        with TimerContext() as t:
            F_inv = self.invert_F(sliced_mat.F)
        self._linear_solve_stats.time_invert_F = t.elapsed_time

        D1 = sliced_mat.D1
        D2 = sliced_mat.D2
        E1 = sliced_mat.E1
        E2 = sliced_mat.E2

        D1_Finv_D2 = D1 @ F_inv @ D2
        E1_Finv_D2 = E1 @ F_inv @ D2
        D1_Finv_E2 = D1 @ F_inv @ E2
        E1_Finv_E2 = E1 @ F_inv @ E2

        return SlicedOmega(
            Ap=sliced_mat.A - D1_Finv_D2,
            Bp=sliced_mat.B - E1_Finv_E2,
            C1p=sliced_mat.C1 - D1_Finv_E2,
            C2p=sliced_mat.C2 - E1_Finv_D2,
            F_inv=F_inv,
        )

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
        # return extract_diag_inv(F)
        try:
            return inv(F)
        except RuntimeError as e:
            print(e)
            return scipy.sparse.eye(F.shape[0])

    def prepare_solve_mass(self, S_A):
        return PetscAMGFlow(S_A)

    def prepare_solve_momentum(self, B):
        return PetscAMGMechanics(mat=B, dim=self.mdg.dim_max())

    def _prepare_solver(self):
        with TimerContext() as t_prepare_solver:
            mat, _ = self.linear_system
            sliced_mat = self.slice_jacobian(mat)
            Phi = bmat([[sliced_mat.E2, sliced_mat.D2]])
            Omega = self.slice_omega(sliced_mat)

            with TimerContext() as t:
                Bp_inv = self.prepare_solve_momentum(Omega.Bp)
            self._linear_solve_stats.time_prepare_momentum = t.elapsed_time

            S_Ap_fs = Omega.Ap + get_fixed_stress_stabilization(self)

            with TimerContext() as t:
                S_Ap_fs_inv = self.prepare_solve_mass(S_Ap_fs)
            self._linear_solve_stats.time_prepare_mass = t.elapsed_time

            Omega_p_inv_fstress = OmegaInv(
                solve_momentum=Bp_inv,
                solve_mass=S_Ap_fs_inv,
                C1=Omega.C1p,
                C2=Omega.C2p,
            )

            preconditioner = UpperBlockPreconditioner(
                F_inv=Omega.F_inv, Omega_inv=Omega_p_inv_fstress, Phi=Phi
            )
            # reordered_mat = concatenate_blocks(
            #     block_matrix,
            #     eq_blocks[2] + eq_blocks[1] + eq_blocks[0],
            #     var_blocks[2] + var_blocks[1] + var_blocks[0],
            # )

            reordered_mat = self.permutation @ mat @ self.permutation.T
            # assert (reordered_mat - permuted_mat).data.size == 0

        self._linear_solve_stats.time_prepare_solver = t_prepare_solver.elapsed_time

        # for debugging
        self.Ap = Omega.Ap
        self.Bp = Omega.Bp
        self.C1p = Omega.C1p
        self.C2p = Omega.C2p
        self.F = sliced_mat.F
        self.S_Ap_fs = S_Ap_fs

        return reordered_mat, preconditioner

    def assemble_linear_system(self) -> None:
        super().assemble_linear_system()
        if not self._solver_initialized:
            self._initialize_solver()
        mat, rhs = self.linear_system
        scaled_mat = mat @ self._rprec
        self.linear_system = scaled_mat, rhs

    def solve_linear_system(self) -> np.ndarray:
        if not self.params.get("iterative_solver", True):
            return super().solve_linear_system()
        mat, rhs = self.linear_system
        if not np.all(np.isfinite(rhs)):
            self._linear_solve_stats.time_solve_linear_system = 0
            self._linear_solve_stats.gmres_iters = 0
            result = np.zeros_like(rhs)
            result[:] = np.nan
            return result

        with TimerContext() as t_solve:

            mat_permuted, prec = self._prepare_solver()

            rhs_permuted = self.permutation @ rhs

            gmres_ = PetscGMRES(mat=mat_permuted, pc=prec)

            with TimerContext() as t_gmres:
                res_permuted = gmres_.solve(rhs_permuted)
            self._linear_solve_stats.time_gmres = t_gmres.elapsed_time

            info = gmres_.ksp.getConvergedReason()

            if info <= 0:
                print(f"GMRES failed, {info=}", file=sys.stderr)
                if info == -9:
                    res_permuted[:] = np.nan

            res = self.permutation.T.dot(res_permuted)

            res = self._rprec @ res

        self._linear_solve_stats.petsc_converged_reason = info
        self._linear_solve_stats.time_solve_linear_system = t_solve.elapsed_time
        self._linear_solve_stats.gmres_iters = len(gmres_.get_residuals())
        return np.atleast_1d(res)
