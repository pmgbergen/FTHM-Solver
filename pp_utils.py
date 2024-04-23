import json
import sys
import time
from dataclasses import asdict
from functools import cached_property, partial
from pathlib import Path

import numpy as np
import porepy as pp
import scipy.sparse
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
from porepy.models.momentum_balance import BoundaryConditionsMomentumBalance
from stats import LinearSolveStats, TimeStepStats
from fpm.block_matrix import BlockMatrixStorage, make_solver, SolveSchema

from invert import USE_NUMBA

from mat_utils import (
    PetscAMGFlow,
    PetscAMGMechanics,
    PetscGMRES,
    TimerContext,
)


class CheckStickingSlidingOpen:

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


def dump_json(name, data):
    save_path = Path("./stats")
    save_path.mkdir(exist_ok=True)
    dict_data = [asdict(x) for x in data]
    json_data = json.dumps(dict_data)
    with open(save_path / name, "w") as file:
        file.write(json_data)


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


def get_variables_group_ids(model, md_variables_groups):
    variable_to_idx = {var: i for i, var in enumerate(model.equation_system.variables)}
    indices = []
    for md_var_group in md_variables_groups:
        group_idx = []
        for md_var in md_var_group:
            group_idx.extend([variable_to_idx[var] for var in md_var.sub_vars])
        indices.append(group_idx)
    return indices


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


def get_fixed_stress_stabilization_nd(model, l_factor: float = 0.6):
    mat_nd = get_fixed_stress_stabilization(model=model, l_factor=l_factor)

    sd_lower = [
        sd for d in reversed(range(model.nd)) for sd in model.mdg.subdomains(dim=d)
    ]
    num_cells = sum(sd.num_cells for sd in sd_lower)

    zero_lower = scipy.sparse.csr_matrix((num_cells, num_cells))
    return scipy.sparse.block_diag([mat_nd, zero_lower]).tocsr()


def get_equations_group_ids(model, equations_group_order):
    equation_to_idx = {}
    idx = 0
    for (
        eq_name,
        domains,
    ) in model.equation_system._equation_image_space_composition.items():
        for domain in domains:
            equation_to_idx[(eq_name, domain)] = idx
            idx += 1

    indices = []
    for group in equations_group_order:
        group_idx = []
        for eq_name, domains in group:
            for domain in domains:
                if (eq_name, domain) in equation_to_idx:
                    group_idx.append(equation_to_idx[(eq_name, domain)])
        indices.append(group_idx)
    return indices


class MyPetscSolver(CheckStickingSlidingOpen, pp.SolutionStrategy):

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
        use_direct = not self.params.get("iterative_solver", True)
        if use_direct:
            sim_name = f"{sim_name}_direct"
        else:
            solver_type = self.params.get("solver_type", "baseline")
            sim_name = f"{sim_name}_solver_{solver_type}"

        if USE_NUMBA:
            sim_name = f"{sim_name}_numba"
        else:
            sim_name = f"{sim_name}_python"

        return sim_name

    def before_nonlinear_loop(self) -> None:
        if not self._solver_initialized:
            self._initialize_solver()
        self._time_step_stats = TimeStepStats()
        self.nonlinear_residual_0 = self.compute_nonlinear_residual(replace_zeros=True)
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
        # self.discretize()

        data = self.sticking_sliding_open()
        self._linear_solve_stats.num_sticking_sliding_open = tuple(
            int(sum(x)) for x in data
        )
        self._linear_solve_stats.sticking = data[0].tolist()
        self._linear_solve_stats.sliding = data[1].tolist()
        self._linear_solve_stats.open_ = data[2].tolist()

        characteristic = self._characteristic.value(self.equation_system).tolist()
        self._linear_solve_stats.transition_sticking_sliding = characteristic

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        save_path = Path("./matrices")
        save_path.mkdir(exist_ok=True)
        mat, rhs = self.linear_system
        name = f"{self.simulation_name}_{int(time.time() * 1000)}"
        mat_id = f"{name}.npz"
        rhs_id = f"{name}_rhs.npy"
        state_id = f"{name}_state.npy"
        iterate_id = f"{name}_iterate.npy"
        scipy.sparse.save_npz(save_path / mat_id, mat)
        np.save(save_path / rhs_id, rhs)
        np.save(
            save_path / state_id,
            self.equation_system.get_variable_values(time_step_index=0),
        )
        np.save(
            save_path / iterate_id,
            self.equation_system.get_variable_values(iterate_index=0),
        )
        self._linear_solve_stats.iterate_id = iterate_id
        self._linear_solve_stats.state_id = state_id
        self._linear_solve_stats.matrix_id = mat_id
        self._linear_solve_stats.rhs_id = rhs_id
        self._linear_solve_stats.simulation_dt = self.time_manager.dt
        self._time_step_stats.linear_solves.append(self._linear_solve_stats)
        super().after_nonlinear_iteration(solution_vector)

    def compute_nonlinear_residual(self, replace_zeros: bool = False):
        nonlinear_residual = self.equation_system.assemble(evaluate_jacobian=False)
        group_residuals = []
        for group in self._equation_groups:
            dofs = np.concatenate([self.eq_dofs[block] for block in group])
            nrm = np.linalg.norm(nonlinear_residual[dofs])
            group_residuals.append(nrm)

        group_residuals = np.array(group_residuals)

        atol = 1e-16
        group_residuals[group_residuals < atol] = 0

        if replace_zeros:
            group_residuals[group_residuals == 0] = 1
        return group_residuals

    # def check_convergence(
    #     self,
    #     solution: np.ndarray,
    #     prev_solution: np.ndarray,
    #     init_solution: np.ndarray,
    #     nl_params,
    # ) -> tuple[float, bool, bool]:
    # if np.any(np.isnan(solution)):
    #     # If the solution contains nan values, we have diverged.
    #     return np.nan, False, True

    # nonlinear_residual = self.compute_nonlinear_residual(replace_zeros=False)
    # # print([np.format_float_scientific(x, precision=2) for x in nonlinear_residual])
    # residual_drop = nonlinear_residual / self.nonlinear_residual_0
    # residual_drop_sum = sum(residual_drop)
    # converged = False
    # diverged = False
    # if residual_drop_sum < nl_params["nl_convergence_tol"]:
    #     converged = True
    # return residual_drop_sum, converged, diverged

    def _initialize_solver(self):
        self.eq_dofs, self.var_dofs = make_row_col_dofs(self)
        self._variable_groups = self.make_variables_groups()
        self._equation_groups = self.make_equations_groups()

        # This is very important, iterative solver relies on this reindexing
        self._reorder_contact = make_reorder_contact(self)
        self._corrected_eq_dofs, self._corrected_eq_groups = correct_eq_groups(self)
        self._solver_initialized = True

    def make_variables_groups(self):
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_lower = [
            k for i in reversed(range(0, dim_max)) for k in self.mdg.subdomains(dim=i)
        ]
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        intf = self.mdg.interfaces()
        intf_frac = self.mdg.interfaces(dim=dim_max - 1)

        return get_variables_group_ids(
            model=self,
            md_variables_groups=[
                [self.pressure(sd_ambient)],
                [self.displacement(sd_ambient)],
                [self.pressure(sd_lower)],
                [self.interface_darcy_flux(intf)],
                [self.contact_traction(sd_frac)],
                [self.interface_displacement(intf_frac)],
            ],
        )

    def make_equations_groups(self):
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_lower = [
            k for i in reversed(range(0, dim_max)) for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()

        return get_equations_group_ids(
            model=self,
            equations_group_order=[
                [("mass_balance_equation", sd_ambient)],  # 0
                [("momentum_balance_equation", sd_ambient)],  # 1
                [("mass_balance_equation", sd_lower)],  # 2
                [("interface_darcy_flux_equation", intf)],  # 3
                [
                    ("normal_fracture_deformation_equation", sd_lower),  # 4
                    ("tangential_fracture_deformation_equation", sd_lower),
                ],
                [("interface_force_balance_equation", intf)],  # 5
            ],
        )

    def assemble_linear_system(self) -> None:
        super().assemble_linear_system()
        mat, rhs = self.linear_system
        mat = mat[self._reorder_contact]
        rhs = rhs[self._reorder_contact]
        self.linear_system = mat, rhs

    def make_solver_schema(self):
        solver_type = self.params.get("solver_type", "baseline")

        if solver_type == "baseline":
            return SolveSchema(
                groups=[2, 3, 4, 5],
                complement=SolveSchema(
                    groups=[1],
                    invertor=lambda bmat: get_fixed_stress_stabilization(self),
                    invertor_type="physical",
                    solve=lambda bmat: PetscAMGMechanics(
                        dim=self.nd, mat=bmat[[1]].mat
                    ),
                    complement=SolveSchema(
                        groups=[0],
                        solve=lambda bmat: PetscAMGFlow(mat=bmat[[0]].mat),
                    ),
                ),
            )

        if solver_type == "1":
            from mat_utils import PetscILU, extract_diag_inv
            from preconditioner_mech import make_J44_inv_bdiag
            from fixed_stress import make_fs

            return SolveSchema(
                groups=[3],
                solve=lambda bmat: PetscILU(bmat[[3]].mat),
                invertor=lambda bmat: extract_diag_inv(bmat[[3]].mat),
                complement=SolveSchema(
                    groups=[4],
                    solve=lambda bmat: make_J44_inv_bdiag(self, bmat=bmat),
                    complement=SolveSchema(
                        groups=[1, 5],
                        solve=lambda bmat: PetscAMGMechanics(
                            mat=bmat[[1, 5]].mat, dim=self.nd
                        ),
                        invertor=lambda bmat: make_fs(self, bmat).mat,
                        invertor_type="physical",
                        complement=SolveSchema(
                            groups=[0, 2],
                            solve=lambda bmat: PetscAMGFlow(mat=bmat[[0, 2]].mat),
                        ),
                    ),
                ),
            )

        raise ValueError(f"{solver_type}")

    def _prepare_solver(self):
        if not self._solver_initialized:
            self._initialize_solver()
        with TimerContext() as t_prepare_solver:
            mat, rhs = self.linear_system

            bmat = BlockMatrixStorage(
                mat=mat,
                global_row_idx=self._corrected_eq_dofs,
                global_col_idx=self.var_dofs,
                groups_row=self._corrected_eq_groups,
                groups_col=self._variable_groups,
            )
            self.bmat = bmat

            schema = self.make_solver_schema()

            bmat_reordered, preconditioner = make_solver(schema=schema, mat_orig=bmat)

        self._linear_solve_stats.time_prepare_solver = t_prepare_solver.elapsed_time

        return bmat_reordered, preconditioner

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

            rhs_permuted = mat_permuted.local_rhs(rhs)

            tol = 1e-10
            pc_side = "right"
            solver_type = self.params.get("solver_type", "baseline")
            if solver_type == "1":
                tol = 1e-15
                pc_side = "left"
            gmres_ = PetscGMRES(mat=mat_permuted.mat, pc=prec, pc_side=pc_side, tol=tol)

            with TimerContext() as t_gmres:
                res_permuted = gmres_.solve(rhs_permuted)
            self._linear_solve_stats.time_gmres = t_gmres.elapsed_time

            info = gmres_.ksp.getConvergedReason()

            if info <= 0:
                print(f"GMRES failed, {info=}", file=sys.stderr)
                if info == -9:
                    res_permuted[:] = np.nan

            res = mat_permuted.reverse_transform_solution(res_permuted)

        self._linear_solve_stats.petsc_converged_reason = info
        self._linear_solve_stats.time_solve_linear_system = t_solve.elapsed_time
        self._linear_solve_stats.gmres_iters = len(gmres_.get_residuals())
        return np.atleast_1d(res)


def make_reorder_contact(model):
    # Combines normal and tangential equations into one equation, AoS alignment
    dofs_contact = np.concatenate([model.eq_dofs[i] for i in model._equation_groups[4]])
    dofs_contact_start = dofs_contact[0]
    dofs_contact_end = dofs_contact[-1] + 1

    reorder = np.arange(model.equation_system.num_dofs())

    if model.nd == 2:
        dofs_contact_0 = dofs_contact[: len(dofs_contact) // model.nd]
        dofs_contact_1 = dofs_contact[len(dofs_contact) // model.nd :]
        reorder[dofs_contact_start:dofs_contact_end] = np.stack(
            [dofs_contact_0, dofs_contact_1]
        ).ravel("f")
    elif model.nd == 3:
        div = len(dofs_contact) // model.nd
        dofs_contact_0 = dofs_contact[:div]
        dofs_contact_1 = dofs_contact[div::2]
        dofs_contact_2 = dofs_contact[div + 1 :: 2]
        reorder[dofs_contact_start:dofs_contact_end] = np.stack(
            [dofs_contact_0, dofs_contact_1, dofs_contact_2]
        ).ravel("f")
    else:
        raise ValueError(f"{model.nd = }")
    return reorder


def reorder_J44(model):
    assert model.nd == 2
    dofs_contact = np.concatenate([model.eq_dofs[i] for i in model._equation_groups[4]])
    reorder = np.zeros(dofs_contact.size, dtype=int)
    half = reorder.size // 2
    reorder[1::2] = np.arange(half)
    reorder[::2] = np.arange(half) + half
    return reorder


def correct_eq_groups(model):
    """We reindex eq_dofs and model._equation_groups to put together normal and
    tangential components of the contact equation for each dof.
    Previously, the groups were: [[f0_norm], [f1_norm], [f0_tang], [f1_tang]].
    This returns new indices: [[f0_norm, f0_tang], [f1_norm, f1_tang]].
    """
    eq_dofs_corrected = [x.copy() for x in model.eq_dofs]
    eq_groups_corrected = [x.copy() for x in model._equation_groups]

    # assert model.nd == 2

    # assume normal equations go first.
    # 2 is hardcoded because both tangential component lay in the same group.
    end_normal = len(model._equation_groups[4]) // 2
    normal_subgroups = model._equation_groups[4][:end_normal]

    eq_dofs_corrected = []
    for i, x in enumerate(model.eq_dofs):
        if i not in model._equation_groups[4]:
            eq_dofs_corrected.append(x)
        else:
            if i in normal_subgroups:
                eq_dofs_corrected.append(None)

    i = model.eq_dofs[normal_subgroups[0]][0]
    for normal in normal_subgroups:
        res = i + np.arange(model.eq_dofs[normal].size * model.nd)
        i = res[-1] + 1
        eq_dofs_corrected[normal] = np.array(res)

    eq_groups_corrected[4] = normal_subgroups
    eq_groups_corrected[5] = (np.array(model._equation_groups[5]) - end_normal).tolist()

    return eq_dofs_corrected, eq_groups_corrected
