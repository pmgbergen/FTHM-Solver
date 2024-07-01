import sys
import time
from functools import cached_property, partial
from pathlib import Path

import numpy as np
import porepy as pp
import scipy.sparse
from matplotlib import pyplot as plt
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
from porepy.models.momentum_balance import BoundaryConditionsMomentumBalance

from block_matrix import BlockMatrixStorage, SolveSchema, make_solver
from fixed_stress import get_fixed_stress_stabilization, make_fs, make_fs_experimental
from mat_utils import (
    PetscAMGFlow,
    PetscAMGMechanics,
    PetscGMRES,
    PetscILU,
    TimerContext,
    csr_ones,
    extract_diag_inv,
    inv_block_diag,
    inv,
)
from plot_utils import dump_json
from preconditioner_mech import make_J44_inv_bdiag
from stats import LinearSolveStats, TimeStepStats


class CheckStickingSlidingOpen:

    def sticking_sliding_open(self):
        print("You might want to consider transition as well")
        frac_dim = self.nd - 1
        subdomains = self.mdg.subdomains(dim=frac_dim)

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")

        # The complimentarity condition
        num_cells = sum([sd.num_cells for sd in subdomains])
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))
        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")
        b_p = b_p.value(self.equation_system)
        open_cells = b_p < 1e-5  # THIS IS THE SAME CONTACT MECHANICS TOLERANCE
        # (REVISIT AFTER MERGE WITH POREPY)

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
        sticking_cells = (b_p > norm) & np.logical_not(open_cells)

        sliding_cells = True ^ (sticking_cells | open_cells)
        return sticking_cells, sliding_cells, open_cells

    def sticking_sliding_open_transition(self):
        frac_dim = self.nd - 1
        subdomains = self.mdg.subdomains(dim=frac_dim)

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")

        # The complimentarity condition
        num_cells = sum([sd.num_cells for sd in subdomains])
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))
        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")
        b_p = b_p.value(self.equation_system)
        open_cells = b_p == 0
        transition_cells = (0 < b_p) & (b_p < 1e-5)
        # THIS IS THE SAME CONTACT MECHANICS TOLERANCE
        # (REVISIT AFTER MERGE WITH POREPY)
        open_and_transition = open_cells | transition_cells

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
        sticking_cells = (b_p > norm) & np.logical_not(open_and_transition)

        sliding_cells = True ^ (sticking_cells | open_and_transition)
        return sticking_cells, sliding_cells, open_cells, transition_cells


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


class DymanicTimeStepping(pp.SolutionStrategy):
    # For some reason, PP does not increase / decrease time step.
    # Should check whether is has been added lately.
    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        self.time_manager.compute_time_step(iteration_counter, recompute_solution=False)

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        prev_sol = self.equation_system.get_variable_values(time_step_index=0)
        self.equation_system.set_variable_values(prev_sol, iterate_index=0)
        self.time_manager.compute_time_step(recompute_solution=True)


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


class StatisticsSavingMixin(CheckStickingSlidingOpen, pp.SolutionStrategy):
    _linear_solve_stats: LinearSolveStats
    _time_step_stats: TimeStepStats

    @cached_property
    def statistics(self) -> list[TimeStepStats]:
        return []

    def simulation_name(self) -> str:
        sim_name = self.params.get("simulation_name", "simulation")
        if hasattr(self, "bc_name"):
            sim_name = f"{sim_name}_{self.bc_name}"
        use_direct = not self.params.get("iterative_solver", True)
        if use_direct:
            sim_name = f"{sim_name}_direct"
        solver_type = self.params.get("solver_type", "baseline")
        if solver_type != '2':
            sim_name = f"{sim_name}_solver_{solver_type}"
        return sim_name

    def before_nonlinear_loop(self) -> None:
        self._time_step_stats = TimeStepStats()
        self.statistics.append(self._time_step_stats)
        super().before_nonlinear_loop()

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        dump_json(self.simulation_name() + ".json", self.statistics)
        super().after_nonlinear_convergence(solution, errors, iteration_counter)

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self._time_step_stats.nonlinear_convergence_status = -1
        dump_json(self.simulation_name() + ".json", self.statistics)
        super().after_nonlinear_failure(solution, errors, iteration_counter)

    def before_nonlinear_iteration(self) -> None:
        self._linear_solve_stats = LinearSolveStats()
        super().before_nonlinear_iteration()

        data = self.sticking_sliding_open_transition()
        self._linear_solve_stats.sticking = data[0].tolist()
        self._linear_solve_stats.sliding = data[1].tolist()
        self._linear_solve_stats.open_ = data[2].tolist()
        self._linear_solve_stats.transition = data[3].tolist()

        characteristic = self._characteristic.value(self.equation_system).tolist()
        self._linear_solve_stats.transition_sticking_sliding = characteristic

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        save_path = Path("./matrices")
        save_path.mkdir(exist_ok=True)
        mat, rhs = self.linear_system
        name = f"{self.simulation_name()}_{int(time.time() * 1000)}"
        if self.params.get("save_arrays", True):
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
        dump_json(self.simulation_name() + ".json", self.statistics)
        super().after_nonlinear_iteration(solution_vector)


class MyPetscSolver(pp.SolutionStrategy):

    _solver_initialized = False
    _linear_solve_stats = LinearSolveStats()  # placeholder

    def before_nonlinear_loop(self) -> None:
        if not self._solver_initialized:
            self._initialize_solver()
        return super().before_nonlinear_loop()

    def _initialize_solver(self):
        self.eq_dofs, self.var_dofs = make_row_col_dofs(self)
        self._variable_groups = self.make_variables_groups()
        self._equation_groups = self.make_equations_groups()

        self.eq_group_dofs = [
            np.concatenate([self.eq_dofs[block] for block in group])
            for group in self._equation_groups
        ]

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
        if len(self._equation_groups[4]) != 0:
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

        elif solver_type == "1":
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
                        # invertor=lambda bmat: make_fs(self, bmat).mat,
                        invertor=lambda bmat: self._fixed_stress.mat,
                        invertor_type="physical",
                        complement=SolveSchema(
                            groups=[0, 2],
                            solve=lambda bmat: PetscAMGFlow(mat=bmat[[0, 2]].mat),
                        ),
                    ),
                ),
            )
        elif solver_type == "2":
            return SolveSchema(
                groups=[4],
                solve=lambda bmat: inv_block_diag(mat=bmat[[4]].mat, nd=self.nd),
                complement=SolveSchema(
                    groups=[3],
                    solve=lambda bmat: PetscILU(bmat[[3]].mat),
                    invertor=lambda bmat: extract_diag_inv(bmat[[3]].mat),
                    # complement=SolveSchema(
                    #     groups=[5],
                    #     solve=lambda bmat: PetscAMGMechanics(
                    #         mat=bmat[[5]].mat, dim=self.nd
                    #     ),
                    #     invertor=lambda bmat: inv_block_diag(
                    #         mat=bmat[[5]].mat, nd=self.nd, lump=True
                    #     ),
                    complement=SolveSchema(
                        groups=[1, 5],
                        solve=lambda bmat: PetscAMGMechanics(
                            mat=bmat[[1, 5]].mat,
                            dim=self.nd,
                            null_space=self.build_mechanics_near_null_space(),
                        ),
                        invertor_type="physical",
                        invertor=lambda bmat: make_fs_experimental(self, bmat).mat,
                        complement=SolveSchema(
                            groups=[0, 2],
                            solve=lambda bmat: PetscAMGFlow(mat=bmat[[0, 2]].mat),
                        ),
                    ),
                    # ),
                ),
            )
            # return SolveSchema(
            #     groups=[4],
            #     solve=lambda bmat: inv_block_diag(mat=bmat[[4]].mat, nd=self.nd),
            #     complement=SolveSchema(
            #         groups=[3],
            #         solve=lambda bmat: PetscILU(bmat[[3]].mat),
            #         invertor=lambda bmat: extract_diag_inv(bmat[[3]].mat),
            #         complement=SolveSchema(
            #             groups=[1, 5],
            #             solve=lambda bmat: PetscAMGMechanics(
            #                 mat=bmat[[1, 5]].mat, dim=self.nd
            #             ),
            #             # invertor=lambda bmat: self._fixed_stress.mat,
            #             invertor=lambda bmat: make_fs_experimental(self, bmat).mat,
            #             invertor_type="physical",
            #             complement=SolveSchema(
            #                 groups=[0, 2],
            #                 solve=lambda bmat: PetscAMGFlow(mat=bmat[[0, 2]].mat),
            #             ),
            #         ),
            #     ),
            # )
        elif solver_type == "2_exact":
            return SolveSchema(
                groups=[4],
                solve=lambda bmat: inv_block_diag(mat=bmat[[4]].mat, nd=self.nd),
                complement=SolveSchema(
                    groups=[3],
                    # solve=lambda bmat: PetscILU(bmat[[3]].mat),
                    solve='direct',
                    invertor=lambda bmat: extract_diag_inv(bmat[[3]].mat),
                    complement=SolveSchema(
                        groups=[1, 5],
                        # solve=lambda bmat: PetscAMGMechanics(
                        #     mat=bmat[[1, 5]].mat,
                        #     dim=self.nd,
                        #     null_space=self.build_mechanics_near_null_space(),
                        # ),
                        solve='direct',
                        invertor_type="physical",
                        invertor=lambda bmat: make_fs_experimental(self, bmat).mat,
                        complement=SolveSchema(
                            groups=[0, 2],
                            solve='direct',
                            # solve=lambda bmat: PetscAMGFlow(mat=bmat[[0, 2]].mat),
                        ),
                    ),
                    # ),
                ),
            )
        elif solver_type == "2_exact_only_fs":
            return SolveSchema(
                groups=[4],
                solve=lambda bmat: inv_block_diag(mat=bmat[[4]].mat, nd=self.nd),
                complement=SolveSchema(
                    groups=[3],
                    # solve=lambda bmat: PetscILU(bmat[[3]].mat),
                    solve='direct',
                    # invertor=lambda bmat: extract_diag_inv(bmat[[3]].mat),
                    complement=SolveSchema(
                        groups=[1, 5],
                        # solve=lambda bmat: PetscAMGMechanics(
                        #     mat=bmat[[1, 5]].mat,
                        #     dim=self.nd,
                        #     null_space=self.build_mechanics_near_null_space(),
                        # ),
                        solve='direct',
                        invertor_type="physical",
                        invertor=lambda bmat: make_fs_experimental(self, bmat).mat,
                        complement=SolveSchema(
                            groups=[0, 2],
                            solve='direct',
                            # solve=lambda bmat: PetscAMGFlow(mat=bmat[[0, 2]].mat),
                        ),
                    ),
                    # ),
                ),
            )

        raise ValueError(f"{solver_type}")

    def build_mechanics_near_null_space(self, groups=(1, 5)):
        cell_centers = []
        if 1 in groups:
            cell_centers.append(self.mdg.subdomains(dim=self.nd)[0].cell_centers)
        if 5 in groups:
            cell_centers.extend(
                [intf.cell_centers for intf in self.mdg.interfaces(dim=self.nd - 1)]
            )
        cell_centers = np.concatenate(cell_centers, axis=1)

        x, y, z = cell_centers
        num_dofs = cell_centers.shape[1]

        null_space = []
        if self.nd == 3:
            vec = np.zeros((3, num_dofs))
            vec[0] = 1
            null_space.append(vec.ravel("f"))
            vec = np.zeros((3, num_dofs))
            vec[1] = 1
            null_space.append(vec.ravel("f"))
            vec = np.zeros((3, num_dofs))
            vec[2] = 1
            null_space.append(vec.ravel("f"))
            # # 0, -z, y
            vec = np.zeros((3, num_dofs))
            vec[1] = -z
            vec[2] = y
            null_space.append(vec.ravel("f"))
            # z, 0, -x
            vec = np.zeros((3, num_dofs))
            vec[0] = z
            vec[2] = -x
            null_space.append(vec.ravel("f"))
            # -y, x, 0
            vec = np.zeros((3, num_dofs))
            vec[0] = -y
            vec[1] = x
            null_space.append(vec.ravel("f"))
        elif self.nd == 2:
            vec = np.zeros((2, num_dofs))
            vec[0] = 1
            null_space.append(vec.ravel("f"))
            vec = np.zeros((2, num_dofs))
            vec[1] = 1
            null_space.append(vec.ravel("f"))
            # -x, y
            vec = np.zeros((2, num_dofs))
            vec[0] = -x
            vec[1] = y
            null_space.append(vec.ravel("f"))
        else:
            raise ValueError

        return np.array(null_space)

    @cached_property
    def _fixed_stress(self):
        # Assuming blocks [1, 5] don't change. But the block [2, 5] changes....
        return make_fs(self, self.bmat)

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
                group_row_names=[
                    "Flow mat.",
                    "Force mat.",
                    "Flow frac.",
                    "Flow intf.",
                    "Contact frac.",
                    "Force intf.",
                ],
                group_col_names=[
                    r"$p_{3D}$",
                    r"$u_{3D}$",
                    r"$p_{frac}$",
                    r"$v_{intf}$",
                    r"$\lambda_{frac}$",
                    r"$u_{intf}$",
                ],
            )
            # reordering the matrix to the order I work with, not how PorePy provides it
            bmat = bmat[:]
            self.bmat = bmat
            schema = self.make_solver_schema()
            self._Qleft = None
            self._Qright = None
            self.Qleft = None
            self.Qright = None
            solver_type = self.params.get("solver_type", "baseline")
            if solver_type.startswith('2'):
                if solver_type == '2_exact_only_fs':
                    J55_inv = inv(bmat[[5]].mat)
                    eig_max_right = 1
                    eig_max_left = 1
                else:
                    J55_inv = inv_block_diag(bmat[[5]].mat, nd=self.nd, lump=False)
                    eig_max_right = abs(bmat[[5]].mat @ J55_inv).data.max()
                    eig_max_left = abs(J55_inv @ bmat[[5]].mat).data.max()

                Qleft = bmat.empty_container()
                Qleft.mat = csr_ones(Qleft.shape[0])
                Qright = Qleft.copy()

                Qleft[4, 5] = -bmat[4, 5].mat @ J55_inv / eig_max_left
                Qright[5, 4] = -J55_inv @ bmat[5, 4].mat / eig_max_right

                # self.Qleft = Qleft
                self.Qright = Qright
                self._Qleft = Qleft
                self._Qright = Qright

                # J_Q = bmat.empty_container()
                # J_Q.mat = Qleft.mat @ bmat.mat

                # bmat_reordered, preconditioner = make_solver(schema, J_Q)
                # Q_perm = Qleft[bmat_reordered.active_groups]
                # self.rhs_Q = Q_perm.mat @ Q_perm.local_rhs(rhs)
                # self.Q_perm = Q_perm
            # else:
            #     bmat_reordered, preconditioner = make_solver(
            #         schema=schema, mat_orig=bmat
            #     )

        self._linear_solve_stats.time_prepare_solver = t_prepare_solver.elapsed_time

        # return bmat_reordered, preconditioner
        return schema

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
            schema = self._prepare_solver()

            mat_Q = self.bmat.copy()
            Qleft, Qright = self.Qleft, self.Qright
            if Qleft is not None:
                assert Qleft.active_groups == self.bmat.active_groups
                mat_Q.mat = Qleft.mat @ mat_Q.mat
            if Qright is not None:
                assert Qright.active_groups == self.bmat.active_groups
                mat_Q.mat = mat_Q.mat @ Qright.mat

            mat_Q_permuted, prec = make_solver(schema, mat_Q)

            rhs_local = mat_Q_permuted.local_rhs(rhs)

            rhs_Q = rhs_local.copy()
            if Qleft is not None:
                Qleft = Qleft[mat_Q_permuted.active_groups]
                rhs_Q = Qleft.mat @ rhs_Q

            tol = 1e-10
            pc_side = "right"
            solver_type = self.params.get("solver_type", "baseline")
            if solver_type == "1":
                tol = 1e-10
                pc_side = "left"

            if solver_type == "2":
                pc_side = "right"
                # tol = 1e-6
                # pc_side = 'left'
                tol = 1e-10

            gmres_ = PetscGMRES(
                mat=mat_Q_permuted.mat,
                pc=prec,
                pc_side=pc_side,
                tol=tol,
            )

            with TimerContext() as t_gmres:
                sol_Q = gmres_.solve(rhs_Q)
                # print(len(gmres_.get_residuals()))
                # mat_Q_permuted.color_local_rhs(rhs_Q); plt.show()

            self._linear_solve_stats.time_gmres = t_gmres.elapsed_time

            info = gmres_.ksp.getConvergedReason()

            if Qright is not None:
                Qright = Qright[mat_Q_permuted.active_groups]
                sol = mat_Q_permuted.reverse_transform_solution(Qright.mat @ sol_Q)
            else:
                sol = mat_Q_permuted.reverse_transform_solution(sol_Q)

            true_residual_nrm_drop = abs(mat @ sol - rhs).max() / abs(rhs).max()

            if info <= 0:
                print(f"GMRES failed, {info=}", file=sys.stderr)
                if info == -9:
                    sol[:] = np.nan
            else:
                if true_residual_nrm_drop >= 1:
                    print("True residual did not decrease")

        self._linear_solve_stats.petsc_converged_reason = info
        self._linear_solve_stats.time_solve_linear_system = t_solve.elapsed_time
        self._linear_solve_stats.gmres_iters = len(gmres_.get_residuals())
        return np.atleast_1d(sol)


def make_reorder_contact(model):
    # Combines normal and tangential equations into one equation, AoS alignment
    if len(model._equation_groups[4]) == 0:
        return np.array([])
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


def correct_eq_groups(model):
    """We reindex eq_dofs and model._equation_groups to put together normal and
    tangential components of the contact equation for each dof.
    Previously, the groups were: [[f0_norm], [f1_norm], [f0_tang], [f1_tang]].
    This returns new indices: [[f0_norm, f0_tang], [f1_norm, f1_tang]].
    """
    if len(model._equation_groups[4]) == 0:
        return model.eq_dofs, model._equation_groups

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


class NewtonBacktracking(pp.SolutionStrategy):

    def _test_residual_norm(self, alpha: float, delta_sol: np.ndarray):
        original_values = self.equation_system.get_variable_values(iterate_index=0)
        return (
            self.compute_nonlinear_residual(state=original_values + delta_sol * alpha)
            / self._newton_res_0
        )

    def plot_residual_drop(self, alphas, res_norms):
        group_names = [
            "Mass 3d",
            "Force 3d",
            "Mass frac",
            "Intf flux",
            "Contact",
            "Intf force",
        ]
        for nrm, label in zip(res_norms, group_names):
            plt.plot(alphas, nrm, label=label)
        plt.yscale("log")
        plt.legend()
        plt.show()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        delta_sol = solution_vector
        del solution_vector
        # opt_res = minimize_scalar(
        #     lambda a: self._test_residual_norm(a, delta_sol).max(),
        #     bounds=[0.5, 1],
        #     options={"maxiter": 6},
        # )
        # alpha = opt_res.x

        alphas = np.linspace(0.5, 1.1, 7, endpoint=True)
        alphas = alphas[alphas != 0]
        res_norms = np.array(
            [self._test_residual_norm(alpha, delta_sol=delta_sol) for alpha in alphas]
        ).T
        alpha = alphas[np.argmin(np.max(res_norms, axis=0))]

        delta_sol *= alpha

        # self.plot_residual_drop(alphas, res_norms)
        # print(file=sys.stderr, flush=True)
        # print(f"{alpha = }", flush=True)
        super().after_nonlinear_iteration(delta_sol)
        pass

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        self._newton_res_0 = self.compute_nonlinear_residual(replace_zeros=True)

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params,
    ) -> tuple[float, bool, bool]:
        res_norm_groups = self.compute_nonlinear_residual() / self._newton_res_0
        res_norm = res_norm_groups.sum()
        # print(f"{res_norm = }")
        converged = res_norm < nl_params["nl_convergence_tol"]
        diverged = res_norm > nl_params["nl_divergence_tol"]
        if not np.all(np.isfinite(res_norm)) or not np.all(np.isfinite(solution)):
            diverged = True
        # if diverged:
        #     print('diverged')
        return res_norm, converged, diverged

    def compute_nonlinear_residual(
        self, replace_zeros: bool = False, state: np.ndarray = None
    ):
        nonlinear_residual = self.equation_system.assemble(
            evaluate_jacobian=False, state=state
        )
        group_residuals = []
        for dofs in self.eq_group_dofs:
            nrm = np.linalg.norm(nonlinear_residual[dofs])
            group_residuals.append(nrm)

        group_residuals = np.array(group_residuals)

        atol = 1e-10
        group_residuals[group_residuals < atol] = 0

        if replace_zeros:
            group_residuals[group_residuals == 0] = 1
            # contact mechanics always starts from 1
            group_residuals[4] = 1
        return group_residuals


class NewtonBacktrackingSimple(pp.SolutionStrategy):

    def _test_residual_norm(self, alpha: float, delta_sol: np.ndarray):
        original_values = self.equation_system.get_variable_values(iterate_index=0)
        return np.linalg.norm(
            self.compute_nonlinear_residual(state=original_values + delta_sol * alpha)
            / self._newton_res_0
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        delta_sol = solution_vector
        del solution_vector

        alphas = np.linspace(0.5, 1.1, 7, endpoint=True)
        alphas = alphas[alphas != 0]
        res_norms = [
            self._test_residual_norm(alpha, delta_sol=delta_sol) for alpha in alphas
        ]

        alpha = alphas[np.argmin(res_norms)]

        delta_sol *= alpha

        self.plot_residual_drop(alphas, res_norms)
        # print(file=sys.stderr, flush=True)
        # print(f"{alpha = }", flush=True)
        super().after_nonlinear_iteration(delta_sol)

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        self._newton_res_0 = self.compute_nonlinear_residual(replace_zeros=True)

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params,
    ) -> tuple[float, bool, bool]:
        res_norm = self.compute_nonlinear_residual() / self._newton_res_0
        # print(f"{res_norm = }")
        converged = res_norm < nl_params["nl_convergence_tol"]
        diverged = res_norm > nl_params["nl_divergence_tol"]
        # if diverged:
        #     print('diverged')
        return res_norm, converged, diverged

    def compute_nonlinear_residual(
        self, replace_zeros: bool = False, state: np.ndarray = None
    ):
        res = np.linalg.norm(
            self.equation_system.assemble(evaluate_jacobian=False, state=state)
        )
        atol = 1e-10
        if res < atol:
            if replace_zeros:
                res = 1
            else:
                res = 0
        return res

    def plot_residual_drop(self, alphas, res_norms):
        plt.plot(alphas, res_norms)
        plt.yscale("log")
        plt.show()
