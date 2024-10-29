import sys
import time
from functools import cached_property, partial
from pathlib import Path
from typing import Sequence

import numpy as np
import porepy as pp
import scipy.sparse
from porepy.models.solution_strategy import SolutionStrategy, ContactIndicators
import scipy.sparse.linalg

from block_matrix import BlockMatrixStorage, FieldSplitScheme, make_solver
from fixed_stress import (
    make_fs_analytical,
    make_fs_analytical_with_interface_flow,
)
from mat_utils import (
    PetscAMGFlow,
    PetscAMGMechanics,
    PetscGMRES,
    PetscILU,
    PetscRichardson,
    csr_ones,
    extract_diag_inv,
    inv_block_diag,
)
from plot_utils import dump_json
from stats import LinearSolveStats, TimeStepStats


class StatisticsSavingMixin(ContactIndicators, pp.SolutionStrategy):
    _linear_solve_stats: LinearSolveStats
    _time_step_stats: TimeStepStats

    @cached_property
    def statistics(self) -> list[TimeStepStats]:
        return []

    def simulation_name(self) -> str:
        name = "stats"
        setup = self.params["setup"]
        name = f'{name}_geo{setup["geometry"]}x{setup["grid_refinement"]}'
        name = f'{name}_sol{setup["solver"]}'
        name = f'{name}_ph{setup["physics"]}'
        name = f'{name}_bb{setup["barton_bandis_stiffness_type"]}'
        name = f'{name}_fr{setup["friction_type"]}'
        return name

    def before_nonlinear_loop(self) -> None:
        self._time_step_stats = TimeStepStats()
        self.statistics.append(self._time_step_stats)
        print()
        print(f"Sim time: {self.time_manager.time}, Dt: {self.time_manager.dt}")
        super().before_nonlinear_loop()

    def after_nonlinear_convergence(self) -> None:
        dump_json(self.simulation_name() + ".json", self.statistics)
        super().after_nonlinear_convergence()

    def after_nonlinear_failure(self) -> None:
        self._time_step_stats.nonlinear_convergence_status = -1
        dump_json(self.simulation_name() + ".json", self.statistics)
        print("Time step did not converge")
        super().after_nonlinear_failure()

    def before_nonlinear_iteration(self) -> None:
        self._linear_solve_stats = LinearSolveStats()
        super().before_nonlinear_iteration()
        self.collect_stats_sticking_sliding_open()
        self.collect_stats_ut_mismatch()
        self.collect_stats_coulomb_mismatch()
        self.collect_stats_u_lambda_max()

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        print(
            f"Newton iter: {len(self._time_step_stats.linear_solves)}, "
            f"Krylov iters: {self._linear_solve_stats.krylov_iters}"
        )
        self._linear_solve_stats.simulation_dt = self.time_manager.dt
        self._time_step_stats.linear_solves.append(self._linear_solve_stats)
        if self.params["setup"].get("save_matrix", False):
            self.save_matrix_state()
        dump_json(self.simulation_name() + ".json", self.statistics)
        super().after_nonlinear_iteration(solution_vector)

    def sticking_sliding_open(self):
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        opening = self.opening_indicator(fractures).value(self.equation_system) < 0
        closed = np.logical_not(opening)
        sliding = np.logical_and(
            closed, self.sliding_indicator(fractures).value(self.equation_system) > 0
        )
        sticking = np.logical_not(opening | sliding)

        return sticking, sliding, opening

    def collect_stats_sticking_sliding_open(self):
        data = self.sticking_sliding_open()
        self._linear_solve_stats.num_sticking = int(sum(data[0]))
        self._linear_solve_stats.num_sliding = int(sum(data[1]))
        self._linear_solve_stats.num_open = int(sum(data[2]))
        print(
            f"sticking: {self._linear_solve_stats.num_sticking}, "
            f"sliding: {self._linear_solve_stats.num_sliding}, "
            f"open: {self._linear_solve_stats.num_open}"
        )

    def collect_stats_ut_mismatch(self):
        sticking, _, _ = self.sticking_sliding_open()
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(fractures)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(fractures)
        u_t_increment = pp.ad.time_increment(u_t).value(self.equation_system)

        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            fractures, dim=self.nd - 1
        )
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        ).value(self.equation_system)
        sticking = (scalar_to_tangential @ sticking).astype(bool)

        u_t_sticking = u_t_increment[sticking]
        try:
            self._linear_solve_stats.sticking_u_mismatch = abs(u_t_sticking).max()
        except ValueError:
            self._linear_solve_stats.sticking_u_mismatch = 0

    def collect_stats_coulomb_mismatch(self):
        _, sliding, _ = self.sticking_sliding_open()

        fractures = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(fractures)
        t_t = (nd_vec_to_tangential @ self.contact_traction(fractures)).value(
            self.equation_system
        )
        b = self.friction_bound(fractures).value(self.equation_system)
        tangential_basis = self.basis(fractures, dim=self.nd - 1)
        t_t_nrm = np.sqrt(sum(comp._mat.T @ t_t**2 for comp in tangential_basis))

        diff = (-t_t_nrm + b)[sliding]
        try:
            self._linear_solve_stats.coulomb_mismatch = abs(diff).max()
        except ValueError:
            self._linear_solve_stats.coulomb_mismatch = 0

    def collect_stats_u_lambda_max(self):
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(fractures)
        nd_vec_to_normal = self.normal_component(fractures)

        t = self.contact_traction(fractures)
        u = self.displacement_jump(fractures)

        t_n = (nd_vec_to_normal @ t).value(self.equation_system)
        t_t = (nd_vec_to_tangential @ t).value(self.equation_system)
        u_n = (nd_vec_to_normal @ u).value(self.equation_system)
        u_t = (nd_vec_to_tangential @ u).value(self.equation_system)

        self._linear_solve_stats.lambdan_max = abs(t_n).max()
        self._linear_solve_stats.lambdat_max = abs(t_t).max()
        self._linear_solve_stats.un_max = abs(u_n).max()
        self._linear_solve_stats.ut_max = abs(u_t).max()

    def save_matrix_state(self):
        save_path = Path("./matrices")
        save_path.mkdir(exist_ok=True)
        mat, rhs = self.linear_system
        name = f"{self.simulation_name()}_{int(time.time() * 1000)}"
        mat_id = f"{name}.npz"
        rhs_id = f"{name}_rhs.npy"
        state_id = f"{name}_state.npy"
        iterate_id = f"{name}_iterate.npy"
        scipy.sparse.save_npz(save_path / mat_id, self.bmat.mat)
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


class MyPetscSolver(pp.SolutionStrategy):

    _linear_solve_stats: LinearSolveStats

    bmat: BlockMatrixStorage
    """The current Jacobian."""

    @cached_property
    def var_dofs(self) -> list[np.ndarray]:
        """Variable degrees of freedom (columns of the Jacobian) in the PorePy order
        (how they are arranged in the model).

        Each list entry correspond to one variable on one grid. Constructed when first
        accessed.

        """
        var_dofs: list[np.ndarray] = []
        for var in self.equation_system.variables:
            var_dofs.append(self.equation_system.dofs_of([var]))
        return var_dofs

    @cached_property
    def _unpermuted_eq_dofs(self) -> list[np.ndarray]:
        """The version of `eq_dofs` that does not encorporates the permutation
        `contact_permutation`.

        """
        eq_dofs: list[np.ndarray] = []
        offset = 0
        for data in self.equation_system._equation_image_space_composition.values():
            local_offset = 0
            for dofs in data.values():
                eq_dofs.append(dofs + offset)
                local_offset += len(dofs)
            offset += local_offset
        return eq_dofs

    @cached_property
    def variable_groups(self) -> list[list[int]]:
        """Prepares the groups of variables in the specific order, that we will use in
        the block Jacobian to access the submatrices:

        `J[x, 0]` - matrix pressure variable;
        `J[x, 1]` - matrix displacement variable;
        `J[x, 2]` - lower-dim pressure variable;
        `J[x, 3]` - interface Darcy flux variable;
        `J[x, 4]` - contact traction variable;
        `J[x, 5]` - interface displacement variable;

        This index is not equivalen to PorePy model natural ordering. Constructed when
        first accessed.

        """
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
                [self.pressure(sd_ambient)],  # 0
                [self.displacement(sd_ambient)],  # 1
                [self.pressure(sd_lower)],  # 2
                [self.interface_darcy_flux(intf)],  # 3
                [self.contact_traction(sd_frac)],  # 4
                [self.interface_displacement(intf_frac)],  # 5
            ],
        )

    @cached_property
    def _unpermuted_equation_groups(self) -> list[list[int]]:
        """The version of `equation_groups` that does not encorporates the permutation
        `contact_permutation`.

        """
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

    @cached_property
    def contact_permutation(self) -> np.ndarray:
        """Permutation of the contact mechanics equations. Must be applied to the
        Jacobian.

        The PorePy arrangement is:
        `[[C0_norm], [C1_norm], [C0_tang], [C1_tang]]`,
        where `C0` and `C1` correspond to the contact equation on fractures 0 and 1.
        We permute it to:
        `[[f0_norm, f0_tang], [f1_norm, f1_tang]]`, a.k.a array of structures.

        """
        return make_reorder_contact(self)

    @cached_property
    def eq_dofs(self):
        """Equation degrees of freedom (rows of the Jacobian) in the PorePy order (how
        they are arranged in the model).

        Each list entry correspond to one equation on one grid. Constructed when first
        accessed. Encorporates the permutation `contact_permutation`.

        """
        if len(self._unpermuted_equation_groups[4]) == 0:
            return self._unpermuted_eq_dofs, self._unpermuted_equation_groups

        eq_dofs_corrected = [x.copy() for x in self._unpermuted_eq_dofs]

        # We assume that normal equations go first.
        num_fracs = len(self.mdg.subdomains(dim=self.nd - 1))
        normal_subgroups = self._unpermuted_equation_groups[4][:num_fracs]

        eq_dofs_corrected = []
        for i, x in enumerate(self._unpermuted_eq_dofs):
            if i not in self._unpermuted_equation_groups[4]:
                eq_dofs_corrected.append(x)
            else:
                if i in normal_subgroups:
                    eq_dofs_corrected.append(None)

        i = self._unpermuted_eq_dofs[normal_subgroups[0]][0]
        for normal in normal_subgroups:
            res = i + np.arange(self._unpermuted_eq_dofs[normal].size * self.nd)
            i = res[-1] + 1
            eq_dofs_corrected[normal] = np.array(res)

        return eq_dofs_corrected

    @cached_property
    def equation_groups(self):
        """Prepares the groups of equation in the specific order, that we will use in
        the block Jacobian to access the submatrices:

        `J[0, x]` - matrix mass balance equation;
        `J[1, x]` - matrix momentum balance equation;
        `J[2, x]` - lower-dim mass balance equation;
        `J[3, x]` - interface Darcy flux equation;
        `J[4, x]` - contact traction equations;
        `J[5, x]` - interface force balance equation;

        This index is not equivalen to PorePy model natural ordering. Constructed when
        first accessed. Encorporates the permutation `contact_permutation`.

        """
        if len(self._unpermuted_equation_groups[4]) == 0:
            return self._unpermuted_equation_groups

        eq_groups_corrected = [x.copy() for x in self._unpermuted_equation_groups]

        # We assume that normal equations go first.
        num_fracs = len(self.mdg.subdomains(dim=self.nd - 1))
        # Now each dof array in group 4 corresponds normal and tangential components of
        # contact relations on a specific fracture.
        eq_groups_corrected[4] = self._unpermuted_equation_groups[4][:num_fracs]
        # Since the number of groups decreased, we need to subtract the difference.
        eq_groups_corrected[5] = (
            np.array(self._unpermuted_equation_groups[5]) - num_fracs
        ).tolist()

        return eq_groups_corrected

    def Qright(self):
        """Assemble the right linear transformation."""
        J = self.bmat
        J55_inv = inv_block_diag(J[5, 5].mat, nd=self.nd, lump=False)
        Qright = J.empty_container()
        Qright.mat = csr_ones(Qright.shape[0])
        Qright[5, 4] = -J55_inv @ J[5, 4].mat

        E = (scipy.sparse.eye(J55_inv.shape[0]) - J[5, 5].mat @ J55_inv) @ J[5, 4].mat
        self._linear_solve_stats.error_matrix_contribution = (
            abs(E.data).max() / abs(J[5, 4].mat.data).max()
        )
        return Qright

    def Qleft(self) -> BlockMatrixStorage:
        """Assemble the left linear transformation."""
        J = self.bmat
        J55_inv = inv_block_diag(J[5, 5].mat, nd=self.nd, lump=False)
        Qleft = J.empty_container()
        Qleft.mat = csr_ones(Qleft.shape[0])
        Qleft[4, 5] = -J[4, 5].mat @ J55_inv
        return Qleft

    def assemble_linear_system(self) -> None:
        super().assemble_linear_system()
        mat, rhs = self.linear_system
        # Applies the `contact_permutation`.
        if len(self.equation_groups[4]) != 0:
            mat = mat[self.contact_permutation]
            rhs = rhs[self.contact_permutation]
            self.linear_system = mat, rhs

            bmat = BlockMatrixStorage(
                mat=mat,
                global_row_idx=self.eq_dofs,
                global_col_idx=self.var_dofs,
                groups_row=self.equation_groups,
                groups_col=self.variable_groups,
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

            # Reordering the matrix to the order I work with, not how PorePy provides
            # it. This is important, the solver relies on it.
            bmat = bmat[:]
            self.bmat = bmat

    def solve_gmres(self, tol) -> np.ndarray:
        mat, rhs = self.linear_system
        schema = make_solver_schema(self)

        do_left_transformation = False
        do_right_transformation = True

        mat_Q = self.bmat.copy()  # Transformed J
        if do_left_transformation:
            Qleft = self.Qleft()
            assert Qleft.active_groups == self.bmat.active_groups
            mat_Q.mat = Qleft.mat @ mat_Q.mat
        if do_right_transformation:
            Qright = self.Qright()
            assert Qright.active_groups == self.bmat.active_groups
            mat_Q.mat = mat_Q.mat @ Qright.mat

        mat_Q_permuted, prec = make_solver(schema, mat_Q)
        # Solver changes the order of groups so that the first-eliminated goes first.

        rhs_local = mat_Q_permuted.local_rhs(rhs)
        # Permute the rhs groups according to the solver.

        rhs_Q = rhs_local.copy()  # If Qleft is used, need to transform the rhs.
        if do_left_transformation:
            # Transform Qleft groups according to the solver.
            Qleft = Qleft[mat_Q_permuted.active_groups]
            rhs_Q = Qleft.mat @ rhs_Q

        gmres_ = PetscGMRES(
            mat=mat_Q_permuted.mat,
            pc=prec,
            pc_side="right",
            tol=tol,
        )
        sol_Q = gmres_.solve(rhs_Q)
        info = gmres_.ksp.getConvergedReason()

        # Reverse transformations
        if do_right_transformation:
            Qright = Qright[mat_Q_permuted.active_groups]
            sol = mat_Q_permuted.reverse_transform_solution(Qright.mat @ sol_Q)
        else:
            sol = mat_Q_permuted.reverse_transform_solution(sol_Q)

        # Verify that the original problem is solved and we did not do anything wrong.
        true_residual_nrm_drop = abs(mat @ sol - rhs).max() / abs(rhs).max()

        if info <= 0:
            print(f"GMRES failed, {info=}", file=sys.stderr)
            if info == -9:
                sol[:] = np.nan
        else:
            if true_residual_nrm_drop >= 1:
                print("True residual did not decrease")

        self._linear_solve_stats.petsc_converged_reason = info
        self._linear_solve_stats.krylov_iters = len(gmres_.get_residuals())
        return np.atleast_1d(sol)

    def solve_richardson(self, tol) -> np.ndarray:
        mat, rhs = self.linear_system
        schema = make_solver_schema(self)

        mat_permuted, prec = make_solver(schema, self.bmat)
        # Solver changes the order of groups so that the first-eliminated goes first.

        rhs_local = mat_permuted.local_rhs(rhs)
        # Permute the rhs groups according to the solver.

        richardson = PetscRichardson(
            mat=mat_permuted.mat, pc=prec, pc_side="left", tol=tol, atol=1e-8
        )

        sol_local = richardson.solve(rhs_local)
        info = richardson.ksp.getConvergedReason()

        sol = mat_permuted.reverse_transform_solution(sol_local)

        # Verify that the original problem is solved and we did not do anything wrong.
        true_residual_nrm_drop = abs(mat @ sol - rhs).max() / abs(rhs).max()

        if info <= 0:
            print(f"Richardson failed, {info=}", file=sys.stderr)
            if info == -9:
                sol[:] = np.nan
        else:
            if true_residual_nrm_drop >= 1:
                print("True residual did not decrease")

        self._linear_solve_stats.petsc_converged_reason = info
        self._linear_solve_stats.krylov_iters = len(richardson.get_residuals())
        return np.atleast_1d(sol)

    def solve_linear_system(self) -> np.ndarray:
        rhs = self.linear_system[1]
        if not np.all(np.isfinite(rhs)):
            self._linear_solve_stats.krylov_iters = 0
            result = np.zeros_like(rhs)
            result[:] = np.nan
            return result

        tol = 1e-8

        solver_type = self.params["setup"]["solver"]
        direct = solver_type == 0
        richardson = solver_type in [1]
        gmres = solver_type in [2, 11, 12]
        if direct:
            return scipy.sparse.linalg.spsolve(*self.linear_system)
        elif richardson:
            return self.solve_richardson(tol=tol)
        elif gmres:
            return self.solve_gmres(tol=tol)
        raise ValueError

    def make_solver_schema(self) -> FieldSplitScheme:
        solver_type = self.params["setup"]["solver"]

        if solver_type in [1, 11]:  # Theoretical solver.
            return FieldSplitScheme(
                # Exactly solve elasticity and contact mechanics, build fixed stress.
                groups=[1, 4, 5],
                invertor=lambda bmat: make_fs_analytical_with_interface_flow(
                    self, bmat
                ).mat,
                invertor_type="physical",
                complement=FieldSplitScheme(
                    groups=[0, 2, 3],
                ),
            )

        elif solver_type == 2:  # Scalable solver.
            return FieldSplitScheme(
                # Exactly eliminate contact mechanics (assuming linearly-transformed system)
                groups=[4],
                solve=lambda bmat: inv_block_diag(mat=bmat[[4]].mat, nd=self.nd),
                complement=FieldSplitScheme(
                    # Eliminate interface flow, it is not coupled with (1, 4, 5)
                    # Use diag() to approximate inverse and ILU to solve linear systems
                    groups=[3],
                    solve=lambda bmat: PetscILU(bmat[[3]].mat),
                    invertor=lambda bmat: extract_diag_inv(bmat[[3]].mat),
                    complement=FieldSplitScheme(
                        # Eliminate elasticity. Use AMG to solve linear systems and fixed
                        # stress to approximate inverse.
                        groups=[1, 5],
                        solve=lambda bmat: PetscAMGMechanics(
                            mat=bmat[[1, 5]].mat,
                            dim=self.nd,
                            null_space=build_mechanics_near_null_space(self),
                        ),
                        invertor_type="physical",
                        invertor=lambda bmat: make_fs_analytical(self, bmat).mat,
                        complement=FieldSplitScheme(
                            # Use AMG to solve mass balance.
                            groups=[0, 2],
                            solve=lambda bmat: PetscAMGFlow(mat=bmat[[0, 2]].mat),
                        ),
                    ),
                ),
            )

        elif solver_type == 12:
            return FieldSplitScheme(
                # Exactly eliminate contact mechanics (assuming linearly-transformed system)
                groups=[4],
                solve=lambda bmat: inv_block_diag(mat=bmat[[4]].mat, nd=self.nd),
                complement=FieldSplitScheme(
                    # Eliminate interface flow, it is not coupled with (1, 4, 5)
                    # Use diag() to approximate inverse and ILU to solve linear systems
                    groups=[3],
                    solve=lambda bmat: PetscILU(bmat[[3]].mat),
                    invertor=lambda bmat: extract_diag_inv(bmat[[3]].mat),
                    complement=FieldSplitScheme(
                        # TODO
                        groups=[1, 5],
                        solve="direct",
                        invertor_type="physical",
                        invertor=lambda bmat: make_fs_analytical(self, bmat).mat,
                        complement=FieldSplitScheme(
                            # TODO
                            groups=[0, 2],
                            solve="direct",
                        ),
                    ),
                ),
            )

        raise ValueError(f"{solver_type}")


def make_reorder_contact(model: MyPetscSolver) -> np.ndarray:
    """Permutation of the contact mechanics equations. The PorePy arrangement is:
    `[C_n^0, C_n^1, ..., C_n^K, C_y^0, C_z^0, C_y^1, C_z^1, ..., C_z^K, C_z^k]`,
    where `C_n` is a normal component, `C_y` and `C_z` are two tangential
    components. Superscript corresponds to its position in space. We permute it to:
    `[C_n^0, C_y^0, C_z^0, ..., C_n^K, C_y^K, C_z^K]`, a.k.a array of structures.

    """
    if len(model._unpermuted_equation_groups[4]) == 0:
        return np.array([])
    dofs_contact = np.concatenate(
        [model._unpermuted_eq_dofs[i] for i in model._unpermuted_equation_groups[4]]
    )
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


def build_mechanics_near_null_space(model: MyPetscSolver, groups=(1, 5)):
    cell_centers = []
    if 1 in groups:
        cell_centers.append(model.mdg.subdomains(dim=model.nd)[0].cell_centers)
    if 5 in groups:
        cell_centers.extend(
            [intf.cell_centers for intf in model.mdg.interfaces(dim=model.nd - 1)]
        )
    cell_centers = np.concatenate(cell_centers, axis=1)

    x, y, z = cell_centers
    num_dofs = cell_centers.shape[1]

    null_space = []
    if model.nd == 3:
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
    elif model.nd == 2:
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


def get_variables_group_ids(
    model: SolutionStrategy,
    md_variables_groups: Sequence[
        Sequence[pp.ad.MixedDimensionalVariable | pp.ad.Variable]
    ],
) -> list[list[int]]:
    """Used to assemble the index that will later help accessing the submatrix
    corresponding to a group of variables, which may include one or more variable.

    Example: Group 0 corresponds to the pressure on all the subdomains. It will contain
    indices [0, 1, 2] which point to the pressure variable dofs on sd1, sd2 and sd3,
    respectively. Combination of different variables in one group is also possible.

    """

    variable_to_idx = {var: i for i, var in enumerate(model.equation_system.variables)}
    indices = []
    for md_var_group in md_variables_groups:
        group_idx = []
        for md_var in md_var_group:
            group_idx.extend([variable_to_idx[var] for var in md_var.sub_vars])
        indices.append(group_idx)
    return indices


def get_equations_group_ids(
    model: SolutionStrategy,
    equations_group_order: Sequence[Sequence[tuple[str, pp.GridLikeSequence]]],
) -> list[list[int]]:
    """Used to assemble the index that will later help accessing the submatrix
    corresponding to a group of equation, which may include one or more equation.

    Example: Group 0 corresponds to the mass balance equation on all the subdomains.
    It will contain indices [0, 1, 2] which point to the mass balance equation dofs on
    sd1, sd2 and sd3, respectively. Combination of different equation in one group is
    also possible.

    """
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
