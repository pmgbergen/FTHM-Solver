import sys
from functools import cached_property
from typing import Sequence

import numpy as np
import porepy as pp
from porepy.models.solution_strategy import SolutionStrategy

from block_matrix import BlockMatrixStorage, FieldSplitScheme

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
from stats import LinearSolveStats


class IterativeLinearSolver(pp.SolutionStrategy):

    _linear_solve_stats = LinearSolveStats()
    """A placeholder to statistics. The solver mixin only writes in it, not reads."""

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
    def eq_dofs(self) -> list[np.ndarray]:
        """Equation degrees of freedom (rows of the Jacobian) in the PorePy order (how
        they are arranged in the model).

        Each list entry correspond to one equation on one grid. Constructed when first
        accessed. Encorporates the permutation `contact_permutation`.

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
        raise NotImplementedError

    @cached_property
    def equation_groups(self) -> list[list[int]]:
        raise NotImplementedError

    def group_row_names(self) -> list[str] | None:
        return None

    def group_col_names(self) -> list[str] | None:
        return None

    def make_solver_scheme(self) -> FieldSplitScheme:
        raise NotImplementedError

    def assemble_linear_system(self) -> None:
        super().assemble_linear_system()
        mat, rhs = self.linear_system

        bmat = BlockMatrixStorage(
            mat=mat,
            global_dofs_row=self.eq_dofs,
            global_dofs_col=self.var_dofs,
            groups_to_blocks_row=self.equation_groups,
            groups_to_blocks_col=self.variable_groups,
            group_names_row=self.group_row_names(),
            group_names_col=self.group_col_names(),
        )

        self.bmat = bmat

    def solve_linear_system(self) -> np.ndarray:
        # Check that rhs is finite.
        mat, rhs = self.linear_system
        if not np.all(np.isfinite(rhs)):
            self._linear_solve_stats.krylov_iters = 0
            result = np.zeros_like(rhs)
            result[:] = np.nan
            return result

        # Check if we reached steady state and no solve needed.
        # residual_norm = self.compute_residual_norm(rhs, None)
        # if residual_norm < self.params["nl_convergence_tol_res"]:
        #     result = np.zeros_like(rhs)
        #     return result

        if self.params["setup"].get("save_matrix", False):
            self.save_matrix_state()

        scheme = self.make_solver_scheme()
        # Constructing the solver.
        bmat = self.bmat[scheme.get_groups()]

        try:
            solver = scheme.make_solver(bmat)
        except:
            self.save_matrix_state()
            raise

        # Permute the rhs groups to match mat_permuted.
        rhs_local = bmat.project_rhs_to_local(rhs)

        try:
            sol_local = solver.solve(rhs_local)
        except:
            self.save_matrix_state()
            raise

        info = solver.ksp.getConvergedReason()

        # Permute the solution groups to match the original porepy arrangement.
        sol = bmat.project_solution_to_global(sol_local)

        # Verify that the original problem is solved and we did not do anything wrong.
        true_residual_nrm_drop = abs(mat @ sol - rhs).max() / abs(rhs).max()

        if info <= 0:
            print(f"GMRES failed, {info=}", file=sys.stderr)
            if info == -9:
                sol[:] = np.nan
        else:
            if true_residual_nrm_drop >= 1:
                print("True residual did not decrease")

        # Write statistics
        self._linear_solve_stats.petsc_converged_reason = info
        self._linear_solve_stats.krylov_iters = len(solver.get_residuals())
        return np.atleast_1d(sol)


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
