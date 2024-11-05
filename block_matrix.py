from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence
import itertools

import scipy.linalg
import seaborn as sns
import numpy as np
import matplotlib
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix
from matplotlib import pyplot as plt

from mat_utils import (
    FieldSplit,
    TwoStagePreconditioner,
    inv,
    cond,
    PetscGMRES,
    PetscRichardson,
)
from plot_utils import plot_mat, spy


def color_spy(
    mat,
    row_idx,
    col_idx,
    row_names=None,
    col_names=None,
    aspect: Literal["equal", "auto"] = "equal",
    show: bool = False,
    marker=None,
    draw_marker=True,
    color=True,
    hatch=True,
    alpha=0.3,
):

    if draw_marker:
        spy(mat, show=False, aspect=aspect, marker=marker)
    else:
        spy(csr_matrix(mat.shape), show=False, aspect=aspect)

    row_sep = [0]
    for row in row_idx:
        row_sep.append(row[-1] + 1)
    row_sep = sorted(row_sep)

    col_sep = [0]
    for col in col_idx:
        col_sep.append(col[-1] + 1)
    col_sep = sorted(col_sep)

    if row_names is None:
        row_names = list(range(len(row_sep) - 1))
    if col_names is None:
        col_names = list(range(len(col_sep) - 1))

    hatch_types = itertools.cycle(["/", "\\"])

    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_names)):
        ystart, yend = row_sep[i : i + 2]
        row_label_pos.append(ystart + (yend - ystart) / 2)
        kwargs = {}
        if color:
            kwargs["facecolor"] = f"C{i}"
        else:
            kwargs["fill"] = False
        if hatch:
            kwargs["hatch"] = next(hatch_types)
            # kwargs['color'] = 'none'
            kwargs["edgecolor"] = "red"
            # kwargs['facecolor'] = 'blue'

        plt.axhspan(ystart - 0.5, yend - 0.5, alpha=alpha, **kwargs)
    ax.yaxis.set_ticks(row_label_pos)
    ax.set_yticklabels(row_names, rotation=0)

    # hatch_types = itertools.cycle(["|", "-"])

    col_label_pos = []
    for i in range(len(col_names)):
        xstart, xend = col_sep[i : i + 2]
        col_label_pos.append(xstart + (xend - xstart) / 2)
        if color:
            kwargs["facecolor"] = f"C{i}"
        if hatch:
            kwargs["hatch"] = next(hatch_types)
        plt.axvspan(xstart - 0.5, xend - 0.5, alpha=alpha, **kwargs)
    ax.xaxis.set_ticks(col_label_pos)
    ax.set_xticklabels(col_names, rotation=0)

    if show:
        plt.show()


def get_nonzero_indices(A, row_indices, col_indices):
    """
    Get the indices of A.data that correspond to the specified subset of rows and columns.

    Parameters:
    A (csr_matrix): The input sparse matrix.
    row_indices (list or array): The list of row indices to consider.
    col_indices (list or array): The list of column indices to consider.

    Returns:
    list: Indices in A.data corresponding to non-zero elements in the specified subset.
    """
    result_indices = []
    col_set = set(col_indices)  # For quick lookup

    for row in row_indices:
        start_ptr = A.indptr[row]
        end_ptr = A.indptr[row + 1]

        for data_idx in range(start_ptr, end_ptr):
            col_idx = A.indices[data_idx]
            if col_idx in col_set:
                result_indices.append(data_idx)

    return result_indices


class BlockMatrixStorage:
    def __init__(
        self,
        mat: spmatrix,
        global_dofs_row: list[np.ndarray],
        global_dofs_col: list[np.ndarray],
        groups_to_blocks_row: list[list[int]],
        groups_to_blocks_col: list[list[int]],
        local_dofs_row: Optional[list[np.ndarray]] = None,
        local_dofs_col: Optional[list[np.ndarray]] = None,
        active_groups_row: Optional[list[int]] = None,
        active_groups_col: Optional[list[int]] = None,
        group_names_row: list[str] = None,
        group_names_col: list[str] = None,
    ):
        self.mat: spmatrix = mat
        self.groups_to_blocks_row: list[list[int]] = groups_to_blocks_row
        self.groups_to_blocks_col: list[list[int]] = groups_to_blocks_col
        self.group_names_row: Optional[list[str]] = group_names_row
        self.group_names_col: Optional[list[str]] = group_names_col

        def init_global_dofs(global_dofs: list[np.ndarray]):
            # Cast dofs to numpy arrays.
            return [np.atleast_1d(x) for x in global_dofs]

        self.global_dofs_row: list[np.ndarray] = init_global_dofs(global_dofs_row)
        self.global_dofs_col: list[np.ndarray] = init_global_dofs(global_dofs_col)

        def init_local_dofs(
            local_dofs: list[np.ndarray] | None, global_dofs: list[np.ndarray]
        ):
            if local_dofs is None:
                local_dofs = global_dofs
            return [np.atleast_1d(x) if x is not None else x for x in local_dofs]

        self.local_dofs_row: list[np.ndarray] = init_local_dofs(
            local_dofs_row, self.global_dofs_row
        )
        self.local_dofs_col: list[np.ndarray] = init_local_dofs(
            local_dofs_col, self.global_dofs_col
        )

        def init_active_groups(
            groups_to_blocks: list[list[int]], active_groups: list[int] | None
        ) -> list[int]:
            if active_groups is not None:
                tmp = active_groups
            else:
                tmp = list(
                    np.argsort([x[0] if len(x) else -1 for x in groups_to_blocks])
                )
            # Filter empty groups, e.g., when no fractures are present.
            return [group_idx for group_idx in tmp if len(groups_to_blocks[group_idx])]

        self.active_groups: tuple[list[int], list[int]] = (
            init_active_groups(groups_to_blocks_row, active_groups_row),
            init_active_groups(groups_to_blocks_col, active_groups_col),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.mat.shape

    def __repr__(self) -> str:
        return (
            f"BlockMatrixStorage of shape {self.shape} with {self.mat.nnz} elements "
            f"with {len(self.active_groups[0])}x{len(self.active_groups[1])} "
            "active groups"
        )

    def _correct_getitem_key(self, key) -> tuple[list[int], list[int]]:
        """User can index the matrix: `J[1, 2]`, `J[[1, 2]]`, `J[[1, 2], [3, 4]]`,
        `J[:, [1, 2]]`, `J[[1, 2], :]`. This returns the key in the format
        `J[[1, 2], [1, 2]]`.

        """
        if isinstance(key, list):
            key = key, key
        if isinstance(key, slice):
            key = key, key
        assert isinstance(key, tuple)
        assert len(key) == 2

        def correct_key(k, total):
            if isinstance(k, slice):
                start = k.start or 0
                stop = k.stop or total
                step = k.step or 1
                k = list(range(start, stop, step))
            try:
                iter(k)
            except TypeError:
                k = [k]
            return k

        groups_i, groups_j = key
        groups_i = correct_key(groups_i, total=len(self.groups_to_blocks_row))
        groups_j = correct_key(groups_j, total=len(self.groups_to_blocks_col))
        return groups_i, groups_j

    def __getitem__(self, key) -> "BlockMatrixStorage":
        groups_i, groups_j = self._correct_getitem_key(key)

        def inner(input_dofs_idx, take_groups, all_groups):
            dofs_global_idx = []
            dofs_local_idx = [None] * len(input_dofs_idx)
            offset = 0
            for group in take_groups:
                for dof_idx in all_groups[group]:
                    assert (
                        input_dofs_idx[dof_idx] is not None
                    ), f"Taking inactive row {group}"
                    dofs_global_idx.append(input_dofs_idx[dof_idx])
                    dofs_local_idx[dof_idx] = (
                        np.arange(len(input_dofs_idx[dof_idx])) + offset
                    )
                    offset += len(input_dofs_idx[dof_idx])
            if len(dofs_global_idx):
                return np.concatenate(dofs_global_idx), dofs_local_idx
            else:
                return np.array([], dtype=int), dofs_local_idx

        row_idx, local_row_idx = inner(
            self.local_dofs_row, groups_i, self.groups_to_blocks_row
        )
        col_idx, local_col_idx = inner(
            self.local_dofs_col, groups_j, self.groups_to_blocks_col
        )

        I, J = np.meshgrid(row_idx, col_idx, sparse=True, indexing="ij", copy=False)
        submat = self.mat[I, J]
        return BlockMatrixStorage(
            mat=submat,
            local_dofs_row=local_row_idx,
            local_dofs_col=local_col_idx,
            global_dofs_row=self.global_dofs_row,
            global_dofs_col=self.global_dofs_col,
            groups_to_blocks_col=self.groups_to_blocks_col,
            groups_to_blocks_row=self.groups_to_blocks_row,
            active_groups_row=tuple(groups_i),
            active_groups_col=tuple(groups_j),
            group_names_col=self.group_names_col,
            group_names_row=self.group_names_row,
        )

    def __setitem__(self, key, value):
        groups_i, groups_j = self._correct_getitem_key(key)

        if isinstance(value, BlockMatrixStorage):
            value = value.mat

        def inner(input_dofs_idx, take_groups, all_groups):
            dofs_idx = []
            for group in take_groups:
                for dof_idx in all_groups[group]:
                    assert (
                        input_dofs_idx[dof_idx] is not None
                    ), f"Taking inactive row {group}"
                    dofs_idx.append(input_dofs_idx[dof_idx])
            return np.concatenate(dofs_idx)

        row_idx = inner(self.local_dofs_row, groups_i, self.groups_to_blocks_row)
        col_idx = inner(self.local_dofs_col, groups_j, self.groups_to_blocks_col)
        I, J = np.meshgrid(row_idx, col_idx, sparse=True, indexing="ij", copy=False)
        self.mat[I, J] = value

    def copy(self) -> "BlockMatrixStorage":
        res = self.empty_container()
        res.mat = self.mat.copy()
        return res

    def empty_container(self) -> "BlockMatrixStorage":
        return BlockMatrixStorage(
            mat=scipy.sparse.csr_matrix(self.mat.shape),
            local_dofs_row=self.local_dofs_row,
            local_dofs_col=self.local_dofs_col,
            global_dofs_row=self.global_dofs_row,
            global_dofs_col=self.global_dofs_col,
            groups_to_blocks_row=self.groups_to_blocks_row,
            groups_to_blocks_col=self.groups_to_blocks_col,
            active_groups_row=self.active_groups[0],
            active_groups_col=self.active_groups[1],
            group_names_col=self.group_names_col,
            group_names_row=self.group_names_row,
        )

    def project_rhs_to_local(self, global_rhs: np.ndarray) -> np.ndarray:
        """Global rhs is the rhs arranged in the porepy model manner. This method
        permutes and restricts the global rhs to make it match the current matrix
        arrangement."""
        row_idx = [
            self.global_dofs_row[j]
            for i in self.active_groups[0]
            for j in self.groups_to_blocks_row[i]
        ]
        row_idx = np.concatenate(row_idx)
        return global_rhs[row_idx]

    def project_rhs_to_global(self, local_rhs: np.ndarray) -> np.ndarray:
        """Local rhs is the rhs arranged to match the current matrix. This method
        permutes and prolongates with zeros the local rhs to restore the global
        arrangement."""
        row_idx = np.concatenate(
            [
                self.global_dofs_row[j]
                for i in self.active_groups[0]
                for j in self.groups_to_blocks_row[i]
            ]
        )
        total_size = sum(x.size for x in self.global_dofs_col)
        result = np.zeros(total_size, dtype=local_rhs.dtype)
        result[row_idx] = local_rhs
        return result

    def project_solution_to_global(self, x: np.ndarray) -> np.ndarray:
        """The same as `project_rhs_to_global, but in the solution space."""
        col_idx = [
            self.global_dofs_col[j]
            for i in self.active_groups[1]
            for j in self.groups_to_blocks_col[i]
        ]
        col_idx = np.concatenate(col_idx)
        total_size = sum(x.size for x in self.global_dofs_col)
        result = np.zeros(total_size)
        result[col_idx] = x
        return result

    def set_zeros(
        self, group_row_idx: list[int] | int, group_col_idx: list[int] | int
    ) -> None:
        """Set the values in the given block rows and columns to zeros. Does not change
        the sparsity pattern, so this is much cheaper than doing it in the naive way."""
        group_row_idx, group_col_idx = self._correct_getitem_key(
            (group_row_idx, group_col_idx)
        )
        all_rows, all_cols = self.get_active_local_dofs(grouped=True)

        nonzero_idx = get_nonzero_indices(
            A=self.mat,
            row_indices=np.concatenate([all_rows[i] for i in group_row_idx]),
            col_indices=np.concatenate([all_cols[i] for i in group_col_idx]),
        )
        self.mat.data[nonzero_idx] = 0

    # Visualization

    def get_active_local_dofs(self, grouped=False):

        def inner(idx, groups, active_groups):
            data = []
            for active_group in active_groups:
                group_i = groups[active_group]
                group_data = []
                for i in group_i:
                    dofs = idx[i]
                    if dofs is not None:
                        group_data.append(dofs)
                if len(group_data) > 0:
                    data.append(group_data)
            return data

        row_idx = inner(
            self.local_dofs_row, self.groups_to_blocks_row, self.active_groups[0]
        )
        col_idx = inner(
            self.local_dofs_col, self.groups_to_blocks_col, self.active_groups[1]
        )
        if not grouped:
            row_idx = [y for x in row_idx for y in x]
            col_idx = [y for x in col_idx for y in x]
        else:
            row_idx = [np.concatenate(x) for x in row_idx]
            col_idx = [np.concatenate(x) for x in col_idx]
        return row_idx, col_idx

    def get_active_group_names(self):
        def inner(group_names, active_groups):
            if group_names is not None:
                names = [
                    f"{i}: {group_names[i]}" if group_names[i] != "" else str(i)
                    for i in active_groups
                ]
            else:
                names = active_groups
            return names

        row_names = inner(self.group_names_row, self.active_groups[0])
        col_names = inner(self.group_names_col, self.active_groups[1])
        return row_names, col_names

    def color_spy(
        self,
        groups=True,
        show=True,
        aspect: Literal["equal", "auto"] = "equal",
        marker=None,
        color=True,
        hatch=False,
        draw_marker=True,
        alpha=0.3,
    ):
        row_idx, col_idx = self.get_active_local_dofs(grouped=groups)
        if not groups:
            row_names = col_names = None
        else:
            row_names, col_names = self.get_active_group_names()
        color_spy(
            self.mat,
            row_idx,
            col_idx,
            row_names=row_names,
            col_names=col_names,
            show=show,
            aspect=aspect,
            marker=marker,
            alpha=alpha,
            color=color,
            hatch=hatch,
            draw_marker=draw_marker,
        )

    def matshow(
        self,
        log=True,
        show=True,
        threshold: float = 1e-30,
        aspect: Literal["equal", "auto"] = "equal",
    ):
        plot_mat(self.mat, log=log, show=show, threshold=threshold, aspect=aspect)

    def matshow_blocks(self, log=True, show=True, groups=True):
        self.matshow(log=log, show=False)
        self.color_spy(
            show=show, groups=groups, color=False, hatch=True, draw_marker=False
        )

    def plot_max(
        self,
        groups=True,
        annot=True,
    ):
        row_idx, col_idx = self.get_active_local_dofs(grouped=groups)
        data = []

        for row in row_idx:
            row_data = []
            for col in col_idx:
                I, J = np.meshgrid(row, col, sparse=True, indexing="ij", copy=False)
                submat = self.mat[I, J]
                if submat.data.size == 0:
                    row_data.append(np.nan)
                else:
                    row_data.append(abs(submat).max())
            data.append(row_data)

        if groups:
            y_tick_labels, x_tick_labels = self.get_active_group_names()
        else:
            y_tick_labels = x_tick_labels = "auto"

        ax = plt.gca()
        sns.heatmap(
            data=np.array(data),
            square=False,
            annot=annot,
            norm=matplotlib.colors.LogNorm(),
            fmt=".1e",
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            ax=ax,
            linewidths=0.01,
            linecolor="grey",
            cbar=False,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
        )

    def color_left_vector(
        self, local_rhs: np.ndarray, groups: bool = True, log: bool = True, label=None
    ):
        y_tick_labels, x_tick_labels = self.get_active_group_names()
        row_idx, col_idx = self.get_active_local_dofs(grouped=groups)
        row_names = y_tick_labels
        alpha = 0.3

        # this repeats the code of color_spy()
        row_sep = [0]
        for row in row_idx:
            row_sep.append(row[-1] + 1)
        row_sep = sorted(row_sep)

        if row_names is None:
            row_names = list(range(len(row_sep) - 1))

        ax = plt.gca()
        row_label_pos = []
        for i in range(len(row_names)):
            ystart, yend = row_sep[i : i + 2]
            row_label_pos.append(ystart + (yend - ystart) / 2)
            kwargs = {}
            kwargs["facecolor"] = f"C{i}"
            plt.axvspan(ystart - 0.5, yend - 0.5, alpha=alpha, **kwargs)
        ax.xaxis.set_ticks(row_label_pos)
        ax.set_xticklabels(row_names, rotation=45)
        if log:
            local_rhs = abs(local_rhs)
            plt.yscale("log")

        plt.plot(local_rhs, label=label)


class PreconditionerScheme:

    def make_solver(self, mat_orig: BlockMatrixStorage):
        pass

    def get_groups(self) -> list[int]:
        pass


@dataclass
class FieldSplitScheme(PreconditionerScheme):
    groups: list[int]
    solve: callable | Literal["direct", "use_invertor"] = "direct"
    invertor: callable | Literal["use_solve", "direct"] = "use_solve"
    invertor_type: Literal["physical", "algebraic", "operator", "test_vector"] = (
        "algebraic"
    )
    complement: Optional["FieldSplitScheme"] = None
    factorization_type: Literal["full", "upper", "lower"] = "upper"

    compute_cond: bool = False
    color_spy: bool = False
    only_complement: bool = False

    def __str__(self):
        res = (
            f"Groups: {self.groups}\n"
            # f"Solve: {self.solve}\n"
            # f"Invertor: {self.invertor}\n"
            f"Invertor type: {self.invertor_type}\n"
        )
        if self.complement is not None:
            complement_str = str(self.complement)
            res += complement_str
        return res

    def make_solver(self, mat_orig: BlockMatrixStorage):
        groups_0 = self.groups
        if self.complement is not None:
            groups_1 = self.complement.get_groups()
        else:
            groups_1 = []

        assert len(set(groups_0).intersection(groups_1)) == 0

        submat_00 = mat_orig[groups_0, groups_0]

        if self.color_spy:
            submat_00.color_spy()
            plt.show()
        if self.compute_cond:
            print(
                f"Blocks: {submat_00.active_groups[0]} cond: {cond(submat_00.mat):.2e}"
            )
        solve = self.solve
        invertor = self.invertor
        if solve == "use_invertor":
            solve = self.invertor
            invertor = "use_solve"
        if solve == "direct":
            submat_00_solve = inv(submat_00.mat)
        else:
            submat_00_solve = solve(mat_orig)

        if len(groups_1) == 0:
            return submat_00, submat_00_solve

        submat_10 = mat_orig[groups_1, groups_0]
        submat_01 = mat_orig[groups_0, groups_1]
        submat_11 = mat_orig[groups_1, groups_1]

        if self.invertor_type == "physical":
            submat_11.mat += invertor(mat_orig)

        elif self.invertor_type == "operator":
            submat_11.mat = invertor(mat_orig)

        elif self.invertor_type == "algebraic":
            if invertor == "use_solve":
                submat_00_inv = submat_00_solve
            elif invertor == "direct":
                submat_00_inv = inv(submat_00.mat)
            else:
                submat_00_inv = invertor(mat_orig)

            submat_11.mat -= submat_10.mat @ submat_00_inv @ submat_01.mat

        elif self.invertor_type == "test_vector":
            if invertor == "use_solve":
                submat_00_inv = submat_00_solve
            elif invertor == "direct":
                submat_00_inv = inv(submat_00.mat)
            else:
                submat_00_inv = invertor(mat_orig)

            test_vector = np.ones(submat_11.shape[0])
            diag_approx = submat_10.mat @ submat_00_inv.dot(submat_01.mat @ test_vector)
            submat_11.mat -= scipy.sparse.diags(diag_approx)

        else:
            raise ValueError(f"{self.invertor_type=}")

        complement_mat, complement_solve = self.complement.make_solver(submat_11)
        if self.only_complement:
            print("Returning only Schur complement based on", groups_1)
            return complement_mat, complement_solve

        mat_permuted = mat_orig[groups_0 + groups_1, groups_0 + groups_1]

        assert self.factorization_type in ("upper", "lower", "full")

        prec = FieldSplit(
            solve_momentum=submat_00_solve,
            solve_mass=complement_solve,
            C1=submat_10.mat,
            C2=submat_01.mat,
            groups_0=groups_0,
            groups_1=groups_1,
            factorization_type=self.factorization_type,
        )
        return mat_permuted, prec

    def get_groups(self) -> list[int]:
        groups = [g for g in self.groups]
        if self.complement is not None:
            groups.extend(self.complement.get_groups())
        return groups


@dataclass
class MultiStageScheme(PreconditionerScheme):
    stages: list[Callable[[BlockMatrixStorage], Any]]
    groups: list[int]

    def make_solver(self, mat_orig: BlockMatrixStorage):
        mat_permuted = mat_orig[self.groups]
        return mat_permuted, TwoStagePreconditioner(
            mat_permuted,
            stages=[stage(mat_permuted) for stage in self.stages],
        )

    def get_groups(self) -> list[int]:
        return self.groups


class LinearSolverWithTransformations:

    def __init__(
        self,
        inner,
        Qleft: Optional[BlockMatrixStorage] = None,
        Qright: Optional[BlockMatrixStorage] = None,
    ):
        self.Qleft: BlockMatrixStorage | None = Qleft
        self.Qright: BlockMatrixStorage | None = Qright
        self.inner = inner
        self.pc = inner.pc
        self.ksp = inner.ksp

    def solve(self, rhs):
        rhs_Q = rhs
        if self.Qleft is not None:
            rhs_Q = self.Qleft.mat @ rhs_Q

        sol_Q = self.inner.solve(rhs_Q)

        if self.Qright is not None:
            sol = self.Qright.mat @ sol_Q
        else:
            sol = sol_Q

        return sol

    def get_residuals(self):
        return self.inner.get_residuals()


def apply_ksp_scheme(
    scheme: "KSPScheme",
    bmat: BlockMatrixStorage,
    rhs_global: np.ndarray,
) -> np.ndarray:
    solver = scheme.make_solver(bmat)
    rhs_local = solver.bmat.project_rhs_to_local(rhs_global)
    sol_local = solver.solve(rhs_local)
    info = solver.ksp.getConvergedReason()

    sol_global = solver.bmat.project_solution_to_global(sol_local)

    # Verify that the original problem is solved and we did not do anything wrong.
    r_global_nrm = abs(bmat.mat @ sol_global - rhs_global).max() / abs(rhs_global).max()

    if info <= 0:
        print(f"GMRES failed, {info=}")
        if info == -9:
            sol_global[:] = np.nan
    else:
        if r_global_nrm >= 1:
            print("True residual did not decrease")

    # self._linear_solve_stats.petsc_converged_reason = info
    # self._linear_solve_stats.krylov_iters = len(gmres_.get_residuals())
    return np.atleast_1d(sol_global)


@dataclass
class KSPScheme:
    # groups: list[int]
    preconditioner: PreconditionerScheme
    ksp: Literal["gmres", "richardson"] = "gmres"
    rtol: float = 1e-10
    # max_iter: int = 60
    dtol: Optional[float] = None
    atol: Optional[float] = None
    left_transformations: Optional[
        list[Callable[[BlockMatrixStorage], BlockMatrixStorage]]
    ] = None
    right_transformations: Optional[
        list[Callable[[BlockMatrixStorage], BlockMatrixStorage]]
    ] = None
    pc_side: Literal["left", "right", "auto"] = "auto"

    def make_solver(self, mat_orig: BlockMatrixStorage):
        groups = self.get_groups()
        # assert prec_groups == self.groups
        bmat = mat_orig[groups]

        if self.left_transformations is None or len(self.left_transformations) == 0:
            Qleft = None
        else:
            Qleft = self.left_transformations[0](bmat)[groups]
            for tmp in self.left_transformations[1:]:
                tmp = tmp(bmat)[groups]
                Qleft.mat @= tmp.mat

        if self.right_transformations is None or len(self.right_transformations) == 0:
            Qright = None
        else:
            Qright = self.right_transformations[0](bmat)[groups]
            for tmp in self.right_transformations[1:]:
                tmp = tmp(bmat)[groups]
                Qright.mat @= tmp.mat

        bmat_Q = bmat
        if Qleft is not None:
            bmat_Q.mat = Qleft.mat @ bmat_Q.mat
        if Qright is not None:
            bmat_Q.mat = bmat_Q.mat @ Qright.mat

        tmp, prec = self.preconditioner.make_solver(bmat_Q)
        assert tmp.active_groups == bmat.active_groups

        if self.ksp == "gmres":
            pc_side = "right" if self.pc_side == "auto" else self.pc_side
            if self.atol is not None or self.dtol is not None:
                print("Ignoring atol and rtol!")
            solver = PetscGMRES(bmat_Q.mat, pc=prec, tol=self.rtol, pc_side=pc_side)
        elif self.ksp == "richardson":
            pc_side = "left" if self.pc_side == "auto" else self.pc_side
            if self.dtol is not None:
                print("Ignoring dtol!")
            solver = PetscRichardson(
                bmat_Q.mat, pc=prec, tol=self.rtol, atol=self.atol, pc_side=pc_side
            )
        else:
            raise ValueError(self.ksp)

        if Qleft is not None or Qright is not None:
            solver = LinearSolverWithTransformations(
                inner=solver, Qright=Qright, Qleft=Qleft
            )

        return solver

    def get_groups(self) -> list[int]:
        return self.preconditioner.get_groups()
