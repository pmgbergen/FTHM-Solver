from dataclasses import dataclass
from typing import Literal, Optional, Sequence
import itertools

import seaborn as sns
import numpy as np
import matplotlib
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix
from matplotlib import pyplot as plt

from mat_utils import FieldSplit, inv, cond
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
            kwargs['edgecolor'] = 'black'
        plt.axhspan(ystart - 0.5, yend - 0.5, alpha=alpha, **kwargs)
    ax.yaxis.set_ticks(row_label_pos)
    ax.set_yticklabels(row_names, rotation=0)

    hatch_types = itertools.cycle(["|", "-"])

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


class BlockMatrixStorage:
    def __init__(
        self,
        mat: spmatrix,
        global_row_idx: Sequence[np.ndarray],
        global_col_idx: Sequence[np.ndarray],
        groups_row: list[list[int]],
        groups_col: list[list[int]],
        local_row_idx: Sequence[np.ndarray] = None,
        local_col_idx: Sequence[np.ndarray] = None,
        active_groups_col=None,
        active_groups_row=None,
        group_row_names: Sequence[str] = None,
        group_col_names: Sequence[str] = None,
    ):
        self.mat: spmatrix = mat
        self.groups_row: list[list[int]] = groups_row
        self.groups_col: list[list[int]] = groups_col
        self.groups_row_names: Optional[Sequence[str]] = group_row_names
        self.groups_col_names: Optional[Sequence[str]] = group_col_names

        if local_row_idx is None:
            local_row_idx = global_row_idx
        if local_col_idx is None:
            local_col_idx = global_col_idx
        self.local_row_idx: list[np.ndarray] = [
            np.atleast_1d(x) if x is not None else x for x in local_row_idx
        ]
        self.local_col_idx: list[np.ndarray] = [
            np.atleast_1d(x) if x is not None else x for x in local_col_idx
        ]
        self.global_row_idx: list[np.ndarray] = [
            np.atleast_1d(x) for x in global_row_idx
        ]
        self.global_col_idx: list[np.ndarray] = [
            np.atleast_1d(x) for x in global_col_idx
        ]

        if active_groups_row is None:
            # This does not work if groups_row include [] for inactive groups
            active_groups_row = list(np.argsort([x[0] for x in groups_row]))
        if active_groups_col is None:
            active_groups_col = list(np.argsort([x[0] for x in groups_col]))
        self.active_groups = active_groups_row, active_groups_col

    @property
    def shape(self) -> tuple[int, int]:
        return self.mat.shape

    @property
    def active_subgroups(self):
        return (
            [i for i, x in enumerate(self.local_row_idx) if x is not None],
            [j for j, x in enumerate(self.local_col_idx) if x is not None],
        )

    def __repr__(self) -> str:
        return (
            f"BlockMatrixStorage of shape {self.shape} with {self.mat.nnz} elements "
            f"with {len(self.active_groups[0])}x{len(self.active_groups[1])} "
            "active groups"
        )

    def _correct_getitem_key(self, key) -> tuple[list[int], list[int]]:
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
        groups_i = correct_key(groups_i, total=len(self.groups_row))
        groups_j = correct_key(groups_j, total=len(self.groups_col))
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
            return np.concatenate(dofs_global_idx), dofs_local_idx

        row_idx, local_row_idx = inner(self.local_row_idx, groups_i, self.groups_row)
        col_idx, local_col_idx = inner(self.local_col_idx, groups_j, self.groups_col)

        I, J = np.meshgrid(row_idx, col_idx, sparse=True, indexing="ij", copy=False)
        submat = self.mat[I, J]
        return BlockMatrixStorage(
            mat=submat,
            local_row_idx=local_row_idx,
            local_col_idx=local_col_idx,
            global_row_idx=self.global_row_idx,
            global_col_idx=self.global_col_idx,
            groups_col=self.groups_col,
            groups_row=self.groups_row,
            active_groups_row=tuple(groups_i),
            active_groups_col=tuple(groups_j),
            group_col_names=self.groups_col_names,
            group_row_names=self.groups_row_names,
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

        row_idx = inner(self.local_row_idx, groups_i, self.groups_row)
        col_idx = inner(self.local_col_idx, groups_j, self.groups_col)
        I, J = np.meshgrid(row_idx, col_idx, sparse=True, indexing="ij", copy=False)
        self.mat[I, J] = value

    def group_shape(self, i, j=None) -> tuple[int, int]:
        groups_i, groups_j = self._correct_getitem_key((i, j) if j is not None else i)

        def inner(input_dofs_idx, take_groups, all_groups):
            num_dofs = 0
            for group in take_groups:
                for dof_idx in all_groups[group]:
                    if (dofs := input_dofs_idx[dof_idx]) is not None:
                        num_dofs += dofs.size
            return num_dofs

        row_shape = inner(self.local_row_idx, groups_i, self.groups_row)
        col_shape = inner(self.local_col_idx, groups_j, self.groups_col)
        return row_shape, col_shape

    def slice_domain(self, i, j) -> spmatrix:
        active_subgroups_i, active_subgroups_j = self.active_subgroups
        row_idx = self.local_row_idx[active_subgroups_i[i]]
        col_idx = self.local_col_idx[active_subgroups_j[j]]
        I, J = np.meshgrid(row_idx, col_idx, sparse=True, indexing="ij", copy=False)
        return self.mat[I, J]

    def block_diag_inv(self) -> spmatrix:
        active_idx = [x for x in self.local_row_idx if x is not None]
        bmats = []
        for active in active_idx:
            I, J = np.meshgrid(active, active, sparse=True, indexing="ij", copy=False)
            bmats.append(inv(self.mat[I, J]))
        return scipy.sparse.block_diag(bmats, format="csr")

    def copy(self) -> "BlockMatrixStorage":
        return BlockMatrixStorage(
            mat=self.mat.copy(),
            local_row_idx=self.local_row_idx,
            local_col_idx=self.local_col_idx,
            global_row_idx=self.global_row_idx,
            global_col_idx=self.global_col_idx,
            groups_row=self.groups_row,
            groups_col=self.groups_col,
            active_groups_row=self.active_groups[0],
            active_groups_col=self.active_groups[1],
            group_col_names=self.groups_col_names,
            group_row_names=self.groups_row_names,
        )

    def empty_container(self) -> "BlockMatrixStorage":
        return BlockMatrixStorage(
            mat=scipy.sparse.csr_matrix(self.mat.shape),
            local_row_idx=self.local_row_idx,
            local_col_idx=self.local_col_idx,
            global_row_idx=self.global_row_idx,
            global_col_idx=self.global_col_idx,
            groups_row=self.groups_row,
            groups_col=self.groups_col,
            active_groups_row=self.active_groups[0],
            active_groups_col=self.active_groups[1],
            group_col_names=self.groups_col_names,
            group_row_names=self.groups_row_names,
        )

    def local_rhs(self, rhs: np.ndarray) -> np.ndarray:
        row_idx = [
            self.global_row_idx[j]
            for i in self.active_groups[0]
            for j in self.groups_row[i]
        ]
        row_idx = np.concatenate(row_idx)
        return rhs[row_idx]

    def reverse_transform_solution(self, x: np.ndarray) -> np.ndarray:
        col_idx = [
            self.global_col_idx[j]
            for i in self.active_groups[1]
            for j in self.groups_col[i]
        ]
        col_idx = np.concatenate(col_idx)
        total_size = sum(x.size for x in self.global_col_idx)
        result = np.zeros(total_size)
        result[col_idx] = x
        return result

    def get_global_indices(
        self,
        local_indices,
        group: tuple[int, int],
        subgroup: tuple[int, int] = (0, 0),
    ):
        assert group[0] in self.active_groups[0]
        assert group[1] in self.active_groups[1]

        group_row = self.groups_row[group[0]]
        group_col = self.groups_col[group[1]]
        subgroup_row = group_row[subgroup[0]]
        subgroup_col = group_col[subgroup[1]]
        offset_i = self.local_row_idx[subgroup_row][0]
        offset_j = self.local_col_idx[subgroup_col][0]

        global_i = np.array(local_indices[0]) + offset_i
        global_j = np.array(local_indices[1]) + offset_j
        return global_i, global_j

    def slice_submatrix(
        self, local_indices, group: tuple[int, int], subgroup: tuple[int, int] = (0, 0)
    ) -> spmatrix:
        global_i, global_j = self.get_global_indices(
            local_indices=local_indices, group=group, subgroup=subgroup
        )
        global_i, global_j = np.meshgrid(
            global_i, global_j, sparse=True, indexing="ij", copy=False
        )
        return self.mat[global_i, global_j]

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

        row_idx = inner(self.local_row_idx, self.groups_row, self.active_groups[0])
        col_idx = inner(self.local_col_idx, self.groups_col, self.active_groups[1])
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

        row_names = inner(self.groups_row_names, self.active_groups[0])
        col_names = inner(self.groups_col_names, self.active_groups[1])
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

    def matshow(self, log=True, show=True):
        plot_mat(self.mat, log=log, show=show)

    def matshow_blocks(self, log=True, show=True, groups=True):
        self.matshow(log=log, show=False)
        self.color_spy(
            show=show, groups=groups, color=False, hatch=True, draw_marker=False
        )

    def plot_max(self, group=True):
        row_idx, col_idx = self.get_active_local_dofs(grouped=group)
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

        if group:
            y_tick_labels, x_tick_labels = self.get_active_group_names()
        else:
            y_tick_labels = x_tick_labels = "auto"

        ax = plt.gca()
        sns.heatmap(
            data=np.array(data),
            square=False,
            annot=True,
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


@dataclass
class SolveSchema:
    groups: list[int]
    solve: callable | Literal["direct", "use_invertor"] = "direct"
    invertor: callable | Literal["use_solve", "direct"] = "use_solve"
    invertor_type: Literal["physical", "algebraic", "operator", "test_vector"] = (
        "algebraic"
    )
    complement: Optional["SolveSchema"] = None

    transform_nondiagonal_blocks: callable = lambda mat10, mat01: (mat10, mat01)

    compute_cond: bool = False
    color_spy: bool = False
    only_complement: bool = False


def get_complement_groups(schema: SolveSchema):
    res = []
    if schema.complement is not None:
        res.extend(schema.complement.groups)
        res.extend(get_complement_groups(schema=schema.complement))
    return res


def make_solver(schema: SolveSchema, mat_orig: BlockMatrixStorage):
    groups_0 = schema.groups
    groups_1 = get_complement_groups(schema)

    assert len(set(groups_0).intersection(groups_1)) == 0

    submat_00 = mat_orig[groups_0, groups_0]

    if schema.color_spy:
        submat_00.color_spy()
        plt.show()
    if schema.compute_cond:
        print(f"Blocks: {submat_00.active_groups[0]} cond: {cond(submat_00.mat):.2e}")
    solve = schema.solve
    invertor = schema.invertor
    if solve == "use_invertor":
        solve = schema.invertor
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

    if schema.invertor_type == "physical":
        submat_11.mat += invertor(mat_orig)

    elif schema.invertor_type == "operator":
        submat_11.mat = invertor(mat_orig)

    elif schema.invertor_type == "algebraic":
        if invertor == "use_solve":
            submat_00_inv = submat_00_solve
        elif invertor == "direct":
            submat_00_inv = inv(submat_00.mat)
        else:
            submat_00_inv = invertor(mat_orig)
        submat_10_m, submat_01_m = schema.transform_nondiagonal_blocks(
            submat_10, submat_01
        )
        submat_11.mat -= submat_10_m.mat @ submat_00_inv @ submat_01_m.mat

    elif schema.invertor_type == "test_vector":
        if invertor == "use_solve":
            submat_00_inv = submat_00_solve
        elif invertor == "direct":
            submat_00_inv = inv(submat_00.mat)
        else:
            submat_00_inv = invertor(mat_orig)

        test_vector = np.ones(submat_11.shape[0])
        diag_approx = submat_10.mat @ submat_00_inv.dot(submat_01.mat @ test_vector)
        submat_11.mat -= scipy.sparse.diags(diag_approx)

    # elif schme.invertor_type == 'block_test_vector'

    else:
        raise ValueError(f"{schema.invertor_type=}")

    complement_mat, complement_solve = make_solver(
        schema=schema.complement, mat_orig=submat_11
    )
    if schema.only_complement:
        print("Returning only Schur complement based on", groups_1)
        return complement_mat, complement_solve

    mat_permuted = mat_orig[groups_0 + groups_1, groups_0 + groups_1]
    prec = FieldSplit(
        solve_momentum=submat_00_solve,
        solve_mass=complement_solve,
        C1=submat_10.mat,
        C2=submat_01.mat,
    )
    return mat_permuted, prec


def make_complement(schema: SolveSchema, mat_orig: BlockMatrixStorage):
    groups_0 = schema.groups
    groups_1 = get_complement_groups(schema)

    assert len(set(groups_0).intersection(groups_1)) == 0

    submat_00 = mat_orig[groups_0, groups_0]

    solve = schema.solve
    invertor = schema.invertor
    if solve == "direct":
        solve = lambda bmat: inv(submat_00.mat)

    if len(groups_1) == 0:
        return submat_00

    submat_10 = mat_orig[groups_1, groups_0]
    submat_01 = mat_orig[groups_0, groups_1]
    submat_11 = mat_orig[groups_1, groups_1]

    if schema.invertor_type == "physical":
        submat_11.mat += invertor(mat_orig)

    elif schema.invertor_type == "operator":
        submat_11.mat = invertor(mat_orig)

    elif schema.invertor_type == "algebraic":
        if invertor == "use_solve":
            submat_00_inv = solve(mat_orig)
        elif invertor == "direct":
            submat_00_inv = inv(submat_00.mat)
        else:
            submat_00_inv = invertor(mat_orig)
        submat_10_m, submat_01_m = schema.transform_nondiagonal_blocks(
            submat_10, submat_01
        )
        submat_11.mat -= submat_10_m.mat @ submat_00_inv @ submat_01_m.mat

    else:
        raise ValueError(f"{schema.invertor_type=}")

    complement_mat = make_complement(schema=schema.complement, mat_orig=submat_11)
    print("Returning only Schur complement based on", groups_1)
    return complement_mat
