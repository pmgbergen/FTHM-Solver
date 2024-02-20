from dataclasses import dataclass
from typing import Literal, Optional

import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from mat_utils import OmegaInv, inv, cond
from plot_utils import plot_mat, spy


def color_spy(mat, row_idx, col_idx, row_names=None, col_names=None):

    spy(mat, show=False)

    row_sep = [0]
    for row in row_idx:
        # if row is not None:
        row_sep.append(row[-1] + 1)
    row_sep = sorted(row_sep)

    col_sep = [0]
    for col in col_idx:
        # if col is not None:
        col_sep.append(col[-1] + 1)
    col_sep = sorted(col_sep)

    if row_names is None:
        row_names = list(range(len(row_sep) - 1))
    if col_names is None:
        col_names = list(range(len(col_sep) - 1))

    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_names)):
        ystart, yend = row_sep[i : i + 2]
        row_label_pos.append(ystart + (yend - ystart) / 2)
        plt.axhspan(ystart - 0.5, yend - 0.5, facecolor=f"C{i}", alpha=0.3)
    ax.yaxis.set_ticks(row_label_pos)
    ax.set_yticklabels(row_names, rotation=0)

    col_label_pos = []
    for i in range(len(col_names)):
        xstart, xend = col_sep[i : i + 2]
        col_label_pos.append(xstart + (xend - xstart) / 2)
        plt.axvspan(xstart - 0.5, xend - 0.5, facecolor=f"C{i}", alpha=0.3)
    ax.xaxis.set_ticks(col_label_pos)
    ax.set_xticklabels(col_names, rotation=0)


class BlockMatrixStorage:
    def __init__(
        self,
        mat,
        row_idx,
        col_idx,
        groups_row,
        groups_col,
        active_groups_col=None,
        active_groups_row=None,
    ):
        self.mat = mat
        self.local_row_idx = [np.atleast_1d(x) if x is not None else x for x in row_idx]
        self.local_col_idx = [np.atleast_1d(x) if x is not None else x for x in col_idx]
        self.groups_row = groups_row
        self.groups_col = groups_col

        if active_groups_row is None:
            active_groups_row = tuple(np.argsort([x[0] for x in groups_row]))
        if active_groups_col is None:
            active_groups_col = tuple(np.argsort([x[0] for x in groups_col]))
        self.active_groups = active_groups_row, active_groups_col

    @property
    def shape(self):
        return self.mat.shape

    def __repr__(self) -> str:
        return f"BlockMatrixStorage of shape {self.shape} with {self.mat.nnz} elements"

    def __getitem__(self, key):
        if isinstance(key, list):
            key = key, key
        if isinstance(key, slice):
            key = key, key
        assert isinstance(key, tuple)
        assert len(key) == 2

        def correct_key(k):
            if isinstance(k, slice):
                start = k.start or 0
                stop = k.stop or len(self.groups_row)
                step = k.step or 1
                k = list(range(start, stop, step))
            try:
                iter(k)
            except TypeError:
                k = [k]
            return k

        groups_i, groups_j = key
        groups_i = correct_key(groups_i)
        groups_j = correct_key(groups_j)

        def inner(input_dofs_idx, take_groups, all_groups):
            dofs_idx = []
            dofs_local_idx = [None] * len(input_dofs_idx)
            offset = 0
            for group in take_groups:
                for dof_idx in all_groups[group]:
                    assert (
                        input_dofs_idx[dof_idx] is not None
                    ), f"Taking inactive row {group}"
                    dofs_idx.append(input_dofs_idx[dof_idx])
                    dofs_local_idx[dof_idx] = (
                        np.arange(len(input_dofs_idx[dof_idx])) + offset
                    )
                    offset += len(input_dofs_idx[dof_idx])
            return np.concatenate(dofs_idx), dofs_local_idx

        row_idx, local_row_idx = inner(self.local_row_idx, groups_i, self.groups_row)
        col_idx, local_col_idx = inner(self.local_col_idx, groups_j, self.groups_col)

        I, J = np.meshgrid(row_idx, col_idx, sparse=True, indexing="ij")
        submat = self.mat[I, J]
        return BlockMatrixStorage(
            mat=submat,
            row_idx=local_row_idx,
            col_idx=local_col_idx,
            groups_col=self.groups_col,
            groups_row=self.groups_row,
            active_groups_row=tuple(groups_i),
            active_groups_col=tuple(groups_j),
        )

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

    def color_spy(self, groups=True):
        row_idx, col_idx = self.get_active_local_dofs(grouped=groups)
        if not groups:
            row_names = None
            col_names = None
        else:
            row_names = self.active_groups[0]
            col_names = self.active_groups[1]

        color_spy(self.mat, row_idx, col_idx, row_names=row_names, col_names=col_names)

    def slice_domain(self, i, j):
        row_idx = [x for x in self.local_row_idx if x is not None][i]
        col_idx = [x for x in self.local_col_idx if x is not None][j]
        I, J = np.meshgrid(row_idx, col_idx, sparse=True, indexing="ij")
        return self.mat[I, J]

    def matshow(self, log=True):
        plot_mat(self.mat, log=log)

    def copy(self):
        return BlockMatrixStorage(
            mat=self.mat.copy(),
            row_idx=self.local_row_idx,
            col_idx=self.local_col_idx,
            groups_row=self.groups_row,
            groups_col=self.groups_col,
            active_groups_row=self.active_groups[0],
            active_groups_col=self.active_groups[1],
        )

    def local_rhs(self, rhs):
        row_idx = [x for x in self.local_row_idx if x is not None]
        row_idx = np.concatenate(row_idx)
        perm = np.zeros_like(row_idx)
        perm[row_idx] = np.arange(row_idx.size)
        return rhs[perm]

    def reverse_transform_solution(self, x):
        col_idx = [x for x in self.local_col_idx if x is not None]
        col_idx = np.concatenate(col_idx)
        return x[col_idx]

    def plot_max(self, group=True):
        row_idx, col_idx = self.get_active_local_dofs(grouped=group)
        data = []

        for row in row_idx:
            row_data = []
            for col in col_idx:
                I, J = np.meshgrid(row, col, sparse=True, indexing="ij")
                submat = self.mat[I, J]
                if submat.data.size == 0:
                    row_data.append(np.nan)
                else:
                    row_data.append(abs(submat).max())
            data.append(row_data)

        if group:
            y_tick_labels = self.active_groups[0]
            x_tick_labels = self.active_groups[1]
        else:
            y_tick_labels = x_tick_labels = "auto"

        plt.figure()
        ax = sns.heatmap(
            data=np.array(data),
            square=False,
            annot=True,
            norm=matplotlib.colors.LogNorm(),
            fmt=".1e",
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            # robust=True,
            linewidths=0.01,
            linecolor="grey",
            cbar=False,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
        )
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # ax.set_title(key)


@dataclass
class SolveSchema:
    groups: list[int]
    solve: callable | Literal["direct"] = "direct"
    invertor: callable | Literal["use_solve", "direct"] = "use_solve"
    invertor_type: Literal["physical", "algebraic"] = "algebraic"
    complement: Optional["SolveSchema"] = None

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

    assert set(groups_0).intersection

    submat_00 = mat_orig[groups_0, groups_0]

    if schema.color_spy:
        submat_00.color_spy()
        plt.show()
    if schema.compute_cond:
        print(f"Blocks: {submat_00.active_groups[0]} cond: {cond(submat_00.mat):.2e}")
    solve = schema.solve
    if solve == "direct":
        submat_00_solve = inv(submat_00.mat)
    else:
        submat_00_solve = solve(submat_00)

    if len(groups_1) == 0:
        return submat_00, submat_00_solve

    submat_10 = mat_orig[groups_1, groups_0]
    submat_01 = mat_orig[groups_0, groups_1]
    submat_11 = mat_orig[groups_1, groups_1]

    if schema.invertor_type == "physical":
        submat_11.mat += schema.invertor()

    elif schema.invertor_type == "algebraic":
        if schema.invertor == "use_solve":
            submat_00_inv = submat_00_solve
        elif schema.invertor == "direct":
            submat_00_inv = inv(submat_00.mat)
        else:
            submat_00_inv = schema.invertor(submat_00)
        submat_11.mat -= submat_10.mat @ submat_00_inv @ submat_01.mat

    else:
        raise ValueError(f"{schema.invertor_type=}")

    complement_mat, complement_solve = make_solver(
        schema=schema.complement, mat_orig=submat_11
    )
    if schema.only_complement:
        return complement_mat, complement_solve

    mat_permuted = mat_orig[groups_0 + groups_1, groups_0 + groups_1]
    prec = OmegaInv(
        solve_momentum=submat_00_solve,
        solve_mass=complement_solve,
        C1=submat_10.mat,
        C2=submat_01.mat,
    )
    return mat_permuted, prec
