from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from matplotlib import pyplot as plt

from mat_utils import OmegaInv, inv, cond
from plot_utils import plot_mat, spy


def color_spy(mat, row_idx, col_idx, row_names=None, col_names=None):

    spy(mat, show=False)

    row_sep = [0]
    for row in row_idx:
        if row is not None:
            row_sep.append(row[-1] + 1)
    row_sep = sorted(row_sep)

    col_sep = [0]
    for col in col_idx:
        if col is not None:
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
    def __init__(self, mat, row_idx, col_idx, groups_col, groups_row):
        self.mat = mat
        self.local_row_idx = [np.atleast_1d(x) if x is not None else x for x in row_idx]
        self.local_col_idx = [np.atleast_1d(x) if x is not None else x for x in col_idx]
        self.groups_col = groups_col
        self.groups_row = groups_row

        self.active_groups = tuple(range(len(groups_col))), tuple(
            range(len(groups_row))
        )

    @property
    def shape(self):
        return self.mat.shape

    def __repr__(self) -> str:
        return f"BlockMatrixStorage of shape {self.shape} with {self.mat.nnz} elements"

    def __getitem__(self, key):
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

        i, j = key
        i = correct_key(i)
        j = correct_key(j)

        row_idx = []
        local_row_idx = [None] * len(self.local_row_idx)
        offset = 0
        for i1 in i:
            for b_row_idx in self.groups_row[i1]:
                assert (
                    self.local_row_idx[b_row_idx] is not None
                ), f"Taking inactive row {i1}"
                row_idx.append(self.local_row_idx[b_row_idx])
                local_row_idx[b_row_idx] = (
                    np.arange(len(self.local_row_idx[b_row_idx])) + offset
                )
                offset += len(self.local_row_idx[b_row_idx])
        col_idx = []
        local_col_idx = [None] * len(self.local_col_idx)
        offset = 0
        for j1 in j:
            for b_col_idx in self.groups_col[j1]:
                assert (
                    self.local_col_idx[b_col_idx] is not None
                ), f"Taking inactive col {j1}"
                col_idx.append(self.local_col_idx[b_col_idx])
                local_col_idx[b_col_idx] = (
                    np.arange(len(self.local_col_idx[b_col_idx])) + offset
                )
                offset += len(self.local_col_idx[b_col_idx])

        row_idx = np.concatenate(row_idx)
        col_idx = np.concatenate(col_idx)
        I, J = np.meshgrid(row_idx, col_idx, sparse=True, indexing="ij")
        submat = self.mat[I, J]
        res = BlockMatrixStorage(
            mat=submat,
            row_idx=local_row_idx,
            col_idx=local_col_idx,
            groups_col=self.groups_col,
            groups_row=self.groups_row,
        )
        res.active_groups = tuple(i), tuple(j)
        return res

    def color_spy(self, groups=True):
        if not groups:
            row_idx = self.local_row_idx
            col_idx = self.local_col_idx
            row_names = None
            col_names = None
        else:

            def cycle(groups, local_idx):
                idx = []
                for i in groups:
                    indices = [local_idx[j] for j in i if local_idx[j] is not None]
                    if len(indices) > 0:
                        idx.append(np.concatenate(indices))
                return idx

            row_idx = cycle(self.groups_row, self.local_row_idx)
            col_idx = cycle(self.groups_col, self.local_col_idx)
            row_names = self.active_groups[0]
            col_names = self.active_groups[1]

        color_spy(self.mat, row_idx, col_idx, row_names=row_names, col_names=col_names)

    def matshow(self, log=True):
        plot_mat(self.mat, log=log)

    def copy(self):
        return BlockMatrixStorage(
            mat=self.mat.copy(),
            row_idx=self.local_row_idx,
            col_idx=self.local_col_idx,
            groups_row=self.groups_row,
            groups_col=self.groups_col,
        )

    def local_rhs(self, rhs):
        col_idx = [x for x in self.local_col_idx if x is not None]
        col_idx = np.concatenate(col_idx)
        perm = np.zeros_like(col_idx)
        perm[col_idx] = np.arange(col_idx.size)
        return rhs[perm]


@dataclass
class SolveSchema:
    groups: list[int]
    solve: callable | Literal['direct'] = 'direct'
    invertor: callable | Literal["use_solve", 'direct'] = "use_solve"
    invertor_type: Literal["physical", "algebraic"] = "algebraic"
    complement: Optional["SolveSchema"] = None
    compute_cond: bool = False
    color_spy: bool = False


def get_complement_groups(schema: SolveSchema):
    res = []
    if schema.complement is not None:
        res.extend(schema.complement.groups)
        res.extend(get_complement_groups(schema=schema.complement))
    return res


def make_solver(schema: SolveSchema, mat_orig: BlockMatrixStorage):
    groups_0 = schema.groups
    groups_1 = get_complement_groups(schema)

    submat_00 = mat_orig[groups_0, groups_0]

    if schema.color_spy:
        submat_00.color_spy()
        plt.show()
    if schema.compute_cond:
        print(f'Blocks: {submat_00.active_groups[0]} cond: {cond(submat_00.mat):.2e}')
    solve = schema.solve
    if solve == 'direct':
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
        elif schema.invertor == 'direct':
            submat_00_inv = inv(submat_00.mat)
        else:
            submat_00_inv = schema.invertor(submat_00)
        submat_11.mat -= submat_10.mat @ submat_00_inv @ submat_01.mat

    else:
        raise ValueError(f"{schema.invertor_type=}")

    complement_mat, complement_solve = make_solver(
        schema=schema.complement, mat_orig=submat_11
    )

    mat_permuted = mat_orig[groups_0 + groups_1, groups_0 + groups_1]
    prec = OmegaInv(
        solve_momentum=submat_00_solve,
        solve_mass=complement_solve,
        C1=submat_10.mat,
        C2=submat_01.mat,
    )
    return mat_permuted, prec
