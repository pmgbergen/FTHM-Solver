import numpy as np
from plot_utils import spy, plot_mat
from matplotlib import pyplot as plt


def color_spy(mat, row_idx, col_idx):

    spy(mat, show=False)

    row_sep = [0]
    row_names = []
    for i, row in enumerate(row_idx):
        if row is not None:
            row_sep.append(row[-1] + 1)
            row_names.append(i)
    col_sep = [0]
    col_names = []
    for j, col in enumerate(col_idx):
        if col is not None:
            col_sep.append(col[-1] + 1)
            col_names.append(j)
    
    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_names)):
        ystart, yend = row_sep[i: i + 2]
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
        return BlockMatrixStorage(
            mat=submat,
            row_idx=local_row_idx,
            col_idx=local_col_idx,
            groups_col=self.groups_col,
            groups_row=self.groups_row,
        )

    def color_spy(self):
        color_spy(self.mat, self.local_row_idx, self.local_col_idx)

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
        res = np.zeros_like(rhs)
        res[col_idx] = rhs
        return res
