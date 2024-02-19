import itertools
from pathlib import Path
import time
import json
from typing import Literal, Sequence

import matplotlib as mpl
import numpy as np
import scipy
import scipy.linalg
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.sparse import bmat
from scipy.sparse.linalg import LinearOperator, gmres
from stats import LinearSolveStats

from mat_utils import PetscGMRES, condest
from stats import TimeStepStats

BURBERRY = mpl.cycler(
    color=["#A70100", "#513819", "#956226", "#B8A081", "#747674", "#0D100E"]
)

# mpl.rcParams['axes.prop_cycle'] = BURBERRY


def trim_label(label: str) -> str:
    trim = 15
    if len(label) <= trim:
        return label
    return label[:trim] + "..."


def spy(mat, show=True):
    marker = "+"
    if max(*mat.shape) > 300:
        marker = ","
    plt.spy(mat, marker=marker, markersize=4, color="black")
    if show:
        plt.show()


def plot_jacobian(model, equations=None):
    if equations is None:
        equations = list(model.equation_system.equations.values())
    try:
        equations[0]
    except IndexError:
        equations = list(equations)

    ax = plt.gca()

    eq_labels = []
    eq_labels_pos = []
    y_offset = 0
    jac_list = []
    for i, eq in enumerate(equations):
        jac = eq.value_and_jacobian(model.equation_system).jac
        jac_list.append([jac])
        eq_labels.append(trim_label(eq.name))
        eq_labels_pos.append(y_offset + jac.shape[0] / 2)
        plt.axhspan(
            y_offset - 0.5, y_offset + jac.shape[0] - 0.5, facecolor=f"C{i}", alpha=0.3
        )
        y_offset += jac.shape[0]

    jac = bmat(jac_list)
    spy(jac, show=False)
    if len(eq_labels) == 1:
        ax.set_title(eq_labels[0])
    else:
        ax.yaxis.set_ticks(eq_labels_pos)
        ax.set_yticklabels(eq_labels, rotation=0)

    labels = []
    labels_pos = []
    for i, var in enumerate(model.equation_system.variables):
        dofs = model.equation_system.dofs_of([var])
        plt.axvspan(dofs[0] - 0.5, dofs[-1] + 0.5, facecolor=f"C{i}", alpha=0.3)
        labels_pos.append(np.average(dofs))
        labels.append(trim_label(var.name))

    ax.xaxis.set_ticks(labels_pos)
    ax.set_xticklabels(labels, rotation=45, ha="left")


def plot_mat(mat, log=True):
    mat = mat.A
    if log:
        mat = np.log10(abs(mat))
    plt.matshow(mat)
    plt.colorbar()


def plot_eigs(mat, label="", logx=False):
    eigs, _ = scipy.linalg.eig(mat.A)
    if logx:
        eigs.real = abs(eigs.real)
    plt.scatter(eigs.real, eigs.imag, label=label, marker=r"$\lambda$", alpha=0.5)
    plt.xlabel(r"Re($\lambda)$")
    plt.ylabel(r"Im($\lambda$)")
    plt.legend()
    plt.grid(True)
    if logx:
        plt.xscale("log")


def solve(
    mat,
    prec=None,
    label="",
    callback_type: Literal["pr_norm", "x"] = "pr_norm",
    tol=1e-10,
):
    residuals = []
    rhs = np.ones(mat.shape[0])

    def callback(x):
        if callback_type == "pr_norm":
            residuals.append(float(x))
        else:
            residuals.append(norm(mat.dot(x) - rhs))

    if prec is not None:
        prec = LinearOperator(shape=prec.shape, matvec=prec.dot)

    restart = 50
    t0 = time.time()
    res, info = gmres(
        mat,
        rhs,
        M=prec,
        tol=tol,
        restart=restart,
        callback=callback,
        callback_type=callback_type,
        maxiter=20,
    )
    print("Solve", label, "took:", round(time.time() - t0, 2))

    inner = np.arange(len(residuals))
    if callback_type == "x":
        inner *= restart

    linestyle = "-"
    if info != 0:
        linestyle = "--"

    plt.plot(inner, residuals, label=label, marker=".", linestyle=linestyle)
    plt.yscale("log")
    plt.ylabel("pr. residual")
    plt.xlabel("gmres iter.")
    plt.grid(True)


def color_spy(block_mat, row_idx=None, col_idx=None, row_names=None, col_names=None):
    if row_idx is None:
        row_idx = list(range(block_mat.shape[0]))
    if col_idx is None:
        col_idx = list(range(block_mat.shape[1]))
    if row_names is None:
        row_names = row_idx
    if col_names is None:
        col_names = col_idx
    row_sep = [0]
    col_sep = [0]
    active_submatrices = []
    for i in row_idx:
        active_row = []
        for j in col_idx:
            submat = block_mat[i, j]
            active_row.append(submat)
            if i == row_idx[0]:
                col_sep.append(col_sep[-1] + submat.shape[1])
        row_sep.append(row_sep[-1] + submat.shape[0])
        active_submatrices.append(active_row)
    spy(bmat(active_submatrices), show=False)

    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_idx)):
        ystart, yend = row_sep[i : i + 2]
        row_label_pos.append(ystart + (yend - ystart) / 2)
        plt.axhspan(ystart - 0.5, yend - 0.5, facecolor=f"C{i}", alpha=0.3)
    ax.yaxis.set_ticks(row_label_pos)
    ax.set_yticklabels(row_names, rotation=0)

    col_label_pos = []
    for i in range(len(col_idx)):
        xstart, xend = col_sep[i : i + 2]
        col_label_pos.append(xstart + (xend - xstart) / 2)
        plt.axvspan(xstart - 0.5, xend - 0.5, facecolor=f"C{i}", alpha=0.3)
    ax.xaxis.set_ticks(col_label_pos)
    ax.set_xticklabels(col_names, rotation=0)


MARKERS = itertools.cycle(
    [
        "x",
        "+",
        # "o",
        # "v",
        # "<",
        # ">",
        # "^",
        "1",
        "2",
        "3",
        "4",
    ]
)


def solve_petsc(
    mat,
    prec=None,
    rhs=None,
    label="",
    logx_eigs=False,
    normalize_residual=False,
):
    if rhs is None:
        rhs = np.ones(mat.shape[0])
    gmres = PetscGMRES(mat, pc=prec)

    t0 = time.time()
    _ = gmres.solve(rhs)
    print("Solve", label, "took:", round(time.time() - t0, 2))
    residuals = gmres.get_residuals()
    info = gmres.ksp.getConvergedReason()
    linestyle = "-"
    if info <= 0:
        linestyle = "--"
        print("PETSc Converged Reason:", info)

    plt.gcf().set_size_inches(14, 4)

    ax = plt.subplot(1, 2, 1)
    if normalize_residual:
        residuals /= residuals[0]
    ax.plot(residuals, label=label, marker=".", linestyle=linestyle)
    ax.set_yscale("log")
    ax.set_ylabel("pr. residual")
    ax.set_xlabel("gmres iter.")
    ax.grid(True)
    if label != "":
        ax.legend()

    eigs = gmres.ksp.computeEigenvalues()
    ax = plt.subplot(1, 2, 2)
    if logx_eigs:
        eigs.real = abs(eigs.real)
    # ax.scatter(eigs.real, eigs.imag, label=label, marker="$\lambda$", alpha=0.9)
    ax.scatter(eigs.real, eigs.imag, label=label, alpha=1, s=300, marker=next(MARKERS))
    ax.set_xlabel(r"Re($\lambda)$")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.grid(True)
    if label != "":
        ax.legend()
    if logx_eigs:
        plt.xscale("log")


def get_gmres_iterations(x: Sequence[TimeStepStats]) -> list[float]:
    result = []
    for ts in x:
        for ls in ts.linear_solves:
            result.append(ls.gmres_iters)
    return result


def get_F_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        sliced_mat = model.slice_jacobian(mat)
        res.append(condest(sliced_mat.F))
    return res


def get_S_Ap_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        model.linear_system = mat, rhs
        model._prepare_solver()
        res.append(condest(model.S_Ap_fs))
    return res


def get_Bp_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        sliced_mat = model.slice_jacobian(mat)
        omega = model.slice_omega(sliced_mat)
        res.append(condest(omega.Bp))
    return res


def get_Omega_p_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        sliced_mat = model.slice_jacobian(mat)
        omega = model.slice_omega(sliced_mat)
        res.append(condest(bmat([[omega.Bp, omega.C2p], [omega.C1p, omega.Ap]])))
    return res


def get_jacobian_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        res.append(condest(mat))
    return res


def get_petsc_converged_reason(x: Sequence[TimeStepStats]) -> list[int]:
    result = []
    for ts in x:
        for ls in ts.linear_solves:
            result.append(ls.petsc_converged_reason)
    return result


def get_num_sticking_sliding_open(
    x: Sequence[TimeStepStats],
) -> tuple[list[int], list[int], list[int]]:
    sticking = []
    sliding = []
    open_ = []
    for ts in x:
        for ls in ts.linear_solves:
            st, sl, op = ls.num_sticking_sliding_open
            sticking.append(st)
            sliding.append(sl)
            open_.append(op)
    return sticking, sliding, open_


def group_intervals(arr):
    diffs = np.diff(arr)
    change_positions = np.where(diffs != 0)[0] + 1
    intervals = np.concatenate(([0], change_positions, [len(arr)]))
    return intervals


def color_converged_reason(data: Sequence[TimeStepStats], legend=True, grid=True):
    converged_reason = get_petsc_converged_reason(data)
    intervals = group_intervals(converged_reason)

    reasons_colors = {
        -9: 0,
        -5: 1,
        2: 2,
        -3: 3,
    }

    reasons_explained = {
        -3: "Diverged its",
        -9: "Nan or inf",
        -5: "Diverged breakdown",
        2: "Converged reltol",
    }

    reasons_label = set()

    for i in range(len(intervals) - 1):
        reason = converged_reason[intervals[i]]
        kwargs = {}
        if legend and reason not in reasons_label:
            reasons_label.add(reason)
            kwargs["label"] = reasons_explained[reason]
        plt.axvspan(
            intervals[i] - 0.5,
            intervals[i + 1] - 0.5,
            facecolor=f"C{reasons_colors[reason]}",
            alpha=0.3,
            **kwargs,
        )

    plt.xlim(0, len(converged_reason) - 0.5)
    if legend:
        plt.legend()

    if grid:
        plt.grid()


def load_matrix_rhs(data: Sequence[TimeStepStats], idx: int):
    flat_data: list[LinearSolveStats] = [y for x in data for y in x.linear_solves]
    load_dir = Path("../matrices")
    mat = scipy.sparse.load_npz(load_dir / flat_data[idx].matrix_id)
    rhs = np.load(load_dir / flat_data[idx].rhs_id)
    return mat, rhs


def load_data(path) -> Sequence[TimeStepStats]:
    with open(path, "r") as f:
        payload = json.load(f)
    return [TimeStepStats.from_json(x) for x in payload]
