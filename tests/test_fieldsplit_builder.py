from FTHM_Solver import full_petsc_solver
import FTHM_Solver
from FTHM_Solver.block_matrix import BlockMatrixStorage
from petsc4py import PETSc
import numpy as np
import pytest
import warnings

import scipy.sparse as sps

import io
import sys


def capture_stdout(method, *args) -> str:
    buffer = io.StringIO()
    original_stdout = sys.stdout  # Save the original stdout
    try:
        sys.stdout = buffer  # Redirect stdout to the buffer
        method(*args)  # Call the method whose output you want to capture
    finally:
        sys.stdout = original_stdout  # Restore the original stdout
    return buffer.getvalue()  # Get the captured output as a string


@pytest.fixture
def block_matrix():
    A = sps.csr_matrix(np.random.rand(10, 10))

    b = np.random.rand(10)

    global_row = [
        np.array([0, 1]),
        np.array([2]),
        np.array([3, 4, 5]),
        np.array([6, 7]),
        np.array([8, 9]),
    ]
    global_col = [
        np.array([0, 1]),
        np.array([2]),
        np.array([3, 4, 5]),
        np.array([6, 7]),
        np.array([8, 9]),
    ]

    groups_to_blocks = [[0, 1], [2], [3, 4]]

    block_A = FTHM_Solver.BlockMatrixStorage(
        A, global_row, global_col, groups_to_blocks, groups_to_blocks
    )
    return block_A


def compare_options(expected):
    petsc_opex = PETSc.Options()
    for key, value in expected.items():
        assert petsc_opex[key] == str(value)


def test_ksp_setup(block_matrix):
    # Set up a solver with some non-default options. Check that they are passed to
    # PETSc.
    opts = {"ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_max_it": 100}

    ksp_scheme = FTHM_Solver.PetscKSPScheme(petsc_options=opts)
    ksp_scheme.make_solver(block_matrix)

    compare_options(opts)


def _ksp(block_matrix):
    ksp = PETSc.KSP().create()
    ksp.setOperators(block_matrix)
    ksp.setFromOptions()

    return ksp


def test_preconditioner_no_fieldsplit(block_matrix):
    # Set up a solver with a preconditioner, but no fieldsplit. Check that the
    # preconditioner is passed to PETSc.

    ksp = _ksp(FTHM_Solver.csr_to_petsc(block_matrix.mat))
    pc = ksp.getPC()

    precond = FTHM_Solver.PetscFieldSplitScheme(
        groups=[0],
        elim_options={
            "pc_type": "ilu",
        },
        fieldsplit_options={
            "pc_fieldsplit_schur_precondition": "selfp",
        },
        keep_options={
            "pc_type": "ilu",
        },
    )
    # Unwind the nested configuration. This will also set up the preconditioner, hence
    # there is no need to call pc.setUp().
    precond.configure(block_matrix, pc)

    # With no complement set, the preconditioner should be the same as the eliminator.
    assert pc.getType() == "ilu"
