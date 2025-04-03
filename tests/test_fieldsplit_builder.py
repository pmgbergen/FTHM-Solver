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


def _dofs_from_group_ind(bm, group_ind):
    return np.concatenate(
        [
            # Use the local dofs here to get the correct ordering also for submatrices.
            bm.local_dofs_col[block_ind]
            for gi in group_ind
            for block_ind in bm.groups_to_blocks_col[gi]
        ]
    )


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


def test_preconditioner_one_level_fieldsplit(block_matrix):
    """One level fieldsplit preconditioner."""
    ksp = _ksp(FTHM_Solver.csr_to_petsc(block_matrix.mat))
    pc = ksp.getPC()

    elim_groups = [0]
    keep_groups = [1, 2]

    # Hard coded tags for the groups. We should really test this.
    elim_tag = "0"
    keep_tag = "1-2"

    elim_pc_type = "ilu"
    keep_pc_type = "sor"

    precond = FTHM_Solver.PetscFieldSplitScheme(
        groups=elim_groups,
        elim_options={
            "pc_type": elim_pc_type,
        },
        fieldsplit_options={
            "pc_fieldsplit_schur_precondition": "selfp",
        },
        complement=FTHM_Solver.PetscFieldSplitScheme(
            groups=keep_groups,
            elim_options={
                "pc_type": keep_pc_type,
            },
        ),
    )
    # Unwind the nested configuration. This will also set up the preconditioner, hence
    # there is no need to call pc.setUp().
    precond.configure(block_matrix, pc)

    # With no complement set, the preconditioner should be the same as the eliminator.
    assert pc.getType() == "fieldsplit"

    elim_pc = pc.getFieldSplitSubKSP()[0].getPC()
    assert elim_pc.getType() == elim_pc_type

    keep_pc = pc.getFieldSplitSubKSP()[1].getPC()
    assert keep_pc.getType() == keep_pc_type

    elim_IS = pc.getFieldSplitSubIS(elim_tag)
    keep_IS = pc.getFieldSplitSubIS(keep_tag)

    known_elim_ind = _dofs_from_group_ind(block_matrix, elim_groups)
    known_keep_ind = _dofs_from_group_ind(block_matrix, keep_groups)
    np.testing.assert_array_equal(elim_IS, known_elim_ind)
    np.testing.assert_array_equal(keep_IS, known_keep_ind)


def test_preconditioner_two_level_fieldsplit(block_matrix):
    """Two level fieldsplit preconditioner."""
    ksp = _ksp(FTHM_Solver.csr_to_petsc(block_matrix.mat))
    pc = ksp.getPC()

    # In the first stage we eliminate group 0 and keep groups 1 and 2.
    elim_group_0 = [0]
    keep_groups_0 = [1, 2]
    # In the second stage we eliminate group 1 and keep group 2.
    elim_group_1 = [1]
    keep_groups_1 = [2]

    # Hard coded tags for the groups. We should really test this.
    elim_tag_0 = "0"
    keep_tag_0 = "1-2"
    elim_tag_1 = "1"
    keep_tag_1 = "2"

    # Use different preconditioners for the stages.
    elim_pc_type_0 = "ilu"
    elim_pc_type_1 = "sor"
    keep_pc_type_1 = "lu"
    # Also set a ksp solver for the eliminated blocks.
    ksp_0 = "bicg"
    ksp_1 = "cg"

    precond = FTHM_Solver.PetscFieldSplitScheme(
        groups=elim_group_0,
        elim_options={
            "pc_type": elim_pc_type_0,
            "ksp_type": ksp_0,
        },
        fieldsplit_options={
            "pc_fieldsplit_schur_precondition": "selfp",
        },
        complement=FTHM_Solver.PetscFieldSplitScheme(
            groups=elim_group_1,
            elim_options={
                "pc_type": elim_pc_type_1,
                "ksp_type": ksp_1,
            },
            fieldsplit_options={
                "pc_fieldsplit_schur_precondition": "a11",
            },
            complement=FTHM_Solver.PetscFieldSplitScheme(
                groups=keep_groups_1,
                elim_options={
                    "pc_type": keep_pc_type_1,
                },
            ),
        ),
    )
    # Unwind the nested configuration. This will also set up the preconditioner, hence
    # there is no need to call pc.setUp().
    precond.configure(block_matrix, pc)

    # The outer preconditioner should be a fieldsplit.
    assert pc.getType() == "fieldsplit"

    # Check that the ksp and preconditioner for the eliminated block in the first stage
    # are correct.
    elim_pc = pc.getFieldSplitSubKSP()[0]
    assert elim_pc.type == ksp_0
    assert elim_pc.getPC().type == elim_pc_type_0

    # The first complement is a fieldsplit.
    keep_pc = pc.getFieldSplitSubKSP()[1].getPC()
    assert keep_pc.getType() == "fieldsplit"
    # Check that the ksp and preconditioner for the eliminated block in the second stage
    # are correct.
    assert keep_pc.getFieldSplitSubKSP()[0].type == ksp_1
    assert keep_pc.getFieldSplitSubKSP()[0].getPC().type == elim_pc_type_1
    assert keep_pc.getFieldSplitSubKSP()[1].getPC().type == keep_pc_type_1

    # Also check that the IS for the eliminated and kept blocks are correct. For the
    # first stage, we can do this directly on the block matrix.
    known_elim_ind_0 = _dofs_from_group_ind(block_matrix, elim_group_0)
    known_keep_ind_0 = _dofs_from_group_ind(block_matrix, keep_groups_0)
    elim_0_IS = pc.getFieldSplitSubIS(elim_tag_0)
    keep_0_IS = pc.getFieldSplitSubIS(keep_tag_0)
    np.testing.assert_array_equal(elim_0_IS, known_elim_ind_0)
    np.testing.assert_array_equal(keep_0_IS, known_keep_ind_0)

    # The ISs on the second stage are set with respect to the non-eliminated blocks of
    # the first stage. We need to get the block matrix for the second stage.
    sub_block = block_matrix[keep_groups_0]
    known_elim_ind_1 = _dofs_from_group_ind(sub_block, elim_group_1)
    known_keep_ind_1 = _dofs_from_group_ind(sub_block, keep_groups_1)
    elim_1_IS = keep_pc.getFieldSplitSubIS(elim_tag_1)
    keep_1_IS = keep_pc.getFieldSplitSubIS(keep_tag_1)
    np.testing.assert_array_equal(elim_1_IS, known_elim_ind_1)
    np.testing.assert_array_equal(keep_1_IS, known_keep_ind_1)
