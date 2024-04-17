import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from mat_utils import csr_zeros
from pp_utils import get_fixed_stress_stabilization
from block_matrix import BlockMatrixStorage


def assemble_localization_matrices_mechanics(
    bmat: BlockMatrixStorage,
    base: int,
    nd: int,
) -> list:
    # 2: A  Q1Q1Q1
    # 1: Q2 B   M1
    # 5: Q2 M2  P
    bmat = bmat[[base, 1, 5]]

    Q1 = bmat[base, [1, 5]].mat.tocsr()
    Q2 = bmat[[1, 5], base].mat.tocsc()
    M1 = bmat[1, 5].mat.tocsc()
    M2 = bmat[5, 1].mat.tocsr()

    B_size = M1.shape[0]

    restrictions = []

    assert (Q1.shape[0] % nd) == 0
    num_cells = Q1.shape[0] // nd

    for i in range(num_cells):
        frac_dofs = np.arange(nd * i, nd * (i + 1))
        restr_q1 = Q1[frac_dofs, :].indices
        restr_q2 = Q2[:, frac_dofs].indices
        restr = np.unique(np.concatenate([restr_q1, restr_q2]))

        restr_local = restr - B_size
        assert np.all(restr_local >= 0)
        restr_m1 = M1[:, restr_local].indices
        restr_m2 = M2[restr_local, :].indices

        restr_local = np.unique(np.concatenate([restr_m1, restr_m2]))
        assert len(restr_local) > 0
        assert np.all(restr_local < B_size)
        restr_total = np.concatenate([restr_local, restr])

        col_idx = np.array(restr_total)
        data = np.ones_like(restr_total)
        row_idx = np.arange(col_idx.size)
        localization = scipy.sparse.csr_matrix(
            (data, (row_idx, col_idx)), shape=(col_idx.size, Q1.shape[1])
        )

        restrictions.append(localization)

    return restrictions


def make_local_inverse_15(bmat: BlockMatrixStorage, base: int, nd: int):
    localization_mats = assemble_localization_matrices_mechanics(bmat, base=base, nd=nd)

    J_15 = bmat[[1, 5]].mat
    J15_inv = csr_zeros(J_15.shape[0])

    for R in localization_mats:
        # j15 = R @ J_15 @ R.T
        j15 = R @ R.T

        j15_inv = scipy.sparse.linalg.inv(j15)
        J15_inv += R.T @ j15_inv @ R

    return J15_inv


def make_local_stab_15(bmat: BlockMatrixStorage, base: int, nd: int):
    J15_inv = make_local_inverse_15(bmat=bmat, base=base, nd=nd)
    return -bmat[base, [1, 5]].mat @ J15_inv @ bmat[[1, 5], base].mat


def make_fs(model, J: BlockMatrixStorage):
    result = J.empty_container()[[0, 2]]
    result[0, 0] = get_fixed_stress_stabilization(model)
    result[2, 2] = make_local_stab_15(bmat=J, base=2, nd=1)
    return result
