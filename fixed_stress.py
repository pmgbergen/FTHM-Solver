import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from mat_utils import csr_zeros
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
        if len(restr_local) == 0:
            continue
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

    J_15 = bmat[[1, 5]].mat.tocsr()
    J15_inv = csr_zeros(J_15.shape[0])

    for R in localization_mats:
        j15 = R @ J_15 @ R.T
        # j15 = R @ R.T

        j15_inv = scipy.sparse.linalg.inv(j15.tocsc())
        J15_inv += R.T @ j15_inv @ R

    return J15_inv


def make_local_stab_15(bmat: BlockMatrixStorage, base: int, nd: int):
    J15_inv = make_local_inverse_15(bmat=bmat, base=base, nd=nd)
    return -bmat[base, [1, 5]].mat @ J15_inv @ bmat[[1, 5], base].mat


def get_fixed_stress_stabilization(model, l_factor: float = 0.6):
    mu_lame = model.solid.shear_modulus()
    lambda_lame = model.solid.lame_lambda()
    alpha_biot = model.solid.biot_coefficient()
    dim = model.nd

    l_phys = alpha_biot**2 / (2 * mu_lame / dim + lambda_lame)
    l_min = alpha_biot**2 / (4 * mu_lame + 2 * lambda_lame)

    val = l_min * (l_phys / l_min) ** l_factor

    diagonal_approx = val
    subdomains = model.mdg.subdomains(dim=dim)
    cell_volumes = subdomains[0].cell_volumes
    diagonal_approx *= cell_volumes

    density = model.fluid_density(subdomains).value(model.equation_system)
    diagonal_approx *= density

    dt = model.time_manager.dt
    diagonal_approx /= dt

    return scipy.sparse.diags(diagonal_approx)


def get_fixed_stress_stabilization_nd(model, l_factor: float = 0.6):
    mat_nd = get_fixed_stress_stabilization(model=model, l_factor=l_factor)

    sd_lower = [
        sd for d in reversed(range(model.nd)) for sd in model.mdg.subdomains(dim=d)
    ]
    num_cells = sum(sd.num_cells for sd in sd_lower)

    zero_lower = scipy.sparse.csr_matrix((num_cells, num_cells))
    return scipy.sparse.block_diag([mat_nd, zero_lower]).tocsr()


def make_fs(model, J: BlockMatrixStorage):
    diag = [
        get_fixed_stress_stabilization(model),
        make_local_stab_15(bmat=J, base=2, nd=1),
    ]
    result = J.empty_container()[[0, 2]]
    result.mat = scipy.sparse.block_diag(diag, format="csr")
    return result
