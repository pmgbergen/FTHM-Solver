import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from mat_utils import csr_zeros
from block_matrix import BlockMatrixStorage


# def assemble_localization_matrices_mechanics(
#     bmat: BlockMatrixStorage,
#     base: int,
#     nd: int,
# ) -> list:
#     # 2: A  Q1Q1Q1
#     # 1: Q2 B   M1
#     # 5: Q2 M2  P
#     bmat = bmat[[base, 1, 5]]

#     Q1 = bmat[base, [1, 5]].mat.tocsr()
#     Q2 = bmat[[1, 5], base].mat.tocsc()
#     M1 = bmat[1, 5].mat.tocsc()
#     M2 = bmat[5, 1].mat.tocsr()

#     B_size = M1.shape[0]

#     restrictions = []

#     assert (Q1.shape[0] % nd) == 0
#     num_cells = Q1.shape[0] // nd

#     for i in range(num_cells):
#         frac_dofs = np.arange(nd * i, nd * (i + 1))
#         restr_q1 = Q1[frac_dofs, :].indices
#         restr_q2 = Q2[:, frac_dofs].indices
#         restr = np.unique(np.concatenate([restr_q1, restr_q2]))

#         restr_local = restr - B_size
#         assert np.all(restr_local >= 0)
#         restr_m1 = M1[:, restr_local].indices
#         restr_m2 = M2[restr_local, :].indices

#         restr_local = np.unique(np.concatenate([restr_m1, restr_m2]))
#         if len(restr_local) == 0:
#             continue
#         assert np.all(restr_local < B_size)
#         restr_total = np.concatenate([restr_local, restr])

#         col_idx = np.array(restr_total)
#         data = np.ones_like(restr_total)
#         row_idx = np.arange(col_idx.size)
#         localization = scipy.sparse.csr_matrix(
#             (data, (row_idx, col_idx)), shape=(col_idx.size, Q1.shape[1])
#         )

#         restrictions.append(localization)

#     return restrictions


# def make_local_inverse_15(bmat: BlockMatrixStorage, base: int, nd: int):
#     localization_mats = assemble_localization_matrices_mechanics(bmat, base=base, nd=nd)

#     J_15 = bmat[[1, 5]].mat.tocsr()
#     J15_inv = csr_zeros(J_15.shape[0])

#     for R in localization_mats:
#         j15 = R @ J_15 @ R.T
#         # j15 = R @ R.T

#         j15_inv = scipy.sparse.linalg.inv(j15.tocsc())
#         J15_inv += R.T @ j15_inv @ R

#     return J15_inv


# def make_local_stab_15(bmat: BlockMatrixStorage, base: int, nd: int):
#     J15_inv = make_local_inverse_15(bmat=bmat, base=base, nd=nd)
#     return -bmat[base, [1, 5]].mat @ J15_inv @ bmat[[1, 5], base].mat


def get_fixed_stress_stabilization(model, l_factor: float = 0.6):
    mu_lame = model.solid.shear_modulus
    lambda_lame = model.solid.lame_lambda
    alpha_biot = model.solid.biot_coefficient
    dim = model.nd

    subdomains = model.mdg.subdomains(dim=dim)
    cell_volumes = subdomains[0].cell_volumes
    if alpha_biot == 0:
        return scipy.sparse.diags(0 * cell_volumes)

    l_phys = alpha_biot**2 / (2 * mu_lame / dim + lambda_lame)
    l_min = alpha_biot**2 / (4 * mu_lame + 2 * lambda_lame)

    val = l_min * (l_phys / l_min) ** l_factor

    diagonal_approx = val
    diagonal_approx *= cell_volumes

    density = model.fluid.density(subdomains).value(model.equation_system)
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


# def make_fs(model, J: BlockMatrixStorage):
#     diag = [
#         get_fixed_stress_stabilization(model),
#         make_local_stab_15(bmat=J, base=2, nd=1),
#     ]
#     result = J.empty_container()[[0, 2]]
#     result.mat = scipy.sparse.block_diag(diag, format="csr")
#     return result


def get_fs_fractures_analytical(model):
    alpha_biot = model.solid.biot_coefficient  # [-]
    lame_lambda = model.solid.lame_lambda  # [Pa]
    M = 1 / model.solid.specific_storage  # [Pa]
    compressibility = model.fluid.components[0].compressibility  # [1 / Pa]
    porosity = model.solid.porosity
    resid_aperture = model.solid.residual_aperture  # [m]

    fractures = model.mdg.subdomains(dim=model.nd - 1)
    intersections = [
        frac
        for dim in reversed(range(model.nd - 1))
        for frac in model.mdg.subdomains(dim=dim)
    ]
    # fractures += intersections

    nd_vec_to_normal = model.normal_component(fractures)
    # The normal component of the contact traction and the displacement jump.
    u_n = nd_vec_to_normal @ model.displacement_jump(fractures)
    u_n = u_n.value(model.equation_system)

    # alpha^2 / (lambda * (1 / (C_f * M) + phi_0))
    # val = alpha_biot**2 / (lame_lambda * (1 / (compressibility * M) + porosity))

    # C_f_c * M * alpha^2 / (lambda * (1 + phi_0 * M * C_f))
    # val = (
    #     compressibility
    #     * M
    #     * alpha_biot**2
    #     / (lame_lambda * (1 + porosity * M * compressibility))
    # )

    val = (
        alpha_biot**2
        * u_n  # / resid_aperture# ** 3
        / (lame_lambda / (compressibility * M) + porosity * lame_lambda)
    )

    if len(fractures) == 0:
        return scipy.sparse.csr_matrix((0, 0))

    cell_volumes = np.concatenate([f.cell_volumes for f in fractures])
    val *= cell_volumes

    # intersections ?

    # specific volume ?
    # specific_volume = model.specific_volume(fractures).value(model.equation_system)
    # val *= specific_volume

    density = model.fluid.density(fractures).value(model.equation_system)
    val *= density

    dt = model.time_manager.dt
    val /= dt

    intersect_zeros = np.zeros(sum(f.num_cells for f in intersections))
    val = np.concatenate([val, intersect_zeros])

    return scipy.sparse.diags(val)


def make_fs_analytical(model, J, p_mat_group: int, p_frac_group: int):
    groups = [p_mat_group, p_frac_group]
    diag = [
        get_fixed_stress_stabilization(model),
        get_fs_fractures_analytical(model),
    ]
    result = J.empty_container()[groups]
    result.mat = scipy.sparse.block_diag(diag, format="csr")
    # result[groups] = scipy.sparse.block_diag(diag, format="csr")
    return result


def make_fs_analytical_slow(model, J, p_mat_group: int, p_frac_group: int, groups):
    result = J.empty_container()[groups]
    result[[p_mat_group]] = scipy.sparse.block_diag(
        [get_fixed_stress_stabilization(model)], format="csr"
    )
    result[[p_frac_group]] = scipy.sparse.block_diag(
        [get_fs_fractures_analytical(model)], format="csr"
    )
    return result


def make_fs_analytical_slow_new(
    model, J: BlockMatrixStorage, p_mat_group: int, p_frac_group: int, groups: list[int]
):
    index = J[groups].empty_container()
    # assert p_mat_group in groups
    # assert p_frac_group in groups
    diagonals = []
    for group in groups:
        if group == p_mat_group:
            diagonals.append(get_fixed_stress_stabilization(model))
        elif groups == p_frac_group:
            diagonals.append(get_fs_fractures_analytical(model))
        else:
            diagonals.append(index[[group]].mat)
    index.mat = scipy.sparse.block_diag(diagonals, format="csr")
    return index


def make_fs_thermal(model, J, p_mat_group: int, t_mat_group: int, groups=None):
    if groups is None:
        groups = [p_mat_group, t_mat_group]
    assert p_mat_group in groups
    # assert p_frac_group in groups
    assert t_mat_group in groups

    diag = [
        get_fixed_stress_stabilization(model),
        get_fixed_stress_stabilization_energy(model),
    ]
    result = J.empty_container()[groups]
    result[groups].mat = scipy.sparse.block_diag(diag, format="csr")
    return result


def get_fixed_stress_stabilization_energy(model, l_factor: float = 0.6):
    mu_lame = model.solid.shear_modulus()
    lambda_lame = model.solid.lame_lambda()
    beta_thermal = model.solid.thermal_expansion()
    dim = model.nd

    l_phys = beta_thermal**2 / (2 * mu_lame / dim + lambda_lame)
    l_min = beta_thermal**2 / (4 * mu_lame + 2 * lambda_lame)

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


def block_matrix(
    bmat: BlockMatrixStorage, submatrices: dict[int, scipy.sparse.spmatrix]
) -> BlockMatrixStorage:
    res = bmat.empty_container()
    for idx, submat in submatrices.items():
        if submat is not None:
            res[idx] = submat
    return res
