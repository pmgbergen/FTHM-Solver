import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from .block_matrix import BlockMatrixStorage
import porepy as pp

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


def get_fixed_stress_stabilization(model, l_factor: float = 0.6) -> sps.spmatrix:
    """Define the fixed stress stabilization matrix."""

    mu_lame = model.solid.shear_modulus
    lambda_lame = model.solid.lame_lambda
    alpha_biot = model.solid.biot_coefficient
    dim = model.nd

    subdomains = model.mdg.subdomains(dim=dim)
    cell_volumes = subdomains[0].cell_volumes
    if alpha_biot == 0:
        return sps.diags(0 * cell_volumes)

    # Stabilization value determined by physical reasoning.
    l_phys = alpha_biot**2 / (2 * mu_lame / dim + lambda_lame)
    # Stabilization value determined by theoretical reasoning.
    l_min = alpha_biot**2 / (4 * mu_lame + 2 * lambda_lame)

    # TODO: Where does this formula come from?
    val = l_min * (l_phys / l_min) ** l_factor

    diagonal_approx = val
    diagonal_approx *= cell_volumes

    density = model.fluid.density(subdomains).value(model.equation_system)
    diagonal_approx *= density

    dt = model.time_manager.dt
    diagonal_approx /= dt

    return sps.diags(diagonal_approx)


def get_fixed_stress_stabilization_nd(
    model: pp.PorePyModel, l_factor: float = 0.6
) -> sps.spmatrix:
    mat_nd = get_fixed_stress_stabilization(model=model, l_factor=l_factor)

    sd_lower = [
        sd for d in reversed(range(model.nd)) for sd in model.mdg.subdomains(dim=d)
    ]
    num_cells = sum(sd.num_cells for sd in sd_lower)

    zero_lower = sps.csr_matrix((num_cells, num_cells))
    return sps.block_diag([mat_nd, zero_lower]).tocsr()


# def make_fs(model, J: BlockMatrixStorage):
#     diag = [
#         get_fixed_stress_stabilization(model),
#         make_local_stab_15(bmat=J, base=2, nd=1),
#     ]
#     result = J.empty_container()[[0, 2]]
#     result.mat = scipy.sparse.block_diag(diag, format="csr")
#     return result


def get_fs_fractures_analytical(model: pp.PorePyModel) -> sps.spmatrix:
    alpha_biot = model.solid.biot_coefficient  # [-]
    lame_lambda = model.solid.lame_lambda  # [Pa]
    M = 1 / model.solid.specific_storage  # [Pa]
    compressibility = model.fluid.components[0].compressibility  # [1 / Pa]
    porosity = model.solid.porosity

    fractures = model.mdg.subdomains(dim=model.nd - 1)

    if len(fractures) == 0:
        return sps.csr_matrix((0, 0))

    nd_vec_to_normal = model.normal_component(fractures)
    # The normal component of the contact traction and the displacement jump.
    u_n_operator = nd_vec_to_normal @ model.displacement_jump(fractures)
    u_n = u_n_operator.value(model.equation_system)

    if compressibility != 0:
        val = (
            alpha_biot**2
            * u_n  # / resid_aperture# ** 3
            / (lame_lambda / (compressibility * M) + porosity * lame_lambda)
        )
    else:
        val = 0

    cell_volumes = np.concatenate([f.cell_volumes for f in fractures])
    val *= cell_volumes

    density = model.fluid.density(fractures).value(model.equation_system)
    val *= density

    dt = model.time_manager.dt
    val /= dt

    return sps.diags(val)


def make_fs_analytical(
    model, J: BlockMatrixStorage, p_mat_group: int, p_frac_group: int
) -> BlockMatrixStorage:
    groups = [p_mat_group, p_frac_group]
    diag = [
        get_fixed_stress_stabilization(model),
        get_fs_fractures_analytical(model),
    ]
    result = J.empty_container()[groups]
    result.mat = sps.block_diag(diag, format="csr")
    # result[groups] = sps.block_diag(diag, format="csr")
    return result


def make_fs_analytical_slow(model, J, p_mat_group: int, p_frac_group: int, groups):
    result = J.empty_container()[groups]
    result[[p_mat_group]] = sps.block_diag(
        [get_fixed_stress_stabilization(model)], format="csr"
    )
    result[[p_frac_group]] = sps.block_diag(
        [get_fs_fractures_analytical(model)], format="csr"
    )
    return result


def make_fs_analytical_slow_new(
    model, J: BlockMatrixStorage, p_mat_group: int, p_frac_group: int, groups: list[int]
) -> BlockMatrixStorage:
    index = J[groups].empty_container()
    diagonals = []
    for group in groups:
        if group == p_mat_group:
            diagonals.append(get_fixed_stress_stabilization(model))
        elif group == p_frac_group:
            diagonals.append(get_fs_fractures_analytical(model))
        else:
            diagonals.append(index[[group]].mat)
    index.mat = sps.block_diag(diagonals, format="csr")
    return index


def make_fs_thermal(
    model,
    J: BlockMatrixStorage,
    p_mat_group: int,
    p_frac_group: int,
    t_mat_group: int,
    t_frac_group: int,
    groups: list[int],
) -> BlockMatrixStorage:
    index = J[groups].empty_container()
    # assert p_mat_group in groups
    # assert p_frac_group in groups
    diagonals = []
    for group in groups:
        if group == p_mat_group:
            diagonals.append(get_fixed_stress_stabilization(model))
        elif groups == p_frac_group:
            diagonals.append(get_fs_fractures_analytical(model))
        elif group == t_mat_group:
            diagonals.append(get_fixed_stress_stabilization_energy(model))
        elif group == t_frac_group:
            diagonals.append(get_fs_fractures_energy(model))
        else:
            diagonals.append(index[[group]].mat)
    index.mat = sps.block_diag(diagonals, format="csr")
    return index


def get_fixed_stress_stabilization_energy(model, l_factor: float = 0.6):
    mu_lame = model.solid.shear_modulus
    lambda_lame = model.solid.lame_lambda
    beta_thermal = model.solid.thermal_expansion
    dim = model.nd

    l_phys = beta_thermal**2 / (2 * mu_lame / dim + lambda_lame)
    l_min = beta_thermal**2 / (4 * mu_lame + 2 * lambda_lame)

    val = l_min * (l_phys / l_min) ** l_factor

    diagonal_approx = val
    subdomains = model.mdg.subdomains(dim=dim)
    cell_volumes = subdomains[0].cell_volumes
    diagonal_approx *= cell_volumes

    density = model.fluid.density(subdomains).value(model.equation_system)
    diagonal_approx *= density

    dt = model.time_manager.dt
    diagonal_approx /= dt

    return sps.diags(diagonal_approx)


def get_fs_fractures_energy(model):
    beta_thermal = model.solid.thermal_expansion  # [?]
    lame_lambda = model.solid.lame_lambda  # [Pa]
    M = 1 / model.solid.specific_storage  # [Pa]
    compressibility = model.fluid.components[0].compressibility  # [1 / Pa]
    porosity = model.solid.porosity

    fractures = model.mdg.subdomains(dim=model.nd - 1)
    intersections = [
        frac
        for dim in reversed(range(model.nd - 1))
        for frac in model.mdg.subdomains(dim=dim)
    ]

    nd_vec_to_normal = model.normal_component(fractures)
    # The normal component of the contact traction and the displacement jump.
    u_n = nd_vec_to_normal @ model.displacement_jump(fractures)
    u_n = u_n.value(model.equation_system)

    val = (
        beta_thermal**2
        * u_n  # / resid_aperture# ** 3
        / (lame_lambda / (compressibility * M) + porosity * lame_lambda)
    )

    if len(fractures) == 0:
        return sps.csr_matrix((0, 0))

    cell_volumes = np.concatenate([f.cell_volumes for f in fractures])
    val *= cell_volumes

    density = model.fluid.density(fractures).value(model.equation_system)
    val *= density

    dt = model.time_manager.dt
    val /= dt

    # not sure
    specific_vol = model.specific_volume(fractures).value(model.equation_system)
    val /= specific_vol

    # intersect_zeros = np.zeros(sum(f.num_cells for f in intersections))
    # val = np.concatenate([val, intersect_zeros])

    return sps.diags(val)
