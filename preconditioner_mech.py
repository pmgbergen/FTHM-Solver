from porepy.numerics.linalg.matrix_operations import sparse_kronecker_product
from block_matrix import BlockMatrixStorage, SolveSchema
import scipy.sparse
import numpy as np
from mat_utils import PetscAMGMechanics, inv, lump_nd, inv_block_diag
from scipy.sparse import csr_matrix
from numba import njit
import numba


def build_local_stabilization(
    model,
    bmat: BlockMatrixStorage,
    frac_cells,
    intf_cells,
    mat_cells,
    frac_idx,
    build_schur=True,
):
    j44 = bmat.slice_submatrix(
        local_indices=(frac_cells, frac_cells),
        group=(4, 4),
        subgroup=(frac_idx, frac_idx),
    ).toarray()
    j55 = bmat.slice_submatrix(
        local_indices=(intf_cells, intf_cells),
        group=(5, 5),
        subgroup=(frac_idx, frac_idx),
    ).toarray()
    j45 = bmat.slice_submatrix(
        local_indices=(frac_cells, intf_cells),
        group=(4, 5),
        subgroup=(frac_idx, frac_idx),
    ).toarray()
    j54 = bmat.slice_submatrix(
        local_indices=(intf_cells, frac_cells),
        group=(5, 4),
        subgroup=(frac_idx, frac_idx),
    ).toarray()

    j_local = np.block(
        [
            [j44, j45],
            [j54, j55],
        ]
    )

    j_local_inv = np.linalg.inv(j_local)
    if not build_schur:
        return j_local_inv

    j_vert = bmat.slice_submatrix(
        local_indices=(mat_cells, intf_cells), group=(1, 5), subgroup=(0, frac_idx)
    ).toarray()
    j_hor = bmat.slice_submatrix(
        local_indices=(intf_cells, mat_cells), group=(5, 1), subgroup=(frac_idx, 0)
    ).toarray()
    return -j_vert @ j_local_inv[model.nd :, model.nd :] @ j_hor


def build_mechanics_stabilization(
    model, bmat: BlockMatrixStorage, build_schur=True, lump=False
):
    if lump:
        bmat = bmat.copy()
        j44 = bmat[[4]]
        j55 = bmat[[5]]
        bmat[[4]] = lump_nd(j44.mat, nd=model.nd)
        bmat[[5]] = lump_nd(j55.mat, nd=model.nd)

    shape = bmat[[1]].shape if build_schur else bmat[[4, 5]].shape
    result = scipy.sparse.lil_matrix(shape)

    fractures = model.mdg.subdomains(dim=model.nd - 1)

    for frac_idx, frac in enumerate(fractures):
        intfs = model.mdg.subdomain_to_interfaces(sd=frac)
        for intf in intfs:
            if intf.dim < frac.dim:
                continue
            matrix = model.mdg.interface_to_subdomain_pair(intf=intf)[0]
            assert matrix.dim == model.nd

            secondary_to_mortar = intf.secondary_to_mortar_avg(nd=model.nd).tocsc()
            mortar_to_primary = intf.mortar_to_primary_avg(nd=model.nd).tocsc()
            faces_to_cells = sparse_kronecker_product(
                matrix.cell_faces.T, nd=model.nd
            ).tocsc()
            secondary_to_mortar.eliminate_zeros()
            mortar_to_primary.eliminate_zeros()
            faces_to_cells.eliminate_zeros()

            frac_to_mat_cells = faces_to_cells @ mortar_to_primary @ secondary_to_mortar

            for frac_cell_id in range(frac.num_cells):
                frac_cells = frac_cell_id * model.nd + np.arange(model.nd)
                intf_cells = secondary_to_mortar[:, frac_cells].indices
                mat_cells = frac_to_mat_cells[:, frac_cells].indices

                if build_schur:
                    global_i = global_j = mat_cells
                else:
                    j44 = bmat[[4]]
                    frac_cells_offset = j44.get_global_indices(
                        local_indices=(frac_cells, frac_cells),
                        group=(4, 4),
                        subgroup=(frac_idx, frac_idx),
                    )[0]
                    intf_cells_offset = bmat[[5]].get_global_indices(
                        local_indices=(intf_cells, intf_cells),
                        group=(5, 5),
                        subgroup=(frac_idx, frac_idx),
                    )[0]
                    global_i = global_j = np.concatenate(
                        [frac_cells_offset, j44.shape[0] + intf_cells_offset]
                    )

                global_i, global_j = np.meshgrid(
                    global_i, global_j, copy=False, sparse=True, indexing="ij"
                )
                result[global_i, global_j] += build_local_stabilization(
                    model=model,
                    bmat=bmat,
                    frac_cells=frac_cells,
                    intf_cells=intf_cells,
                    mat_cells=mat_cells,
                    frac_idx=frac_idx,
                    build_schur=build_schur,
                )
    return result.tocsr()


def make_local_fracture_dofs(model):
    matrices = model.mdg.subdomains(dim=model.nd)
    assert len(matrices) == 1
    matrix = matrices[0]
    fractures = model.mdg.subdomains(dim=model.nd - 1)

    faces_to_cells = sparse_kronecker_product(matrix.cell_faces.T, nd=model.nd).tocsc()

    frac_dofs = []
    intf_dofs = []
    matrix_dofs = []
    intf_offset = 0
    frac_offset = 0

    for frac in fractures:
        intfs = model.mdg.subdomain_to_interfaces(sd=frac)
        intf = intfs[0]
        assert all(intf.dim < frac.dim for intf in intfs[1:])

        secondary_to_mortar = intf.secondary_to_mortar_avg(nd=model.nd).tocsc()
        mortar_to_primary = intf.mortar_to_primary_avg(nd=model.nd).tocsc()
        sign_to_mortar_side = intf.sign_of_mortar_sides(nd=model.nd).diagonal()
        secondary_to_mortar.eliminate_zeros()
        mortar_to_primary.eliminate_zeros()
        faces_to_cells.eliminate_zeros()

        intf_to_mat_cells = faces_to_cells @ mortar_to_primary

        for frac_cell_id in range(frac.num_cells):
            frac_cells = frac_cell_id * model.nd + np.arange(model.nd)
            intf_cells = secondary_to_mortar[:, frac_cells].indices

            # One fracture cell for two interface sides
            frac_dofs.append(frac_cells + frac_offset)
            frac_dofs.append(frac_cells + frac_offset)

            signs = sign_to_mortar_side[intf_cells]
            for sign_val in (1, -1):
                sign_idx = signs == sign_val
                intf_cells_side = intf_cells[sign_idx]

                mat_cells_side = intf_to_mat_cells[:, intf_cells_side].indices
                matrix_dofs.append(mat_cells_side)
                intf_dofs.append(intf_cells_side + intf_offset)

        intf_offset += secondary_to_mortar.shape[0]
        frac_offset += secondary_to_mortar.shape[1]

    return np.array(matrix_dofs), np.array(frac_dofs), np.array(intf_dofs)


# def take_local_values(src, i_dofs, j_dofs):
#     res = scipy.sparse.lil_matrix(src.shape)
#     for i_dof, j_dof in zip(i_dofs, j_dofs):
#         I, J = np.meshgrid(i_dof, j_dof, copy=False, sparse=True, indexing="ij")
#         res[I, J] = src[I, J]
#     return res.asformat(src.format)


# def lump_rect(src, idofs_target, jdofs_target, axis=1):
#     nd = idofs_target.shape[1]
#     res = scipy.sparse.lil_matrix(src.shape)
#     for idof_target, jdof_target in zip(idofs_target, jdofs_target):
#         for i in range(nd):
#             for j in range(nd):
#                 if axis == 1:
#                     data = src[idof_target[i], jdof_target[j] % nd::nd]
#                 else:
#                     data = src[idof_target[i] % nd::nd, jdof_target[j]]
#                 data = np.array(data.sum(axis=axis)).ravel()
#                 res[idof_target[i], jdof_target[j]] = data
#     return res.asformat(src.format)


def make_J44_inv(model, bmat: BlockMatrixStorage, lump=False):
    J4_shape = bmat.group_shape([4])
    mech_stab = build_mechanics_stabilization(
        model=model, bmat=bmat, build_schur=False, lump=lump
    )
    return mech_stab[: J4_shape[0], : J4_shape[1]]


def make_J44_inv_bdiag(model, bmat: BlockMatrixStorage):
    assert "You shouldn't be here"
    J44 = bmat[4, 4].mat
    J55_inv = inv_block_diag(bmat[[5]].mat, nd=model.nd)
    # J55_inv = inv(bmat[5, 5].mat)
    stab = bmat[4, 5].mat @ J55_inv @ bmat[5, 4].mat

    st, sl, op, tr = model.sticking_sliding_open_transition()

    assert model.nd == 3
    sliding_tang = np.repeat(sl, model.nd)
    sliding_tang[::model.nd] = 0  # we need only tangential
    stab[sliding_tang] = 0

    sliding_norm = np.repeat(sl, model.nd)
    sliding_norm[1::model.nd] = 0
    sliding_norm[2::model.nd] = 0

    stab[sliding_norm, 0::3] = 0
    stab[sliding_norm, 1::3] = 0

    S44 = J44 - stab
    
    return inv_block_diag(S44, nd=model.nd)


def set_rows_zero(mat: csr_matrix, rows: np.ndarray) -> None:
    rows = np.array(rows)
    if rows.dtype == bool:
        _set_rows_zeros_bool(mat.data, mat.indptr, rows)
    else:
        _set_rows_zeros_int(mat.data, mat.indptr, rows)


@njit
def _set_rows_zeros_bool(
    csr_data: np.ndarray, csr_indptr: np.ndarray, rows: np.ndarray
) -> None:
    for i, set_zero in enumerate(rows):
        if set_zero:
            csr_data[csr_indptr[i] : csr_indptr[i + 1]] = 0.0


@njit
def _set_rows_zeros_int(
    csr_data: np.ndarray, csr_indptr: np.ndarray, rows: np.ndarray
) -> None:
    for i in rows:
        csr_data[csr_indptr[i] : csr_indptr[i + 1]] = 0.0
