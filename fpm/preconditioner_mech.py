from porepy.numerics.linalg.matrix_operations import sparse_kronecker_product
from block_matrix import BlockMatrixStorage
import scipy.sparse
import numpy as np
from mat_utils import lump_nd


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
