from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from block_matrix import BlockMatrixStorage
from mat_utils import csr_to_petsc, make_сlear_petsc_options, petsc_to_csr
from petsc4py import PETSc


def construct_is(bmat, groups) -> PETSc.IS:
    empty_mat = bmat.empty_container()
    dofs = np.concatenate(
        [
            empty_mat.local_dofs_row[x]
            for i in groups
            for x in empty_mat.groups_to_blocks_row[i]
        ],
        dtype=np.int32,
    )
    return PETSc.IS().createGeneral(dofs)


def build_tag(groups):
    return "-".join([str(x) for x in groups])


def insert_petsc_options(options):
    petsc_options = PETSc.Options()
    for k, v in options.items():
        petsc_options[k] = v


@dataclass
class PetscFieldSplitScheme:
    groups: list[int]
    complement: Optional["PetscFieldSplitScheme"] = None
    subsolver_options: dict = None
    fieldsplit_options: dict = None
    block_size: int = 1
    # experimental
    # pcmat: Callable[[PETSc.Mat], PETSc.Mat] = None
    invert: Callable[[PETSc.Mat], PETSc.Mat] = None

    def get_groups(self) -> list[int]:
        groups = [g for g in self.groups]
        if self.complement is not None:
            groups.extend(self.complement.get_groups())
        return groups


def recursive(
    scheme: PetscFieldSplitScheme,
    bmat: BlockMatrixStorage,
    petsc_pc: PETSc.PC,
    prefix: str = "",
):
    subsolver_options = scheme.subsolver_options or {}
    fieldsplit_options = scheme.fieldsplit_options or {}

    elim = scheme.groups
    if scheme.complement is None:
        options = (
            {
                f"{prefix}ksp_type": "preonly",
                f"{prefix}pc_type": "lu",
            }
            | {f"{prefix}{k}": v for k, v in subsolver_options.items()}
            | {f"{prefix}{k}": v for k, v in fieldsplit_options.items()}
        )
        insert_petsc_options(options)
        petsc_pc.setFromOptions()
        A, Ap = petsc_pc.getOperators()
        A.setFromOptions()
        Ap.setFromOptions()
        petsc_pc.setUp()
        return options

    keep = scheme.complement.get_groups()
    elim_tag = build_tag(elim)
    keep_tag = build_tag(keep)
    empty_bmat = bmat.empty_container()[elim + keep]
    petsc_is_keep = construct_is(empty_bmat, keep)
    petsc_is_elim = construct_is(empty_bmat, elim)
    petsc_is_elim.setBlockSize(scheme.block_size)
    # petsc_is_keep.setBlockSize(scheme.complement.block_size)

    if scheme.invert is not None:
        fieldsplit_options["pc_fieldsplit_schur_precondition"] = "user"

    options = (
        {
            f"{prefix}pc_type": "fieldsplit",
            f"{prefix}pc_fieldsplit_type": "schur",
            f"{prefix}pc_fieldsplit_schur_precondition": "selfp",
            f"{prefix}fieldsplit_{elim_tag}_ksp_type": "preonly",
            f"{prefix}fieldsplit_{elim_tag}_pc_type": "lu",
            f"{prefix}fieldsplit_{keep_tag}_ksp_type": "preonly",
        }
        | {
            f"{prefix}fieldsplit_{elim_tag}_{k}": v
            for k, v in subsolver_options.items()
        }
        | {f"{prefix}{k}": v for k, v in fieldsplit_options.items()}
    )

    insert_petsc_options(options)
    petsc_pc.setFromOptions()
    petsc_pc.setFieldSplitIS((elim_tag, petsc_is_elim), (keep_tag, petsc_is_keep))
    petsc_pc.setUp()

    petsc_pc_elim = petsc_pc.getFieldSplitSubKSP()[0].getPC()
    petsc_pc_keep = petsc_pc.getFieldSplitSubKSP()[1].getPC()

    # if scheme.pcmat is not None:
    #     petsc_elim_amat = petsc_pc_elim.getOperators()[0]
    #     petsc_elim_pmat = scheme.pcmat(petsc_elim_amat)
    #     petsc_pc_elim.setOperators(petsc_elim_amat, petsc_elim_pmat)

    if scheme.invert is not None:
        # petsc_keep_S, petsc_keep_Pmat = petsc_pc_keep.getOperators()
        # A00, Ap00, A01, A10, A11 = petsc_keep_S.getSchurComplementSubMatrices()
        # petsc_stab = scheme.invert(None)
        # S = A11.duplicate(copy=True)
        # S.axpy(1, petsc_stab)

        # petsc_pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)
        S = scheme.invert(None)
        petsc_pc_keep.setOperators(A=S, P=S)
    # should we delete the old matrices ???

    options |= recursive(
        scheme.complement,
        bmat,
        prefix=f"{prefix}fieldsplit_{keep_tag}_",
        petsc_pc=petsc_pc_keep,
    )
    return options


def build_petsc_solver(
    bmat: BlockMatrixStorage, scheme: PetscFieldSplitScheme
) -> PETSc.KSP:
    petsc_mat = csr_to_petsc(bmat.mat)

    make_сlear_petsc_options()
    options = {
        "ksp_monitor": None,
        "ksp_type": "gmres",
        "ksp_pc_side": "right",
        "ksp_rtol": 1e-10,
        "ksp_max_it": 120,
    }
    insert_petsc_options(options)
    petsc_ksp = PETSc.KSP().create()
    petsc_ksp.setOperators(petsc_mat)
    petsc_pc = petsc_ksp.getPC()
    options |= recursive(
        scheme=scheme,
        bmat=bmat,
        petsc_pc=petsc_pc,
    )
    petsc_ksp.setFromOptions()
    petsc_ksp.setUp()
    return petsc_ksp, options
