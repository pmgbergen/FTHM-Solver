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
    tmp_options: dict = None
    block_size: int = 1
    # experimental
    pcmat: Callable[[PETSc.Mat], PETSc.Mat] = None
    invert: Callable[[PETSc.Mat], PETSc.Mat] = None
    tmp: PETSc.Mat = None

    def get_groups(self) -> list[int]:
        groups = [g for g in self.groups]
        if self.complement is not None:
            groups.extend(self.complement.get_groups())
        return groups


@dataclass
class PetscKSPScheme:
    preconditioner: PetscFieldSplitScheme
    petsc_options: dict = None

    def get_groups(self) -> list[int]:
        return self.preconditioner.get_groups()

    def make_solver(self, mat_orig: BlockMatrixStorage):
        petsc_mat = csr_to_petsc(mat_orig.mat)

        make_сlear_petsc_options()
        options = {
            # "ksp_monitor": None,
            "ksp_type": "gmres",
            "ksp_pc_side": "right",
            "ksp_rtol": 1e-10,
            "ksp_max_it": 120,
            'ksp_gmres_cgs_refinement_type': 'refine_ifneeded',
            'ksp_gmres_classicalgramschmidt': True
        } | (self.petsc_options or {})
        insert_petsc_options(options)
        petsc_ksp = PETSc.KSP().create()
        petsc_ksp.setOperators(petsc_mat)
        petsc_pc = petsc_ksp.getPC()
        options |= build_petsc_fieldsplit(
            scheme=self.preconditioner,
            bmat=mat_orig,
            petsc_pc=petsc_pc,
        )
        petsc_ksp.setFromOptions()
        petsc_ksp.setUp()
        return PetscKrylovSolver(petsc_ksp)


def build_petsc_fieldsplit(
    scheme: PetscFieldSplitScheme,
    bmat: BlockMatrixStorage,
    petsc_pc: PETSc.PC,
    prefix: str = "",
):
    subsolver_options = scheme.subsolver_options or {}
    fieldsplit_options = scheme.fieldsplit_options or {}
    tmp_options = scheme.tmp_options or {}

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
        petsc_pc.setUp()
        return options

    keep = scheme.complement.get_groups()
    elim_tag = build_tag(elim)
    keep_tag = build_tag(keep)
    empty_bmat = bmat.empty_container()[elim + keep]
    petsc_is_keep = construct_is(empty_bmat, keep)
    petsc_is_elim = construct_is(empty_bmat, elim)
    petsc_is_elim.setBlockSize(scheme.block_size)

    # if scheme.pcmat is not None:
    #     subsolver_options["pc_type"] = "mat"

    if scheme.invert is not None:
        fieldsplit_options["pc_fieldsplit_schur_precondition"] = "user"

    options = (
        {
            f"{prefix}pc_type": "fieldsplit",
            f"{prefix}pc_fieldsplit_type": "schur",
            f"{prefix}pc_fieldsplit_schur_precondition": "selfp",
            f"{prefix}pc_fieldsplit_schur_fact_type": "upper",
            f"{prefix}fieldsplit_{elim_tag}_ksp_type": "preonly",
            f"{prefix}fieldsplit_{elim_tag}_pc_type": "lu",
            f"{prefix}fieldsplit_{keep_tag}_ksp_type": "preonly",
        }
        | {
            f"{prefix}fieldsplit_{elim_tag}_{k}": v
            for k, v in subsolver_options.items()
        }
        | {f"{prefix}fieldsplit_{keep_tag}_{k}": v for k, v in tmp_options.items()}
        | {f"{prefix}{k}": v for k, v in fieldsplit_options.items()}
    )

    # options['fieldsplit_3_mat_schur_complement_ainv_type'] = 'blockdiag'

    insert_petsc_options(options)
    petsc_pc.setFromOptions()
    petsc_pc.setFieldSplitIS((elim_tag, petsc_is_elim), (keep_tag, petsc_is_keep))

    if scheme.invert is not None:
        S = petsc_pc.getOperators()[1].createSubMatrix(petsc_is_keep, petsc_is_keep)
        petsc_stab = scheme.invert(bmat)
        S.axpy(1, petsc_stab)

        petsc_pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)

    petsc_pc.setUp()

    petsc_pc_keep = petsc_pc.getFieldSplitSubKSP()[1].getPC()

    # if scheme.pcmat is not None:
    #     Ainv = scheme.pcmat(None)
    #     petsc_pc_elim.setOperators(Ainv, Ainv)
    #     S = scheme.tmp
    #     petsc_pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)

    
    # should we delete the old matrices ???

    # petsc_pc_keep.setUp()
    # petsc_pc_elim.setUp()

    options |= build_petsc_fieldsplit(
        scheme.complement,
        bmat,
        prefix=f"{prefix}fieldsplit_{keep_tag}_",
        petsc_pc=petsc_pc_keep,
    )
    return options


@dataclass
class LinearTransformedScheme:

    left_transformations: Optional[
        list[Callable[[BlockMatrixStorage], BlockMatrixStorage]]
    ] = None
    right_transformations: Optional[
        list[Callable[[BlockMatrixStorage], BlockMatrixStorage]]
    ] = None
    inner: Optional = None

    def get_groups(self) -> list[int]:
        return self.inner.get_groups()

    def make_solver(self, mat_orig: BlockMatrixStorage):
        groups = self.get_groups()
        bmat = mat_orig[groups]

        if self.left_transformations is None or len(self.left_transformations) == 0:
            Qleft = None
        else:
            Qleft = self.left_transformations[0](bmat)[groups]
            for tmp in self.left_transformations[1:]:
                tmp = tmp(bmat)[groups]
                Qleft.mat @= tmp.mat

        if self.right_transformations is None or len(self.right_transformations) == 0:
            Qright = None
        else:
            Qright = self.right_transformations[0](bmat)[groups]
            for tmp in self.right_transformations[1:]:
                tmp = tmp(bmat)[groups]
                Qright.mat @= tmp.mat

        bmat_Q = bmat
        if Qleft is not None:
            bmat_Q.mat = Qleft.mat @ bmat_Q.mat
        if Qright is not None:
            bmat_Q.mat = bmat_Q.mat @ Qright.mat

        solver = self.inner.make_solver(bmat_Q)

        if Qleft is not None or Qright is not None:
            solver = LinearSolverWithTransformations(
                inner=solver, Qright=Qright, Qleft=Qleft
            )

        return solver


class LinearSolverWithTransformations:

    def __init__(
        self,
        inner,
        Qleft: Optional[BlockMatrixStorage] = None,
        Qright: Optional[BlockMatrixStorage] = None,
    ):
        self.Qleft: BlockMatrixStorage | None = Qleft
        self.Qright: BlockMatrixStorage | None = Qright
        self.inner = inner
        self.ksp = inner.ksp

    def solve(self, rhs):
        rhs_Q = rhs
        if self.Qleft is not None:
            rhs_Q = self.Qleft.mat @ rhs_Q

        sol_Q = self.inner.solve(rhs_Q)

        if self.Qright is not None:
            sol = self.Qright.mat @ sol_Q
        else:
            sol = sol_Q

        return sol

    def get_residuals(self):
        return self.inner.get_residuals()


class PetscKrylovSolver:

    def __init__(
        self,
        ksp,
    ) -> None:
        self.ksp = ksp
        petsc_mat = ksp.getOperators()[0]
        self.petsc_x = petsc_mat.createVecLeft()
        self.petsc_b = petsc_mat.createVecLeft()
        # self.ksp.setComputeEigenvalues(True)
        self.ksp.setConvergenceHistory()

    def __del__(self):
        self.ksp.destroy()
        self.petsc_x.destroy()
        self.petsc_b.destroy()

    def solve(self, b):
        self.petsc_b.setArray(b)
        self.petsc_x.set(0.0)
        self.ksp.solve(self.petsc_b, self.petsc_x)
        res = self.petsc_x.getArray()
        return res

    def get_residuals(self):
        return self.ksp.getConvergenceHistory()
