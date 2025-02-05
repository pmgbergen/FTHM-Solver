from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from block_matrix import BlockMatrixStorage
from mat_utils import csr_to_petsc, make_сlear_petsc_options, petsc_to_csr
from petsc4py import PETSc


def construct_is(bmat, groups) -> PETSc.IS:
    empty_mat = bmat.empty_container()
    dofs = [
        empty_mat.local_dofs_row[x]
        for i in groups
        for x in empty_mat.groups_to_blocks_row[i]
    ]
    if len(dofs) > 0:
        return PETSc.IS().createGeneral(
            np.concatenate(
                dofs,
                dtype=np.int32,
            )
        )
    else:
        return PETSc.IS().createGeneral(np.array([], dtype=np.int32))


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

    def configure(
        self,
        bmat: BlockMatrixStorage,
        petsc_pc: PETSc.PC,
        prefix: str = "",
    ):
        subsolver_options = self.subsolver_options or {}
        fieldsplit_options = self.fieldsplit_options or {}
        tmp_options = self.tmp_options or {}

        elim = self.groups
        if self.complement is None:
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

        keep = self.complement.get_groups()
        elim_tag = build_tag(elim)
        keep_tag = build_tag(keep)
        empty_bmat = bmat.empty_container()[elim + keep]
        petsc_is_keep = construct_is(empty_bmat, keep)
        petsc_is_elim = construct_is(empty_bmat, elim)
        petsc_is_elim.setBlockSize(self.block_size)

        # if scheme.pcmat is not None:
        #     subsolver_options["pc_type"] = "mat"

        if self.invert is not None:
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

        if self.invert is not None:
            S = petsc_pc.getOperators()[1].createSubMatrix(petsc_is_keep, petsc_is_keep)
            petsc_stab = self.invert(bmat)
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

        options |= self.complement.configure(
            bmat,
            prefix=f"{prefix}fieldsplit_{keep_tag}_",
            petsc_pc=petsc_pc_keep,
        )
        return options


@dataclass
class PetscKSPScheme:
    preconditioner: Optional[PetscFieldSplitScheme] = None
    petsc_options: Optional[dict] = None
    compute_eigenvalues: bool = False

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
            "ksp_gmres_cgs_refinement_type": "refine_ifneeded",
            "ksp_gmres_classicalgramschmidt": True,
        } | (self.petsc_options or {})
        insert_petsc_options(options)
        petsc_ksp = PETSc.KSP().create()
        petsc_ksp.setOperators(petsc_mat)
        petsc_ksp.setFromOptions()
        petsc_pc = petsc_ksp.getPC()
        if self.preconditioner is not None:
            options |= self.preconditioner.configure(
                bmat=mat_orig,
                petsc_pc=petsc_pc,
            )
        if self.compute_eigenvalues:
            petsc_ksp.setComputeEigenvalues(True)
        petsc_ksp.setUp()
        self.options = options
        return PetscKrylovSolver(petsc_ksp)


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
        self.options = self.inner.options

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


@dataclass
class PetscCPRScheme:
    groups: list[int]
    pressure_groups: list[int]
    pressure_options: dict = None
    others_options: dict = None
    cpr_options: dict = None

    def get_groups(self) -> list[int]:
        return self.groups

    def configure(
        self, bmat: BlockMatrixStorage, petsc_pc: PETSc.PC, prefix: str = ""
    ) -> dict:
        bmat = bmat[self.groups]
        cpr_options = self.cpr_options or {}
        flow_options = self.pressure_options or {}
        others_options = self.others_options or {}
        other_groups = [gr for gr in self.groups if gr not in self.pressure_groups]
        flow_tag = build_tag(self.pressure_groups)
        others_tag = build_tag(other_groups)
        flow_prefix = f"{prefix}sub_0_fieldsplit_{flow_tag}_"
        others_prefix = f"{prefix}sub_0_fieldsplit_{others_tag}_"
        options = (
            {
                f"{prefix}pc_type": "composite",
                f"{prefix}pc_composite_type": "multiplicative",
                f"{prefix}pc_composite_pcs": "fieldsplit,ilu",
                # f"{prefix}sub_0_ksp_type": "preonly",
                f"{prefix}sub_0_pc_fieldsplit_type": "additive",
            }
            | {f"{prefix}{k}": v for k, v in cpr_options.items()}
            | {f"{flow_prefix}{k}": v for k, v in flow_options.items()}
            | {f"{others_prefix}{k}": v for k, v in others_options.items()}
        )
        insert_petsc_options(options)
        petsc_pc.setFromOptions()

        petsc_is_flow = construct_is(bmat, self.pressure_groups)
        petsc_is_others = construct_is(bmat, other_groups)
        fieldsplit = petsc_pc.getCompositePC(0)
        fieldsplit.setFieldSplitIS(
            (flow_tag, petsc_is_flow), (others_tag, petsc_is_others)
        )

        petsc_pc.setUp()
        fieldsplit.setUp()
        return options
