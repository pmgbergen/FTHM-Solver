from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from .block_matrix import BlockMatrixStorage
from .mat_utils import csr_to_petsc, make_сlear_petsc_options, petsc_to_csr
from petsc4py import PETSc


def construct_is(bmat: BlockMatrixStorage, groups: list[int]) -> PETSc.IS:
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


def build_tag(groups: list[int]) -> str:
    return "-".join([str(x) for x in groups])


def insert_petsc_options(options):
    petsc_options = PETSc.Options()
    for k, v in options.items():
        petsc_options[k] = v


@dataclass
class PetscFieldSplitScheme:
    groups: list[int]
    complement: Optional["PetscFieldSplitScheme"] = None
    elim_options: dict = None
    fieldsplit_options: dict = None
    keep_options: dict = None
    block_size: int = 1
    invert: Callable[[PETSc.Mat], PETSc.Mat] = None
    python_pc: PETSc.PC = None
    # experimental
    near_null_space: list[np.ndarray] = None
    ksp_keep_use_pmat: bool = False

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
        elim_options = self.elim_options or {}
        fieldsplit_options = self.fieldsplit_options or {}
        keep_options = self.keep_options or {}

        elim = self.groups
        if self.complement is None:
            options = (
                {
                    f"{prefix}ksp_type": "preonly",
                    f"{prefix}pc_type": "lu",
                }
                | {f"{prefix}{k}": v for k, v in elim_options.items()}
                | {f"{prefix}{k}": v for k, v in fieldsplit_options.items()}
            )

            if self.python_pc is not None:
                options[f"{prefix}pc_type"] = "python"
                python_pc = self.python_pc(bmat)
                python_pc.petsc_pc.setOptionsPrefix(f"{prefix}python_")
                petsc_pc.setType("python")
                petsc_pc.setPythonContext(python_pc)

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
            | {f"{prefix}fieldsplit_{elim_tag}_{k}": v for k, v in elim_options.items()}
            | {f"{prefix}fieldsplit_{keep_tag}_{k}": v for k, v in keep_options.items()}
            | {f"{prefix}{k}": v for k, v in fieldsplit_options.items()}
        )

        insert_petsc_options(options)
        petsc_pc.setFromOptions()
        petsc_pc.setFieldSplitIS((elim_tag, petsc_is_elim), (keep_tag, petsc_is_keep))

        if self.invert is not None:
            S = petsc_pc.getOperators()[1].createSubMatrix(petsc_is_keep, petsc_is_keep)
            petsc_stab = self.invert(bmat)
            S.axpy(1, petsc_stab)

            petsc_pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)

        petsc_pc.setUp()

        petsc_ksp_keep = petsc_pc.getFieldSplitSubKSP()[1]
        petsc_pc_keep = petsc_ksp_keep.getPC()
        petsc_ksp_elim = petsc_pc.getFieldSplitSubKSP()[0]
        petsc_pc_elim = petsc_ksp_elim.getPC()

        if self.ksp_keep_use_pmat:
            amat, pmat = petsc_ksp_keep.getOperators()
            petsc_ksp_keep.setOperators(pmat, pmat)

        if self.near_null_space is not None:
            null_space_vectors = []
            for b in self.near_null_space:
                null_space_vec_petsc = PETSc.Vec().create()  # possibly mem leak
                null_space_vec_petsc.setSizes(b.shape[0], self.block_size)
                null_space_vec_petsc.setUp()
                null_space_vec_petsc.setArray(b)
                null_space_vectors.append(null_space_vec_petsc)
            # possibly mem leak
            null_space_petsc = PETSc.NullSpace().create(True, null_space_vectors)
            petsc_pc_elim.getOperators()[1].setNearNullSpace(null_space_petsc)

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
        flow_options = {"ksp_type": "preonly"} | (self.pressure_options or {})
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


@dataclass
class PetscCompositeScheme:
    groups: list[int]
    solvers: list[PetscFieldSplitScheme]
    petsc_options: dict = None

    def get_groups(self) -> list[int]:
        return self.groups

    def configure(
        self, bmat: BlockMatrixStorage, petsc_pc: PETSc.PC, prefix: str = ""
    ) -> dict:
        options = {
            f"{prefix}{k}": v
            for k, v in (
                {
                    "pc_type": "composite",
                    "pc_composite_type": "multiplicative",
                    "pc_composite_pcs": ",".join(["none"] * len(self.solvers)),
                }
                | (self.petsc_options or {})
            ).items()
        }
        insert_petsc_options(options)
        petsc_pc.setFromOptions()
        petsc_pc.setUp()
        for i, solver in enumerate(self.solvers):
            sub_pc = petsc_pc.getCompositePC(i)
            sub_options = solver.configure(
                bmat=bmat, petsc_pc=sub_pc, prefix=f"{prefix}sub_{i}_"
            )
            options |= sub_options

        return options


class PcPythonPermutation:
    def __init__(self, perm: np.ndarray, block_size: int):
        self.petsc_pc = PETSc.PC().create()
        self.petsc_is_perm = PETSc.IS().createGeneral(perm.astype(np.int32))
        self.P_perm = PETSc.Mat()
        self.b = PETSc.Vec().create()
        self.bs = block_size
        self.b.setSizes(perm.size)
        self.b.setUp()

    def __del__(self):
        self.petsc_pc.destroy()
        self.petsc_is_perm.destroy()
        self.b.destroy()

    def view(self, pc: PETSc.PC, viewer: PETSc.Viewer) -> None:
        self.petsc_pc.view(viewer)

    def setFromOptions(self, pc: PETSc.PC) -> None:
        self.petsc_pc.setFromOptions()

    def setUp(self, pc: PETSc.PC) -> None:
        _, P = pc.getOperators()
        self.P_perm = P.permute(self.petsc_is_perm, self.petsc_is_perm)
        self.P_perm.setBlockSize(self.bs)
        self.petsc_pc.setOperators(self.P_perm, self.P_perm)
        self.petsc_pc.setUp()

    def reset(self, pc: PETSc.PC) -> None:
        self.petsc_pc.reset()
        self.P_perm.destroy()

    def apply(self, pc: PETSc.PC, b: PETSc.Vec, x: PETSc.Vec) -> None:
        b.copy(self.b)
        self.b.permute(self.petsc_is_perm)
        self.petsc_pc.apply(self.b, x)
        x.permute(self.petsc_is_perm, invert=True)
