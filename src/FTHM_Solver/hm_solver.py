from functools import cached_property
from .block_matrix import BlockMatrixStorage, FieldSplitScheme, KSPScheme
from .fixed_stress import make_fs_analytical, make_fs_analytical_slow
from .iterative_solver import (
    IterativeLinearSolver,
    get_equations_group_ids,
    get_variables_group_ids,
)

import numpy as np
import scipy.sparse
from .mat_utils import (
    PetscAMGFlow,
    PetscAMGMechanics,
    PetscILU,
    csr_ones,
    extract_diag_inv,
    inv_block_diag,
)


class IterativeHMSolver(IterativeLinearSolver):

    CONTACT_GROUP = 0

    def group_row_names(self) -> list[str]:
        return [
            "Contact frac.",
            "Flow intf.",
            "Force mat.",
            "Force intf.",
            "Flow mat.",
            "Flow frac.",
        ]

    def group_col_names(self) -> list[str]:
        return [
            r"$\lambda_{frac}$",
            r"$v_{intf}$",
            r"$u_{3D}$",
            r"$u_{intf}$",
            r"$p_{3D}$",
            r"$p_{frac}$",
        ]

    @cached_property
    def variable_groups(self) -> list[list[int]]:
        """Prepares the groups of variables in the specific order, that we will use in
        the block Jacobian to access the submatrices:

        `J[x, 0]` - matrix pressure variable;
        `J[x, 1]` - matrix displacement variable;
        `J[x, 2]` - lower-dim pressure variable;
        `J[x, 3]` - interface Darcy flux variable;
        `J[x, 4]` - contact traction variable;
        `J[x, 5]` - interface displacement variable;

        This index is not equivalen to PorePy model natural ordering. Constructed when
        first accessed.

        """
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_lower = [
            k for i in reversed(range(0, dim_max)) for k in self.mdg.subdomains(dim=i)
        ]
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        intf = self.mdg.interfaces()
        intf_frac = self.mdg.interfaces(dim=dim_max - 1)

        return get_variables_group_ids(
            model=self,
            md_variables_groups=[
                [self.contact_traction(sd_frac)],  # 0
                [self.interface_darcy_flux(intf)],  # 1
                [self.displacement(sd_ambient)],  # 2
                [self.interface_displacement(intf_frac)],  # 3
                [self.pressure(sd_ambient)],  # 4
                [self.pressure(sd_lower)],  # 5
            ],
        )

    @cached_property
    def equation_groups(self) -> list[list[int]]:
        """Prepares the groups of equation in the specific order, that we will use in
        the block Jacobian to access the submatrices:

        `J[0, x]` - matrix mass balance equation;
        `J[1, x]` - matrix momentum balance equation;
        `J[2, x]` - lower-dim mass balance equation;
        `J[3, x]` - interface Darcy flux equation;
        `J[4, x]` - contact traction equations;
        `J[5, x]` - interface force balance equation;

        This index is not equivalen to PorePy model natural ordering. Constructed when
        first accessed. Encorporates the permutation `contact_permutation`.

        """
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_lower = [
            k for i in reversed(range(0, dim_max)) for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()

        return self._correct_contact_equations_groups(
            equation_groups=get_equations_group_ids(
                model=self,
                equations_group_order=[
                    [  # 0
                        ("normal_fracture_deformation_equation", sd_lower),
                        ("tangential_fracture_deformation_equation", sd_lower),
                    ],
                    [("interface_darcy_flux_equation", intf)],  # 1
                    [("momentum_balance_equation", sd_ambient)],  # 2
                    [("interface_force_balance_equation", intf)],  # 3
                    [("mass_balance_equation", sd_ambient)],  # 4
                    [("mass_balance_equation", sd_lower)],  # 5
                ],
            ),
            contact_group=self.CONTACT_GROUP,
        )

    @cached_property
    def contact_permutation(self) -> np.ndarray:
        """Permutation of the contact mechanics equations. Must be applied to the
        Jacobian.

        The PorePy arrangement is:
        `[[C0_norm], [C1_norm], [C0_tang], [C1_tang]]`,
        where `C0` and `C1` correspond to the contact equation on fractures 0 and 1.
        We permute it to:
        `[[f0_norm, f0_tang], [f1_norm, f1_tang]]`, a.k.a array of structures.

        """
        return make_reorder_contact(self, contact_group=self.CONTACT_GROUP)

    @cached_property
    def eq_dofs(self):
        unpermuted_eq_dofs = super().eq_dofs
        return self._correct_contact_eq_dofs(
            unpermuted_eq_dofs, contact_group=self.CONTACT_GROUP
        )

    def _correct_contact_eq_dofs(
        self, unpermuted_eq_dofs: list[np.ndarray], contact_group: int
    ) -> list[np.ndarray]:
        if len(self.equation_groups[contact_group]) == 0:
            return unpermuted_eq_dofs

        # We assume that normal equations go first.
        normal_blocks = self.equation_groups[contact_group]
        num_fracs = len(self.mdg.subdomains(dim=self.nd - 1))
        # One tangential block matches 1 normal for 2D and 1 tangential block for 3D.
        all_contact_blocks = [
            nb + i * num_fracs for i in range(2) for nb in normal_blocks
        ]

        eq_dofs_corrected = []
        for i, x in enumerate(unpermuted_eq_dofs):
            if i not in all_contact_blocks:
                eq_dofs_corrected.append(x)
            elif i in normal_blocks:
                eq_dofs_corrected.append(None)

        i = unpermuted_eq_dofs[normal_blocks[0]][0]
        for nb in normal_blocks:
            res = i + np.arange(unpermuted_eq_dofs[nb].size * self.nd)
            i = res[-1] + 1
            eq_dofs_corrected[nb] = np.array(res)

        return eq_dofs_corrected

    def _correct_contact_equations_groups(
        self, equation_groups: list[list[int]], contact_group: int
    ) -> list[list[int]]:
        """PorePy provides 2 contact equation blocks for each fracture: normal and
        tangential. This merges them.

        """
        if len(equation_groups[contact_group]) == 0:
            return equation_groups

        eq_groups_corrected = [x.copy() for x in equation_groups]

        num_fracs = len(self.mdg.subdomains(dim=self.nd - 1))
        block_after_contact = max(equation_groups[contact_group]) + 1
        # Now each dof array in the contact group corresponds normal and tangential
        # components of contact relations on a specific fracture.
        eq_groups_corrected[contact_group] = equation_groups[contact_group][:num_fracs]

        # Since the number of groups decreased, we need to subtract the difference.
        for blocks in eq_groups_corrected:
            for i in range(len(blocks)):
                if blocks[i] >= block_after_contact:
                    blocks[i] -= num_fracs
        return eq_groups_corrected

    def Qright(self, contact_group: int, u_intf_group: int) -> BlockMatrixStorage:
        """Assemble the right linear transformation."""
        J = self.bmat
        # Sorted according to groups. If not done, the matrix can be in porepy order,
        # which does not guarantee that diagonal groups are truly on diagonals.
        Qright = J.empty_container()[:]

        if contact_group not in J.active_groups[0]:
            Qright.mat = csr_ones(Qright.shape[0])
            return Qright

        J55 = J[u_intf_group, u_intf_group].mat

        J55_inv = inv_block_diag(J55, nd=self.nd, lump=False)

        Qright.mat = csr_ones(Qright.shape[0])

        st, sl, op = self.sticking_sliding_open()
        st = np.repeat(st, self.nd)
        sl = np.repeat(sl, self.nd)
        op = np.repeat(op, self.nd)

        J54 = J[u_intf_group, contact_group].mat

        tmp = -J55_inv @ J54
        Qright[u_intf_group, contact_group] = tmp
        return Qright

    def Qleft(self, contact_group: int, u_intf_group: int) -> BlockMatrixStorage:
        """Assemble the left linear transformation."""
        J = self.bmat
        # Sorted according to groups. If not done, the matrix can be in porepy order,
        # which does not guarantee that diagonal groups are truly on diagonals.
        Qleft = J.empty_container()[:]

        if contact_group not in J.active_groups[0]:
            Qleft.mat = csr_ones(Qleft.shape[0])
            return Qleft

        J55_inv = inv_block_diag(
            J[u_intf_group, u_intf_group].mat, nd=self.nd, lump=False
        )
        # J55_inv = inv(J[u_intf_group, u_intf_group].mat)
        Qleft.mat = csr_ones(Qleft.shape[0])
        Qleft[contact_group, u_intf_group] = (
            -J[contact_group, u_intf_group].mat @ J55_inv
        )
        return Qleft

    def assemble_linear_system(self) -> None:
        super().assemble_linear_system()
        mat, rhs = self.linear_system

        # Apply the `contact_permutation`.
        mat = mat[self.contact_permutation]
        rhs = rhs[self.contact_permutation]
        self.bmat.mat = mat
        self.linear_system = mat, rhs

    def solve_linear_system(self) -> np.ndarray:
        rhs = self.linear_system[1]
        if not np.all(np.isfinite(rhs)):
            self._linear_solve_stats.krylov_iters = 0
            result = np.zeros_like(rhs)
            result[:] = np.nan
            return result

        solver_type = self.params["setup"]["solver"]
        direct = solver_type == 0
        if direct:
            return scipy.sparse.linalg.spsolve(*self.linear_system)
        else:
            return super().solve_linear_system()

    def make_solver_scheme(self) -> FieldSplitScheme:
        solver_type = self.params["setup"]["solver"]

        if solver_type == 2:  # GMRES + AMG
            return KSPScheme(
                ksp="gmres",
                rtol=1e-10,
                pc_side="right",
                right_transformations=[
                    lambda bmat: self.Qright(
                        contact_group=self.CONTACT_GROUP, u_intf_group=3
                    )
                ],
                preconditioner=FieldSplitScheme(
                    # Exactly eliminate contact mechanics (assuming linearly-transformed system)
                    groups=[0],
                    solve=lambda bmat: inv_block_diag(mat=bmat[[0]].mat, nd=self.nd),
                    complement=FieldSplitScheme(
                        # Eliminate interface flow,
                        # Use diag() to approximate inverse and ILU to solve linear systems
                        groups=[1],
                        solve=lambda bmat: PetscILU(bmat[[1]].mat),
                        invertor=lambda bmat: extract_diag_inv(bmat[[1]].mat),
                        complement=FieldSplitScheme(
                            # Eliminate elasticity. Use AMG to solve linear systems and fixed
                            # stress to approximate inverse.
                            groups=[2, 3],
                            solve=lambda bmat: PetscAMGMechanics(
                                mat=bmat[[2, 3]].mat,
                                dim=self.nd,
                                null_space=build_mechanics_near_null_space(self),
                            ),
                            invertor_type="physical",
                            invertor=lambda bmat: make_fs_analytical(
                                self, bmat, p_mat_group=4, p_frac_group=5
                            ).mat,
                            complement=FieldSplitScheme(
                                # Use AMG to solve mass balance.
                                groups=[4, 5],
                                solve=lambda bmat: PetscAMGFlow(mat=bmat[[4, 5]].mat),
                            ),
                        ),
                    ),
                ),
            )

        elif solver_type == 1:  # Same as sequential iterative scheme.
            return KSPScheme(
                # ksp="gmres",
                # rtol=1e-10,
                # pc_side="right",
                ksp="richardson",
                atol=1e-10,
                rtol=1e-10,
                pc_side="left",
                right_transformations=[],
                preconditioner=FieldSplitScheme(
                    groups=[0, 2, 3],
                    invertor_type="physical",
                    invertor=lambda bmat: make_fs_analytical_slow(
                        self, bmat, p_mat_group=4, p_frac_group=5, groups=[1, 4, 5]
                    ).mat,
                    complement=FieldSplitScheme(
                        groups=[1, 4, 5],
                    ),
                ),
            )

        elif solver_type == 21:  # GMRES + Direct subsolvers
            return KSPScheme(
                ksp="gmres",
                rtol=1e-10,
                pc_side="right",
                right_transformations=[
                    lambda bmat: self.Qright(
                        contact_group=self.CONTACT_GROUP, u_intf_group=3
                    )
                ],
                preconditioner=FieldSplitScheme(
                    groups=[0],
                    complement=FieldSplitScheme(
                        groups=[1],
                        complement=FieldSplitScheme(
                            groups=[2, 3],
                            invertor_type="physical",
                            invertor=lambda bmat: make_fs_analytical(
                                self, bmat, p_mat_group=4, p_frac_group=5
                            ).mat,
                            complement=FieldSplitScheme(
                                groups=[4, 5],
                            ),
                        ),
                    ),
                ),
            )

        else:
            raise ValueError


def make_reorder_contact(model: IterativeHMSolver, contact_group: int) -> np.ndarray:
    """Permutation of the contact mechanics equations. The PorePy arrangement is:
    `[C_n^0, C_n^1, ..., C_n^K, C_y^0, C_z^0, C_y^1, C_z^1, ..., C_z^K, C_z^k]`,
    where `C_n` is a normal component, `C_y` and `C_z` are two tangential
    components. Superscript corresponds to its position in space. We permute it to:
    `[C_n^0, C_y^0, C_z^0, ..., C_n^K, C_y^K, C_z^K]`, a.k.a array of structures.

    """
    reorder = np.arange(model.equation_system.num_dofs())
    if len(model.equation_groups[contact_group]) == 0:
        return reorder

    dofs_contact = np.concatenate(
        [model.eq_dofs[i] for i in model.equation_groups[contact_group]]
    )
    dofs_contact_start = dofs_contact[0]
    dofs_contact_end = dofs_contact[-1] + 1

    if model.nd == 2:
        dofs_contact_0 = dofs_contact[: len(dofs_contact) // model.nd]
        dofs_contact_1 = dofs_contact[len(dofs_contact) // model.nd :]
        reorder[dofs_contact_start:dofs_contact_end] = np.stack(
            [dofs_contact_0, dofs_contact_1]
        ).ravel("f")
    elif model.nd == 3:
        div = len(dofs_contact) // model.nd
        dofs_contact_0 = dofs_contact[:div]
        dofs_contact_1 = dofs_contact[div::2]
        dofs_contact_2 = dofs_contact[div + 1 :: 2]
        reorder[dofs_contact_start:dofs_contact_end] = np.stack(
            [dofs_contact_0, dofs_contact_1, dofs_contact_2]
        ).ravel("f")
    else:
        raise ValueError(f"{model.nd = }")
    return reorder


def build_mechanics_near_null_space(
    model: IterativeHMSolver, include_sd=True, include_intf=True
):
    cell_centers = []
    if include_sd:
        cell_centers.append(model.mdg.subdomains(dim=model.nd)[0].cell_centers)
    if include_intf:
        cell_centers.extend(
            [intf.cell_centers for intf in model.mdg.interfaces(dim=model.nd - 1)]
        )
    cell_centers = np.concatenate(cell_centers, axis=1)

    x, y, z = cell_centers
    num_dofs = cell_centers.shape[1]

    null_space = []
    if model.nd == 3:
        vec = np.zeros((3, num_dofs))
        vec[0] = 1
        null_space.append(vec.ravel("f"))
        vec = np.zeros((3, num_dofs))
        vec[1] = 1
        null_space.append(vec.ravel("f"))
        vec = np.zeros((3, num_dofs))
        vec[2] = 1
        null_space.append(vec.ravel("f"))
        # # 0, -z, y
        vec = np.zeros((3, num_dofs))
        vec[1] = -z
        vec[2] = y
        null_space.append(vec.ravel("f"))
        # z, 0, -x
        vec = np.zeros((3, num_dofs))
        vec[0] = z
        vec[2] = -x
        null_space.append(vec.ravel("f"))
        # -y, x, 0
        vec = np.zeros((3, num_dofs))
        vec[0] = -y
        vec[1] = x
        null_space.append(vec.ravel("f"))
    elif model.nd == 2:
        vec = np.zeros((2, num_dofs))
        vec[0] = 1
        null_space.append(vec.ravel("f"))
        vec = np.zeros((2, num_dofs))
        vec[1] = 1
        null_space.append(vec.ravel("f"))
        # -x, y
        vec = np.zeros((2, num_dofs))
        vec[0] = -x
        vec[1] = y
        null_space.append(vec.ravel("f"))
    else:
        raise ValueError

    return np.array(null_space)
