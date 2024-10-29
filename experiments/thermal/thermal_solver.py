from functools import cached_property
from block_matrix import BlockMatrixStorage, FieldSplitScheme
from fixed_stress import make_fs_analytical
from mat_utils import (
    extract_diag_inv,
    inv_block_diag,
    PetscAMGFlow,
    PetscAMGFlow,
    PetscAMGMechanics,
    PetscILU,
)
from pp_utils import (
    MyPetscSolver,
    build_mechanics_near_null_space,
    get_equations_group_ids,
    get_variables_group_ids,
)
import porepy as pp


class ThermalSolver(MyPetscSolver):
    def assemble_linear_system(self) -> None:
        pp.SolutionStrategy.assemble_linear_system(self)  # TODO
        mat, rhs = self.linear_system
        # Applies the `contact_permutation`.
        if len(self.equation_groups[4]) != 0:
            mat = mat[self.contact_permutation]
            rhs = rhs[self.contact_permutation]
            self.linear_system = mat, rhs

            bmat = BlockMatrixStorage(
                mat=mat,
                global_row_idx=self.eq_dofs,
                global_col_idx=self.var_dofs,
                groups_row=self.equation_groups,
                groups_col=self.variable_groups,
                group_row_names=[
                    "Flow mat.",
                    "Force mat.",
                    "Flow frac.",
                    "Flow intf.",
                    "Contact frac.",
                    "Force intf.",
                    "Energy mat.",
                    "Energy frac.",
                    "Energy intf.",
                ],
                group_col_names=[
                    r"$p_{3D}$",
                    r"$u_{3D}$",
                    r"$p_{frac}$",
                    r"$v_{intf}$",
                    r"$\lambda_{frac}$",
                    r"$u_{intf}$",
                    "$T_{3D}$",
                    "$T_{frac}$",
                    "$T_{intf}$",
                ],
            )

            # Reordering the matrix to the order I work with, not how PorePy provides
            # it. This is important, the solver relies on it.
            bmat = bmat[:]
            self.bmat = bmat

    @cached_property
    def variable_groups(model) -> list[list[int]]:
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
        dim_max = model.mdg.dim_max()
        sd_ambient = model.mdg.subdomains(dim=dim_max)
        sd_lower = [
            k for i in reversed(range(0, dim_max)) for k in model.mdg.subdomains(dim=i)
        ]
        sd_frac = model.mdg.subdomains(dim=dim_max - 1)
        intf = model.mdg.interfaces()
        intf_frac = model.mdg.interfaces(dim=dim_max - 1)

        return get_variables_group_ids(
            model=model,
            md_variables_groups=[
                [model.pressure(sd_ambient)],  # 0
                [model.displacement(sd_ambient)],  # 1
                [model.pressure(sd_lower)],  # 2
                [model.interface_darcy_flux(intf)],  # 3
                [model.contact_traction(sd_frac)],  # 4
                [model.interface_displacement(intf_frac)],  # 5
                [model.temperature(sd_ambient)],  # 6
                [model.temperature(sd_lower)],  # 7
                [  # 8
                    model.interface_fourier_flux(intf),
                    model.interface_enthalpy_flux(intf),
                    model.well_enthalpy_flux(intf),
                ],
            ],
        )

    @cached_property
    def _unpermuted_equation_groups(self) -> list[list[int]]:
        """The version of `equation_groups` that does not encorporates the permutation
        `contact_permutation`.

        """
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_lower = [
            k for i in reversed(range(0, dim_max)) for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()

        return get_equations_group_ids(
            model=self,
            equations_group_order=[
                [("mass_balance_equation", sd_ambient)],  # 0
                [("momentum_balance_equation", sd_ambient)],  # 1
                [("mass_balance_equation", sd_lower)],  # 2
                [("interface_darcy_flux_equation", intf)],  # 3
                [  # 4
                    ("normal_fracture_deformation_equation", sd_lower),
                    ("tangential_fracture_deformation_equation", sd_lower),
                ],
                [("interface_force_balance_equation", intf)],  # 5
                [("energy_balance_equation", sd_ambient)],  # 6
                [("energy_balance_equation", sd_lower)],  # 7
                [  # 8
                    ("interface_fourier_flux_equation", intf),
                    ("interface_enthalpy_flux_equation", intf),
                    ("well_enthalpy_flux_equation", sd_lower),
                ],
            ],
        )

    def make_solver_schema(self) -> FieldSplitScheme:
        solver_type = self.params["setup"]["solver"]

        if solver_type == 2:  # Scalable solver.
            return FieldSplitScheme(
                # Exactly eliminate contact mechanics (assuming linearly-transformed system)
                groups=[4],
                solve=lambda bmat: inv_block_diag(mat=bmat[[4]].mat, nd=self.nd),
                complement=FieldSplitScheme(
                    # Eliminate interface flow, it is not coupled with (1, 4, 5)
                    # Use diag() to approximate inverse and ILU to solve linear systems
                    groups=[3],
                    solve=lambda bmat: PetscILU(bmat[[3]].mat),
                    invertor=lambda bmat: extract_diag_inv(bmat[[3]].mat),
                    complement=FieldSplitScheme(
                        # Eliminate elasticity. Use AMG to solve linear systems and fixed
                        # stress to approximate inverse.
                        groups=[1, 5],
                        solve=lambda bmat: PetscAMGMechanics(
                            mat=bmat[[1, 5]].mat,
                            dim=self.nd,
                            null_space=build_mechanics_near_null_space(self),
                        ),
                        invertor_type="physical",
                        invertor=lambda bmat: make_fs_analytical(self, bmat).mat,
                        complement=FieldSplitScheme(
                            # Use AMG to solve mass balance.
                            groups=[0, 2],
                            solve=lambda bmat: PetscAMGFlow(mat=bmat[[0, 2]].mat),
                            complement=FieldSplitScheme(
                                groups=[8],
                                solve=lambda bmat: inv_block_diag(
                                    bmat[[8]].mat, nd=self.nd, lump=True
                                ),
                                complement=FieldSplitScheme(
                                    groups=[6, 7],
                                    solve=lambda bmat: PetscAMGFlow(bmat[[6, 7]].mat),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        raise ValueError(f"{solver_type}")
