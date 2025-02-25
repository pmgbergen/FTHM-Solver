from functools import cached_property

import scipy.sparse
from block_matrix import (
    BlockMatrixStorage,
    FieldSplitScheme,
    KSPScheme,
    MultiStageScheme,
)
import numpy as np
from fixed_stress import make_fs_analytical_slow_new, make_fs_thermal
from full_petsc_solver import (
    LinearTransformedScheme,
    PcPythonPermutation,
    PetscCPRScheme,
    PetscCompositeScheme,
    PetscFieldSplitScheme,
    PetscKSPScheme,
)
from mat_utils import (
    BJacobiILU,
    PetscHypreILU,
    PetscSOR,
    RestrictedOperator,
    csr_to_petsc,
    extract_diag_inv,
    inv_block_diag,
    PetscAMGFlow,
    PetscAMGMechanics,
    PetscILU,
    make_scaling,
    make_scaling_1,
)
from hm_solver import (
    IterativeHMSolver,
    build_mechanics_near_null_space,
)
from iterative_solver import (
    get_equations_group_ids,
    get_variables_group_ids,
)


class THMSolver(IterativeHMSolver):

    def simulation_name(self) -> str:
        name = "stats_thermal"
        setup = self.params["setup"]
        name = f'{name}_geo{setup["geometry"]}x{setup["grid_refinement"]}'
        name = f'{name}_sol{setup["solver"]}'
        if (bc := setup.get("thermal_diffusion_bc")) not in ("dir", None):
            name = f"{name}_bc{bc}"
        return name

    CONTACT_GROUP = 0

    def group_row_names(self) -> list[str]:
        return [
            "Contact frac.",
            "Flow intf.",
            "Energy intf.",
            "Force mat.",
            "Force intf.",
            "Flow mat.",
            "Flow frac.",
            "Flow lower",
            "Energy mat.",
            "Energy frac.",
            "Energy lower",
        ]

    def group_col_names(self) -> list[str]:
        return [
            r"$\lambda_{frac}$",
            r"$v_{intf}$",
            "$T_{intf}$",
            r"$u_{3D}$",
            r"$u_{intf}$",
            r"$p_{3D}$",
            r"$p_{frac}$",
            "$p_{lower}$",
            "$T_{3D}$",
            "$T_{frac}$",
            "$T_{lower}$",
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
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        sd_lower = [
            k
            for i in reversed(range(0, dim_max - 1))
            for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()
        intf_frac = self.mdg.interfaces(dim=dim_max - 1)

        return get_variables_group_ids(
            model=self,
            md_variables_groups=[
                [self.contact_traction(sd_frac)],  # 0
                [self.interface_darcy_flux(intf)],  # 1
                [  # 2
                    self.interface_fourier_flux(intf),
                    self.interface_enthalpy_flux(intf),
                    self.well_enthalpy_flux(intf),
                ],
                [self.displacement(sd_ambient)],  # 3
                [self.interface_displacement(intf_frac)],  # 4
                [self.pressure(sd_ambient)],  # 5
                [self.pressure(sd_frac)],  # 6
                [self.pressure(sd_lower)],  # 7
                [self.temperature(sd_ambient)],  # 8
                [self.temperature(sd_frac)],  # 9
                [self.temperature(sd_lower)],  # 10
            ],
        )

    @cached_property
    def equation_groups(self) -> list[list[int]]:
        dim_max = self.mdg.dim_max()
        sd_ambient = self.mdg.subdomains(dim=dim_max)
        sd_frac = self.mdg.subdomains(dim=dim_max - 1)
        sd_lower = [
            k
            for i in reversed(range(0, dim_max - 1))
            for k in self.mdg.subdomains(dim=i)
        ]
        intf = self.mdg.interfaces()

        return self._correct_contact_equations_groups(
            get_equations_group_ids(
                model=self,
                equations_group_order=[
                    [  # 0
                        ("normal_fracture_deformation_equation", sd_frac),
                        ("tangential_fracture_deformation_equation", sd_frac),
                    ],
                    [("interface_darcy_flux_equation", intf)],  # 1
                    [  # 2
                        ("interface_fourier_flux_equation", intf),
                        ("interface_enthalpy_flux_equation", intf),
                        ("well_enthalpy_flux_equation", intf),  # ???
                    ],
                    [("momentum_balance_equation", sd_ambient)],  # 3
                    [("interface_force_balance_equation", intf)],  # 4
                    [("mass_balance_equation", sd_ambient)],  # 5
                    [("mass_balance_equation", sd_frac)],  # 6
                    [("mass_balance_equation", sd_lower)],  # 7
                    [("energy_balance_equation", sd_ambient)],  # 8
                    [("energy_balance_equation", sd_frac)],  # 9
                    [("energy_balance_equation", sd_lower)],  # 10
                ],
            ),
            contact_group=self.CONTACT_GROUP,
        )

    def scale_energy_balance(self, bmat: BlockMatrixStorage):
        res = bmat.empty_container()
        subdomains = [
            sd for d in reversed(range(self.nd)) for sd in self.mdg.subdomains(dim=d)
        ]
        diag = 1 / self.specific_volume(subdomains).value(self.equation_system)
        res.mat = scipy.sparse.eye(res.shape[0], format="csr")
        if len(subdomains) == 0:
            return res
        res[[9, 10]] = scipy.sparse.diags(diag)
        return res

    def make_solver_scheme(self) -> FieldSplitScheme:
        solver_type: str = self.params["setup"]["solver"]

        nd = self.nd
        contact = [0]
        intf = [1, 2]
        mech = [3, 4]
        flow = [5, 6, 7]
        temp = [8, 9, 10]

        pt_solver_cpr_global = PetscFieldSplitScheme(
            groups=flow,
            fieldsplit_options={
                "pc_fieldsplit_type": "additive",
            },
            elim_options={
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "pc_hypre_boomeramg_strong_threshold": 0.7,
            },
            complement=PetscFieldSplitScheme(
                groups=temp,
                elim_options={
                    "pc_type": "none",
                },
            ),
        )

        def make_2stage_scheme(global_prec):
            return PetscCompositeScheme(
                groups=flow + temp,
                solvers=[
                    global_prec,
                    PetscFieldSplitScheme(
                        groups=flow + temp,
                        python_pc=lambda bmat: PcPythonPermutation(
                            make_pt_permutation(bmat, p_groups=flow, t_groups=temp),
                            block_size=2,
                        ),
                        elim_options={
                            "python_pc_type": "ilu",
                            # "python_pc_type": "hypre",
                            # "python_pc_hypre_type": "Euclid",
                        },
                    ),
                ],
            )

        pt_solver_SAMG = PetscFieldSplitScheme(
            groups=flow + temp,
            python_pc=lambda bmat: PcPythonPermutation(
                make_pt_permutation(bmat, p_groups=flow, t_groups=temp),
                block_size=2,
            ),
            elim_options={
                "python_pc_type": "hypre",
                "python_pc_hypre_type": "boomeramg",
                "python_pc_hypre_boomeramg_strong_threshold": 0.7,
                "python_pc_hypre_boomeramg_P_max": 16,
                # "python_pc_type": "hmg",
                # "python_hmg_inner_pc_type": "hypre",
                # "python_hmg_inner_pc_hypre_type": "boomeramg",
                # "python_hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.7,
                # # "python_pc_hypre_boomeramg_P_max": 16,
                # "python_mg_levels_ksp_type": "richardson",
                # "python_mg_levels_ksp_max_it": 1,
            },
        )

        pt_solver_S4_diag = PetscFieldSplitScheme(
            groups=flow,
            elim_options={
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "hypre_boomeramg_strong_threshold": 0.7,
            },
            fieldsplit_options={
                "pc_fieldsplit_schur_precondition": "selfp",
            },
            complement=PetscFieldSplitScheme(
                groups=temp,
                elim_options={
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    "hypre_boomeramg_strong_threshold": 0.7,
                },
            ),
        )

        pt_solver_AAMG = PetscFieldSplitScheme(
            groups=flow,
            fieldsplit_options={
                "pc_fieldsplit_type": "additive",
            },
            elim_options={
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "pc_hypre_boomeramg_strong_threshold": 0.7,
            },
            complement=PetscFieldSplitScheme(
                groups=temp,
                elim_options={
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    "pc_hypre_boomeramg_strong_threshold": 0.7,
                },
            ),
        )

        if solver_type == "SAMG":
            pressure_temperature_solver = pt_solver_SAMG
        elif solver_type == "S4_diag":
            pressure_temperature_solver = pt_solver_S4_diag
        elif solver_type == "S4_Roy":
            raise NotImplementedError("S4_Roy")
        elif solver_type == "CPR" or (solver_type.find("+ILU") != -1):
            if solver_type == "CPR":
                global_prec = pt_solver_cpr_global
            elif solver_type.find("SAMG") != -1:
                global_prec = pt_solver_SAMG
            elif solver_type.find("S4_diag") != -1:
                global_prec = pt_solver_S4_diag
            elif solver_type.find("S4_Roy") != -1:
                raise NotImplementedError("S4_Roy")
            elif solver_type.find("AAMG") != -1:
                global_prec = pt_solver_AAMG
            else:
                raise ValueError(solver_type)
            pressure_temperature_solver = make_2stage_scheme(global_prec)
        else:
            raise ValueError(solver_type)

        return LinearTransformedScheme(
            right_transformations=[
                lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
            ],
            left_transformations=[
                lambda bmat: self.scale_energy_balance(bmat),
            ],
            inner=PetscKSPScheme(
                petsc_options={
                    # "ksp_type": "fgmres",
                    "ksp_monitor": None,
                    "ksp_rtol": 1e-12,
                },
                compute_eigenvalues=False,
                preconditioner=PetscFieldSplitScheme(
                    groups=contact,
                    block_size=self.nd,
                    fieldsplit_options={
                        "pc_fieldsplit_schur_precondition": "selfp",
                    },
                    elim_options={
                        "pc_type": "pbjacobi",
                    },
                    keep_options={
                        "mat_schur_complement_ainv_type": "blockdiag",
                    },
                    complement=PetscFieldSplitScheme(
                        groups=intf,
                        elim_options={
                            "pc_type": "ilu",
                        },
                        fieldsplit_options={
                            "pc_fieldsplit_schur_precondition": "selfp",
                        },
                        complement=PetscFieldSplitScheme(
                            groups=mech,
                            elim_options=(
                                {
                                    # "pc_type": "hypre",
                                    # "pc_hypre_type": "boomeramg",
                                    # "pc_hypre_boomeramg_strong_threshold": 0.7,
                                    # #
                                    # 'pc_hypre_boomeramg_max_row_sum': 1.0,
                                    # # "pc_hypre_boomeramg_smooth_type": "Euclid",
                                    "pc_type": "hmg",
                                    "hmg_inner_pc_type": "hypre",
                                    "hmg_inner_pc_hypre_type": "boomeramg",
                                    "hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.7,
                                    "mg_levels_ksp_type": "richardson",
                                    "mg_levels_ksp_max_it": 2,
                                    # 3D model has bad grid
                                    'mg_levels_pc_type': 'ilu' if nd == 3 else 'sor',
                                }
                            ),
                            keep_options={},
                            block_size=self.nd,
                            invert=lambda bmat: csr_to_petsc(
                                make_fs_analytical_slow_new(
                                    self,
                                    bmat,
                                    p_mat_group=5,
                                    p_frac_group=6,
                                    groups=flow + temp,
                                ).mat,
                                bsize=1,
                            ),
                            complement=pressure_temperature_solver,
                        ),
                    ),
                ),
            ),
        )

        # if solver_type in [1, 1.1, 1.2]:  # Direct subsolvers.
        #     if solver_type == 1:
        #         fs = 1
        #     elif solver_type == 1.1:
        #         fs = 0
        #     elif solver_type == 1.2:
        #         fs = 10
        #     else:
        #         raise ValueError
        #     return KSPScheme(
        #         ksp="richardson",
        #         right_transformations=[
        #             lambda bmat: self.Qright(
        #                 contact_group=self.CONTACT_GROUP, u_intf_group=4
        #             ),
        #             # lambda bmat: make_scaling(bmat),
        #             # lambda bmat: make_scaling_1(bmat, {6: [7], 9: [10]}),
        #         ],
        #         preconditioner=FieldSplitScheme(
        #             # Exactly eliminate contact mechanics (assuming linearly-transformed system)
        #             groups=[0],
        #             solve=lambda bmat: inv_block_diag(mat=bmat[[0]].mat, nd=self.nd),
        #             complement=FieldSplitScheme(
        #                 groups=[1],
        #                 solve=lambda bmat: PetscILU(bmat[[1]].mat),
        #                 invertor=lambda bmat: extract_diag_inv(bmat[[1]].mat),
        #                 complement=FieldSplitScheme(
        #                     groups=[2],
        #                     solve=lambda bmat: PetscILU(bmat[[2]].mat),
        #                     invertor=lambda bmat: extract_diag_inv(bmat[[2]].mat),
        #                     complement=FieldSplitScheme(
        #                         # Eliminate elasticity. Use AMG to solve linear systems and fixed
        #                         # stress to approximate inverse.
        #                         groups=[3, 4],
        #                         solve=lambda bmat: PetscAMGMechanics(
        #                             mat=bmat[[3, 4]].mat,
        #                             dim=self.nd,
        #                             null_space=build_mechanics_near_null_space(self),
        #                         ),
        #                         invertor_type="physical",
        #                         invertor=lambda bmat: make_fs_analytical_slow_new(
        #                             self,
        #                             bmat,
        #                             p_mat_group=5,
        #                             p_frac_group=6,
        #                             groups=[5, 6, 7, 8, 9, 10],
        #                         ).mat
        #                         * fs,
        #                         complement=MultiStageScheme(
        #                             # CPR for P-T coupling
        #                             groups=[5, 6, 7, 8, 9, 10],
        #                             stages=[
        #                                 lambda bmat: RestrictedOperator(
        #                                     bmat,
        #                                     solve_scheme=FieldSplitScheme(
        #                                         groups=[5, 6, 7],
        #                                         solve=lambda bmat: PetscAMGFlow(
        #                                             bmat.mat
        #                                         ),
        #                                     ),
        #                                 ),
        #                                 # lambda bmat: PetscSOR(bmat.mat),
        #                                 # lambda bmat: PetscHypreILU(bmat.mat),
        #                                 # lambda bmat: BJacobiILU(bmat),
        #                                 lambda bmat: PetscILU(bmat.mat),
        #                             ],
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )

        # elif solver_type == 2:  # Scalable solver.
        #     return KSPScheme(
        #         right_transformations=[
        #             lambda bmat: self.Qright(
        #                 contact_group=self.CONTACT_GROUP, u_intf_group=4
        #             ),
        #             # lambda bmat: make_scaling(bmat),
        #             # lambda bmat: make_scaling_1(bmat, {6: [7], 9: [10]}),
        #         ],
        #         preconditioner=FieldSplitScheme(
        #             # Exactly eliminate contact mechanics (assuming linearly-transformed system)
        #             groups=[0],
        #             solve=lambda bmat: inv_block_diag(mat=bmat[[0]].mat, nd=self.nd),
        #             complement=FieldSplitScheme(
        #                 groups=[1],
        #                 solve=lambda bmat: PetscILU(bmat[[1]].mat),
        #                 invertor=lambda bmat: extract_diag_inv(bmat[[1]].mat),
        #                 complement=FieldSplitScheme(
        #                     groups=[2],
        #                     solve=lambda bmat: PetscILU(bmat[[2]].mat),
        #                     invertor=lambda bmat: extract_diag_inv(bmat[[2]].mat),
        #                     complement=FieldSplitScheme(
        #                         # Eliminate elasticity. Use AMG to solve linear systems and fixed
        #                         # stress to approximate inverse.
        #                         groups=[3, 4],
        #                         solve=lambda bmat: PetscAMGMechanics(
        #                             mat=bmat[[3, 4]].mat,
        #                             dim=self.nd,
        #                             null_space=build_mechanics_near_null_space(self),
        #                         ),
        #                         invertor_type="physical",
        #                         invertor=lambda bmat: make_fs_analytical_slow_new(
        #                             self,
        #                             bmat,
        #                             p_mat_group=5,
        #                             p_frac_group=6,
        #                             groups=[5, 6, 7, 8, 9, 10],
        #                         ).mat,
        #                         complement=MultiStageScheme(
        #                             # CPR for P-T coupling
        #                             groups=[5, 6, 7, 8, 9, 10],
        #                             stages=[
        #                                 lambda bmat: RestrictedOperator(
        #                                     bmat,
        #                                     solve_scheme=FieldSplitScheme(
        #                                         groups=[5, 6, 7],
        #                                         solve=lambda bmat: PetscAMGFlow(
        #                                             bmat.mat
        #                                         ),
        #                                     ),
        #                                 ),
        #                                 # lambda bmat: PetscSOR(bmat.mat),
        #                                 # lambda bmat: PetscHypreILU(bmat.mat),
        #                                 # lambda bmat: BJacobiILU(bmat),
        #                                 lambda bmat: PetscILU(bmat.mat),
        #                             ],
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )

        # elif solver_type == 3:
        #     contact = [0]
        #     intf = [1, 2]
        #     mech = [3, 4]
        #     flow = [5, 6, 7]
        #     temp = [8, 9, 10]
        #     return LinearTransformedScheme(
        #         right_transformations=[
        #             lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
        #         ],
        #         inner=PetscKSPScheme(
        #             petsc_options={
        #                 "ksp_type": "fgmres",
        #                 # "ksp_monitor": None,
        #             },
        #             preconditioner=PetscFieldSplitScheme(
        #                 groups=contact,
        #                 block_size=self.nd,
        #                 fieldsplit_options={
        #                     "pc_fieldsplit_schur_precondition": "selfp",
        #                 },
        #                 elim_options={
        #                     "pc_type": "pbjacobi",
        #                 },
        #                 keep_options={
        #                     "mat_schur_complement_ainv_type": "blockdiag",
        #                 },
        #                 complement=PetscFieldSplitScheme(
        #                     groups=intf,
        #                     elim_options={
        #                         "pc_type": "ilu",
        #                     },
        #                     fieldsplit_options={
        #                         "pc_fieldsplit_schur_precondition": "selfp",
        #                     },
        #                     complement=PetscFieldSplitScheme(
        #                         groups=mech,
        #                         elim_options=(
        #                             {
        #                                 "ksp_type": "gmres",
        #                                 "ksp_rtol": 1e-2,
        #                                 "ksp_pc_side": "right",
        #                                 "ksp_monitor": None,
        #                                 # # old
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                 "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             }
        #                             if self.nd == 3
        #                             else {
        #                                 "ksp_type": "gmres",
        #                                 "ksp_rtol": 1e-2,
        #                                 "ksp_pc_side": "right",
        #                                 "ksp_monitor": None,
        #                                 # old
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.6,
        #                                 "pc_hypre_boomeramg_P_max": 1,
        #                                 "pc_hypre_boomeramg_max_iter": 1,
        #                                 "pc_hypre_boomeramg_cycle_type": "W",
        #                             }
        #                         ),
        #                         keep_options={
        #                             # "ksp_type": "gmres",
        #                             # "ksp_rtol": 1e-3,
        #                             # "ksp_pc_side": "right",
        #                             # "ksp_monitor": None,
        #                         },
        #                         block_size=self.nd,
        #                         invert=lambda bmat: csr_to_petsc(
        #                             make_fs_analytical_slow_new(
        #                                 self,
        #                                 bmat,
        #                                 p_mat_group=5,
        #                                 p_frac_group=6,
        #                                 groups=flow + temp,
        #                             ).mat,
        #                             bsize=1,
        #                         ),
        #                         complement=PetscCPRScheme(
        #                             groups=flow + temp,
        #                             pressure_groups=flow,
        #                             pressure_options={
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                 "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             },
        #                             others_options={
        #                                 "ksp_type": "preonly",
        #                                 "pc_type": "none",
        #                             },
        #                             cpr_options={
        #                                 # "pc_composite_pcs": "fieldsplit,ilu",
        #                                 "pc_composite_pcs": "fieldsplit,ksp",
        #                                 "sub_1_ksp_ksp_pc_side": "right",
        #                                 "sub_1_ksp_ksp_rtol": 1e-2,
        #                                 "sub_1_ksp_pc_type": "hypre",
        #                                 "sub_1_ksp_pc_hypre_type": "Euclid",
        #                                 "sub_1_ksp_ksp_monitor": None,
        #                                 "sub_1_ksp_ksp_max_it": 120,
        #                             },
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )

        # elif solver_type == 4:
        #     contact = [0]
        #     intf = [1, 2]
        #     mech = [3, 4]
        #     flow = [5, 6, 7]
        #     temp = [8, 9, 10]
        #     return LinearTransformedScheme(
        #         right_transformations=[
        #             lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
        #         ],
        #         inner=PetscKSPScheme(
        #             petsc_options={
        #                 # 'ksp_type': 'fgmres',
        #                 # "ksp_monitor": None,
        #             },
        #             preconditioner=PetscFieldSplitScheme(
        #                 groups=contact,
        #                 block_size=self.nd,
        #                 fieldsplit_options={
        #                     "pc_fieldsplit_schur_precondition": "selfp",
        #                 },
        #                 elim_options={
        #                     "pc_type": "pbjacobi",
        #                 },
        #                 keep_options={
        #                     "mat_schur_complement_ainv_type": "blockdiag",
        #                 },
        #                 complement=PetscFieldSplitScheme(
        #                     groups=intf,
        #                     elim_options={
        #                         "pc_type": "ilu",
        #                     },
        #                     fieldsplit_options={
        #                         "pc_fieldsplit_schur_precondition": "selfp",
        #                     },
        #                     complement=PetscFieldSplitScheme(
        #                         groups=mech,
        #                         elim_options={
        #                             # "pc_type": "hypre",
        #                             # "pc_hypre_type": "boomeramg",
        #                             # "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                             # # not sure:
        #                             # "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             "pc_type": "hypre",
        #                             "pc_hypre_type": "boomeramg",
        #                             "pc_hypre_boomeramg_strong_threshold": 0.6,
        #                             # not sure:
        #                             "pc_hypre_boomeramg_P_max": 1,
        #                             "pc_hypre_boomeramg_max_iter": 1,
        #                             "pc_hypre_boomeramg_cycle_type": "W",
        #                         },
        #                         block_size=self.nd,
        #                         invert=lambda bmat: csr_to_petsc(
        #                             make_fs_thermal(
        #                                 self,
        #                                 bmat,
        #                                 p_mat_group=5,
        #                                 p_frac_group=6,
        #                                 t_mat_group=8,
        #                                 t_frac_group=9,
        #                                 groups=flow + temp,
        #                             ).mat,
        #                             bsize=1,
        #                         ),
        #                         complement=PetscCPRScheme(
        #                             groups=flow + temp,
        #                             pressure_groups=flow,
        #                             pressure_options={
        #                                 "ksp_type": "preonly",
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                             },
        #                             others_options={
        #                                 "ksp_type": "preonly",
        #                                 "pc_type": "none",
        #                             },
        #                             cpr_options={
        #                                 "pc_composite_pcs": "fieldsplit,ilu",
        #                             },
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )

        # elif solver_type == 5:
        #     contact = [0]
        #     intf = [1, 2]
        #     mech = [3, 4]
        #     flow = [5, 6, 7]
        #     temp = [8, 9, 10]
        #     return LinearTransformedScheme(
        #         right_transformations=[
        #             lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
        #         ],
        #         inner=PetscKSPScheme(
        #             petsc_options={
        #                 # "ksp_type": "fgmres",
        #                 "ksp_monitor": None,
        #             },
        #             compute_eigenvalues=True,
        #             preconditioner=PetscFieldSplitScheme(
        #                 groups=contact,
        #                 block_size=self.nd,
        #                 fieldsplit_options={
        #                     "pc_fieldsplit_schur_precondition": "selfp",
        #                 },
        #                 elim_options={
        #                     "pc_type": "pbjacobi",
        #                 },
        #                 keep_options={
        #                     "mat_schur_complement_ainv_type": "blockdiag",
        #                 },
        #                 complement=PetscFieldSplitScheme(
        #                     groups=intf,
        #                     elim_options={
        #                         "pc_type": "ilu",
        #                     },
        #                     fieldsplit_options={
        #                         "pc_fieldsplit_schur_precondition": "selfp",
        #                     },
        #                     complement=PetscFieldSplitScheme(
        #                         groups=mech,
        #                         elim_options=(
        #                             {
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                 "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             }
        #                         ),
        #                         keep_options={},
        #                         block_size=self.nd,
        #                         invert=lambda bmat: csr_to_petsc(
        #                             make_fs_analytical_slow_new(
        #                                 self,
        #                                 bmat,
        #                                 p_mat_group=5,
        #                                 p_frac_group=6,
        #                                 groups=flow + temp,
        #                             ).mat,
        #                             bsize=1,
        #                         ),
        #                         complement=PetscCPRScheme(
        #                             groups=flow + temp,
        #                             pressure_groups=flow,
        #                             pressure_options={
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                 "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             },
        #                             others_options={
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                             },
        #                             cpr_options={
        #                                 "pc_composite_pcs": "fieldsplit,ksp",
        #                                 "sub_1_ksp_ksp_type": "preonly",
        #                                 "sub_1_ksp_pc_type": "hypre",
        #                                 "sub_1_ksp_pc_hypre_type": "Euclid",
        #                             },
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )

        # elif solver_type == 6:
        #     contact = [0]
        #     intf = [1, 2]
        #     mech = [3, 4]
        #     flow = [5, 6, 7]
        #     temp = [8, 9, 10]
        #     return LinearTransformedScheme(
        #         right_transformations=[
        #             lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
        #         ],
        #         inner=PetscKSPScheme(
        #             petsc_options={
        #                 # "ksp_type": "fgmres",
        #                 "ksp_monitor": None,
        #             },
        #             compute_eigenvalues=True,
        #             preconditioner=PetscFieldSplitScheme(
        #                 groups=contact,
        #                 block_size=self.nd,
        #                 fieldsplit_options={
        #                     "pc_fieldsplit_schur_precondition": "selfp",
        #                 },
        #                 elim_options={
        #                     "pc_type": "pbjacobi",
        #                 },
        #                 keep_options={
        #                     "mat_schur_complement_ainv_type": "blockdiag",
        #                 },
        #                 complement=PetscFieldSplitScheme(
        #                     groups=intf,
        #                     elim_options={
        #                         "pc_type": "ilu",
        #                     },
        #                     fieldsplit_options={
        #                         "pc_fieldsplit_schur_precondition": "selfp",
        #                     },
        #                     complement=PetscFieldSplitScheme(
        #                         groups=mech,
        #                         elim_options=(
        #                             {
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                 "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             }
        #                         ),
        #                         keep_options={},
        #                         block_size=self.nd,
        #                         invert=lambda bmat: csr_to_petsc(
        #                             make_fs_analytical_slow_new(
        #                                 self,
        #                                 bmat,
        #                                 p_mat_group=5,
        #                                 p_frac_group=6,
        #                                 groups=flow + temp,
        #                             ).mat,
        #                             bsize=1,
        #                         ),
        #                         complement=PetscCompositeScheme(
        #                             groups=flow + temp,
        #                             solvers=[
        #                                 PetscFieldSplitScheme(
        #                                     groups=flow,
        #                                     fieldsplit_options={
        #                                         "pc_fieldsplit_type": "additive",
        #                                     },
        #                                     elim_options={
        #                                         "pc_type": "hypre",
        #                                         "pc_hypre_type": "boomeramg",
        #                                         "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                         "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                                     },
        #                                     complement=PetscFieldSplitScheme(
        #                                         groups=temp,
        #                                         elim_options={
        #                                             "pc_type": "none",
        #                                         },
        #                                     ),
        #                                 ),
        #                                 PetscFieldSplitScheme(
        #                                     groups=flow + temp,
        #                                     python_pc=lambda bmat: PcPythonPermutation(
        #                                         make_pt_permutation(bmat), block_size=2
        #                                     ),
        #                                     elim_options={
        #                                         "python_pc_type": "hypre",
        #                                         "python_pc_hypre_type": "euclid",
        #                                         "python_pc_hypre_euclid_bj": True,
        #                                     },
        #                                 ),
        #                             ],
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )

        # elif solver_type == 7:
        #     contact = [0]
        #     intf = [1, 2]
        #     mech = [3, 4]
        #     flow = [5, 6, 7]
        #     temp = [8, 9, 10]
        #     return LinearTransformedScheme(
        #         right_transformations=[
        #             lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
        #         ],
        #         inner=PetscKSPScheme(
        #             petsc_options={
        #                 # "ksp_type": "fgmres",
        #                 "ksp_monitor": None,
        #             },
        #             compute_eigenvalues=True,
        #             preconditioner=PetscFieldSplitScheme(
        #                 groups=contact,
        #                 block_size=self.nd,
        #                 fieldsplit_options={
        #                     "pc_fieldsplit_schur_precondition": "selfp",
        #                 },
        #                 elim_options={
        #                     "pc_type": "pbjacobi",
        #                 },
        #                 keep_options={
        #                     "mat_schur_complement_ainv_type": "blockdiag",
        #                 },
        #                 complement=PetscFieldSplitScheme(
        #                     groups=intf,
        #                     elim_options={
        #                         "pc_type": "ilu",
        #                     },
        #                     fieldsplit_options={
        #                         "pc_fieldsplit_schur_precondition": "selfp",
        #                     },
        #                     complement=PetscFieldSplitScheme(
        #                         groups=mech,
        #                         elim_options=(
        #                             {
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                 "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             }
        #                         ),
        #                         keep_options={},
        #                         block_size=self.nd,
        #                         invert=lambda bmat: csr_to_petsc(
        #                             make_fs_analytical_slow_new(
        #                                 self,
        #                                 bmat,
        #                                 p_mat_group=5,
        #                                 p_frac_group=6,
        #                                 groups=flow + temp,
        #                             ).mat,
        #                             bsize=1,
        #                         ),
        #                         complement=PetscCompositeScheme(
        #                             groups=flow + temp,
        #                             solvers=[
        #                                 PetscFieldSplitScheme(
        #                                     groups=flow,
        #                                     fieldsplit_options={
        #                                         "pc_fieldsplit_type": "additive",
        #                                     },
        #                                     elim_options={
        #                                         "pc_type": "hypre",
        #                                         "pc_hypre_type": "boomeramg",
        #                                         "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                         "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                                     },
        #                                     complement=PetscFieldSplitScheme(
        #                                         groups=temp,
        #                                         elim_options={
        #                                             "pc_type": "none",
        #                                         },
        #                                     ),
        #                                 ),
        #                                 PetscFieldSplitScheme(
        #                                     groups=flow + temp,
        #                                     # python_pc=lambda bmat: PcPythonPermutation(
        #                                     #     make_pt_permutation(bmat), block_size=2
        #                                     # ),
        #                                     elim_options={
        #                                         "pc_type": "hypre",
        #                                         "pc_hypre_type": "euclid",
        #                                         "pc_hypre_euclid_bj": True,
        #                                     },
        #                                 ),
        #                             ],
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )
        # elif solver_type == 8:
        #     return LinearTransformedScheme(
        #         right_transformations=[
        #             lambda bmat: self.Qright(contact_group=0, u_intf_group=4)
        #         ],
        #         inner=PetscKSPScheme(
        #             petsc_options={
        #                 # "ksp_type": "fgmres",
        #                 "ksp_monitor": None,
        #             },
        #             compute_eigenvalues=True,
        #             preconditioner=PetscFieldSplitScheme(
        #                 groups=contact,
        #                 block_size=self.nd,
        #                 fieldsplit_options={
        #                     "pc_fieldsplit_schur_precondition": "selfp",
        #                 },
        #                 elim_options={
        #                     "pc_type": "pbjacobi",
        #                 },
        #                 keep_options={
        #                     "mat_schur_complement_ainv_type": "blockdiag",
        #                 },
        #                 complement=PetscFieldSplitScheme(
        #                     groups=intf,
        #                     elim_options={
        #                         "pc_type": "ilu",
        #                     },
        #                     fieldsplit_options={
        #                         "pc_fieldsplit_schur_precondition": "selfp",
        #                     },
        #                     complement=PetscFieldSplitScheme(
        #                         groups=mech,
        #                         elim_options=(
        #                             {
        #                                 "pc_type": "hypre",
        #                                 "pc_hypre_type": "boomeramg",
        #                                 "pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                 "pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             }
        #                         ),
        #                         keep_options={},
        #                         block_size=self.nd,
        #                         invert=lambda bmat: csr_to_petsc(
        #                             make_fs_analytical_slow_new(
        #                                 self,
        #                                 bmat,
        #                                 p_mat_group=5,
        #                                 p_frac_group=6,
        #                                 groups=flow + temp,
        #                             ).mat,
        #                             bsize=1,
        #                         ),
        #                         complement=PetscFieldSplitScheme(
        #                             groups=flow + temp,
        #                             python_pc=lambda bmat: PcPythonPermutation(
        #                                 make_pt_permutation(bmat), block_size=2
        #                             ),
        #                             elim_options={
        #                                 "python_pc_type": "hypre",
        #                                 "python_pc_hypre_type": "boomeramg",
        #                                 "python_pc_hypre_boomeramg_nodal_coarsen": 1,
        #                                 "python_pc_hypre_boomeramg_vec_interp_variant": 1,
        #                                 "python_pc_hypre_boomeramg_strong_threshold": 0.7,
        #                                 "python_pc_hypre_boomeramg_smooth_type": "Euclid",
        #                             },
        #                         ),
        #                     ),
        #                 ),
        #             ),
        #         ),
        #     )

        # raise ValueError(f"{solver_type}")


def get_dofs_of_groups(
    groups_to_block: list[list[int]], dofs: list[np.ndarray], groups: list[int]
) -> np.ndarray:
    blocks = [blk for g in groups for blk in groups_to_block[g]]
    return np.concatenate([dofs[blk] for blk in blocks])


def make_pt_permutation(
    J: BlockMatrixStorage, p_groups: list[int], t_groups: list[int]
):
    J = J[p_groups + t_groups]
    t_dofs = get_dofs_of_groups(
        groups_to_block=J.groups_to_blocks_row, dofs=J.local_dofs_row, groups=t_groups
    )
    p_dofs = get_dofs_of_groups(
        groups_to_block=J.groups_to_blocks_row, dofs=J.local_dofs_row, groups=p_groups
    )
    return np.row_stack([p_dofs, t_dofs]).ravel("F")
