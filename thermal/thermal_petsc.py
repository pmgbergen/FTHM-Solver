import numpy as np
from matplotlib import pyplot as plt
from fixed_stress import make_fs_analytical_slow_new
from full_petsc_solver import (
    PetscKrylovSolver,
    construct_is,
    insert_petsc_options,
    PETSc,
)
from mat_utils import csr_to_petsc
from plot_utils import *
from thermal.thermal_runscript_2 import make_model
import porepy as pp


model = make_model(
    {
        "geometry": 0.2,
        "barton_bandis_stiffness_type": 2,
        "friction_type": 1,
        "grid_refinement": 1,
        "solver": 3,
    }
)
model.prepare_simulation()
model.before_nonlinear_loop()
model.before_nonlinear_iteration()
model.assemble_linear_system()
rhs = model.linear_system[1]


contact = [0]
intf = [1, 2]
mech = [3, 4]
flow = [5, 6, 7]
temp = [8, 9, 10]

# J = model.bmat[contact + intf + mech + flow + temp]

# scheme = LinearTransformedScheme(
#     right_transformations=[
#         lambda bmat: model.Qright(contact_group=0, u_intf_group=4)
#     ],
#     inner=PetscKSPScheme(
#         petsc_options={
#             # 'ksp_type': 'fgmres',
#             "ksp_monitor": None,
#         },
#         preconditioner=PetscFieldSplitScheme(
#             groups=contact,
#             block_size=model.nd,
#             fieldsplit_options={
#                 "pc_fieldsplit_schur_precondition": "selfp",
#             },
#             subsolver_options={
#                 "pc_type": "pbjacobi",
#             },
#             tmp_options={
#                 "mat_schur_complement_ainv_type": "blockdiag",
#             },
#             complement=PetscFieldSplitScheme(
#                 groups=intf,
#                 subsolver_options={
#                     "pc_type": "ilu",
#                 },
#                 fieldsplit_options={
#                     "pc_fieldsplit_schur_precondition": "selfp",
#                 },
#                 complement=PetscFieldSplitScheme(
#                     groups=mech,
#                     subsolver_options={
#                         "pc_type": "hypre",
#                         "pc_hypre_type": "boomeramg",
#                         "pc_hypre_boomeramg_strong_threshold": 0.7,
#                     },
#                     block_size=model.nd,
#                     invert=lambda bmat: csr_to_petsc(
#                         make_fs_analytical_slow_new(
#                             model,
#                             bmat,
#                             p_mat_group=5,
#                             p_frac_group=6,
#                             groups=flow + temp,
#                         ).mat,
#                         bsize=1,
#                     ),
#                     complement=PetscCPRScheme(
#                         groups=flow + temp,
#                         pressure_groups=flow,
#                         pressure_options={
#                             "ksp_type": "preonly",
#                             "pc_type": "hypre",
#                             "pc_hypre_type": "boomeramg",
#                         },
#                         others_options={
#                             "ksp_type": "preonly",
#                             "pc_type": "none",
#                         },
#                         cpr_options={
#                             'pc_composite_pcs': 'fieldsplit,jacobi',
#                         }
#                     ),
#                 ),
#             ),
#         ),
#     ),
# )
# solver = scheme.make_solver(J)

# rhs_local = J.project_rhs_to_local(rhs)
# solver.solve(rhs_local)

# print(solver.ksp.getConvergedReason())
# # solver.ksp.view()


J = model.bmat[[8, 9, 10]]


ksp = PETSc.KSP().create()
insert_petsc_options(
    {
        "ksp_monitor": None,
        "ksp_type": "gmres",
        "ksp_pc_side": "right",
        'pc_type': 'ilu',
        # "pc_type": "fieldsplit",
        # "pc_fieldsplit_type": "additive",
        # "fieldsplit_8_ksp_type": "preonly",
        # "fieldsplit_8_pc_type": "ilu",
        # "fieldsplit_9_ksp_type": "preonly",
        # "fieldsplit_9_pc_type": "ilu",
        # "fieldsplit_10_ksp_type": "preonly",
        # "fieldsplit_10_pc_type": "ilu",
    }
)
petsc_mat = csr_to_petsc(J.mat)
ksp.setFromOptions()
ksp.setOperators(petsc_mat, petsc_mat)
petsc_pc = ksp.getPC()
# petsc_is_8 = construct_is(J, [8])
# petsc_is_9 = construct_is(J, [9])
# petsc_is_10 = construct_is(J, [10])
# petsc_pc.setFieldSplitIS(("8", petsc_is_8), ("9", petsc_is_9), ("10", petsc_is_10))
ksp.setUp()

solver = PetscKrylovSolver(ksp)
solver.solve(J.project_rhs_to_local(rhs))
print(solver.ksp.getConvergedReason())

# solver.ksp.view()