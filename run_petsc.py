import numpy as np
from matplotlib import pyplot as plt
from plot_utils import *
from experiments.mandel_runscript_2 import make_model
import porepy as pp


model = make_model(
    {
        "physics": 1,
        "geometry": 0.2,
        "barton_bandis_stiffness_type": 2,
        "friction_type": 1,
        "grid_refinement": 1,
        "solver": 2,
    }
)
model.prepare_simulation()
model.before_nonlinear_loop()
model.before_nonlinear_iteration()
model.assemble_linear_system()
rhs = model.linear_system[1]
J = model.bmat

from full_petsc_solver import PetscFieldSplitScheme, build_petsc_solver
from fixed_stress import make_fs_analytical
from mat_utils import csr_to_petsc, inv, inv_block_diag


J = model.bmat[[0, 1, 2, 3, 4, 5]]
Qr = model.Qright(contact_group=0, u_intf_group=3)[[0, 1, 2, 3, 4, 5]]
tmp = J.empty_container()
tmp.mat = J.mat @ Qr.mat

tmp = tmp[:]

petsc_ksp, options = build_petsc_solver(
    bmat=tmp,
    scheme=PetscFieldSplitScheme(
        groups=[0],
        block_size=model.nd,
        fieldsplit_options={
            "pc_fieldsplit_schur_precondition": "selfp",
        },
        subsolver_options={
            "pc_type": "ilu",
        },
        tmp_options={
            "mat_schur_complement_ainv_type": "blockdiag",
        },
        complement=PetscFieldSplitScheme(
            groups=[1],
            fieldsplit_options={
                "pc_fieldsplit_schur_precondition": "selfp",
            },
            subsolver_options={
                "pc_type": "ilu",
            },
            complement=PetscFieldSplitScheme(
                groups=[2, 3],
                block_size=model.nd,
                invert=lambda _: csr_to_petsc(
                    make_fs_analytical(model, tmp, p_mat_group=4, p_frac_group=5).mat,
                    bsize=1,
                ),
                # fieldsplit_options={
                #     "pc_fieldsplit_schur_precondition": "selfp",
                # },
                subsolver_options={
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    "pc_hypre_boomeramg_strong_threshold": 0.7,
                },
                complement=PetscFieldSplitScheme(
                    groups=[4, 5],
                    subsolver_options={
                        "pc_type": "hypre",
                        "pc_hypre_type": "boomeramg",
                        "pc_hypre_boomeramg_strong_threshold": 0.7,
                    },
                ),
            ),
        ),
    ),
)
for k, v in options.items():
    print(k, v)

petsc_ksp.view()

rhs_local = tmp.project_rhs_to_local(rhs)

petsc_mat = petsc_ksp.getOperators()[0]
petsc_rhs = petsc_mat.createVecLeft()
petsc_x0 = petsc_mat.createVecLeft()
petsc_rhs.setArray(rhs_local)
petsc_x0.set(0.0)

petsc_ksp.solve(petsc_rhs, petsc_x0)

print(petsc_ksp.getConvergedReason())
