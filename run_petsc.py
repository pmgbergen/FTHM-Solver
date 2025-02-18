import numpy as np
from matplotlib import pyplot as plt
from plot_utils import *
from experiments.mandel_runscript_1 import make_model
import porepy as pp


model = make_model(
    {
        "physics": 1,
        "geometry": 0.2,
        "barton_bandis_stiffness_type": 2,
        "friction_type": 1,
        "grid_refinement": 1,
        "solver": 2,
        "high_boundary_pressure_ratio": 13,
    }
)
model.prepare_simulation()
model.before_nonlinear_loop()
model.before_nonlinear_iteration()
model.assemble_linear_system()
rhs = model.linear_system[1]
J = model.bmat

from full_petsc_solver import (
    PetscFieldSplitScheme,
    PetscKSPScheme,
    LinearTransformedScheme,
)
from fixed_stress import make_fs_analytical
from mat_utils import csr_to_petsc, inv, inv_block_diag


J = model.bmat[:]
scheme = LinearTransformedScheme(
    right_transformations=[lambda bmat: model.Qright(contact_group=0, u_intf_group=3)],
    inner=PetscKSPScheme(
        petsc_options={
            "ksp_monitor": None,
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-15,
            "ksp_max_it": 90,
            "ksp_gmres_restart": 30,
        },
        preconditioner=PetscFieldSplitScheme(
            groups=[0],
            block_size=model.nd,
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
                groups=[1],
                fieldsplit_options={
                    "pc_fieldsplit_schur_precondition": "selfp",
                },
                elim_options={
                    "pc_type": "ilu",
                },
                complement=PetscFieldSplitScheme(
                    groups=[2, 3],
                    block_size=model.nd,
                    invert=lambda bmat: csr_to_petsc(
                        make_fs_analytical(
                            model, bmat, p_mat_group=4, p_frac_group=5
                        ).mat,
                        bsize=1,
                    ),
                    # fieldsplit_options={
                    #     "pc_fieldsplit_schur_precondition": "selfp",
                    # },
                    elim_options={
                        "pc_type": "hypre",
                        "pc_hypre_type": "boomeramg",
                        "pc_hypre_boomeramg_strong_threshold": 0.7,
                    },
                    complement=PetscFieldSplitScheme(
                        groups=[4, 5],
                        elim_options={
                            # 'pc_type': 'lu'
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            # "pc_hypre_boomeramg_strong_threshold": 0.7,
                        },
                    ),
                ),
            ),
        ),
    ),
)

solver = scheme.make_solver(J)
solver.solve(J.project_rhs_to_local(rhs))
# solver.ksp.view()
print(solver.ksp.getConvergedReason())
