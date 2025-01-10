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

tmp = tmp[[0, 3]]

petsc_ksp, options = build_petsc_solver(
    bmat=tmp,
    scheme=PetscFieldSplitScheme(
        groups=[0],
        block_size=2,
        fieldsplit_options={
            # "pc_fieldsplit_schur_precondition": "a11",
            # 'pc_fieldsplit_schur_precondition': 'self',
            # "pc_fieldsplit_schur_precondition": "user",
            "pc_fieldsplit_schur_precondition": "selfp",
            # "pc_fieldsplit_schur_precondition": "full",
        },
        subsolver_options={},
        # invert=lambda _: csr_to_petsc(tmp[[3]].mat - tmp[3,0].mat @ inv_block_diag(tmp[0,0].mat, nd=model.nd) @ tmp[0,3].mat),
        # pcmat=lambda _: csr_to_petsc(inv_block_diag(tmp[0, 0].mat, nd=model.nd)),
        # tmp=csr_to_petsc(tmp[[3]].mat - tmp[3,0].mat @ inv_block_diag(tmp[0,0].mat, nd=model.nd) @ tmp[0,3].mat),
        complement=PetscFieldSplitScheme(
            groups=[3],
            fieldsplit_options={
                # 'mat_schur_complement_ainv_type': 'full',
                # 'mat_schur_complement_ainv_type': 'lump',
                # 'mat_schur_complement_ainv_type': 'blockdiag',
                # 'mat_schur_complement_ainv_type': 'diag',
                # 'mat_block_size': 1,
            },
            subsolver_options={
                "pc_type": "lu",
            },
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
