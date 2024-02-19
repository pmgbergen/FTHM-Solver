import scipy.sparse
from mat_utils import *
from plot_utils import *

from fpm.fpm_1 import make_model
from pp_utils import make_block_mat

# Bp = scipy.sparse.load_npz("Bp.npz")
# solve_petsc(Bp, PetscAMGMechanics(mat=Bp, dim=2), label="B amg")
# plt.show()

for scale in [1/10, 1/20, 1/40, 1/80]:
    model = make_model(scale)
    model.prepare_simulation()
    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    mat, rhs = model.linear_system
    model._initialize_solver()
    block_matrix = make_block_mat(model, mat)
    eq_blocks = model.make_equations_groups()
    var_blocks = model.make_variables_groups()

    _, prec = model._prepare_solver()
    B_amg = prec.Omega_inv.S_A_inv
    B = B_amg.get_matrix()
    solve_petsc(B.copy(), B_amg, label=str(scale))

plt.legend()
plt.show()