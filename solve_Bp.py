import scipy.sparse
from mat_utils import PetscAMGMechanics
from plot_utils import solve_petsc, plt

Bp = scipy.sparse.load_npz("Bp.npz")
solve_petsc(Bp, PetscAMGMechanics(mat=Bp, dim=2), label="B amg")
plt.show()
