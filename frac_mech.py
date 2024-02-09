# %%

from numpy import ndarray
import porepy as pp
from matplotlib import pyplot as plt
import numpy as np
from porepy.grids.boundary_grid import BoundaryGrid
from porepy.grids.grid import Grid
from porepy.models.momentum_balance import (
    MomentumBalance,
    BoundaryConditionsMomentumBalance,
)
from porepy.viz.diagnostics_mixin import DiagnosticsMixin
from porepy.applications.md_grids import fracture_sets
from porepy.models.geometry import ModelGeometry
from porepy.params.bc import BoundaryConditionVectorial


class MyGeometry(ModelGeometry):
    def set_fractures(self) -> None:
        # self._fractures = fracture_sets.benchmark_2d_case_3(size=0.9)
        # self._fractures = fracture_sets.orthogonal_fractures_2d(size=0.7)
        self._fractures = [pp.LineFracture([[0.9, 0.1], [0.1, 0.9]])]


class MyBC(BoundaryConditionsMomentumBalance):
    def bc_type_mechanics(self, sd: Grid) -> BoundaryConditionVectorial:
        boundary_faces = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces.south, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the fracture
        # faces.
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: BoundaryGrid) -> ndarray:
        stress = np.zeros((self.nd, boundary_grid.num_cells))
        boundary_faces = self.domain_boundary_sides(boundary_grid)
        stress[0, boundary_faces.north] = 0.001
        return stress.ravel("F")


class FracMechModel(DiagnosticsMixin, MyGeometry, MyBC, MomentumBalance):
    pass


solid_constants = pp.SolidConstants(
    {"residual_aperture": 1e-4, "normal_permeability": 1e-4}
)
params = {
    "fracture_permeability": 1e-4,
    "material_constants": {"solid": solid_constants},
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 1 / 16},
}

model = FracMechModel(params)
pp.run_time_dependent_model(model, params)

# %%
pp.plot_grid(
    model.mdg,
    # vector_value=model.displacement_variable,
    plot_2d=True,
    fracturewidth_1d=3,
    rgb=[0.5, 0.5, 1],
)

# %%
diagnostics = model.run_diagnostics()
model.plot_diagnostics(diagnostics, key="max")  # %%

# %%
block = diagnostics[3, 2]
col = block["block_dofs_col"]
row = block["block_dofs_row"]

full_mat, rhs = model.linear_system
C, R = np.meshgrid(col, row, indexing="ij", sparse=True)

A = full_mat[C, R]

plt.matshow(A.A)
plt.colorbar()
# %%
