# %%

from numpy import ndarray
import porepy as pp
from matplotlib import pyplot as plt
import numpy as np
from porepy.grids.boundary_grid import BoundaryGrid
from porepy.grids.grid import Grid
from porepy.models.poromechanics import BoundaryConditionsPoromechanics, Poromechanics
from porepy.viz.diagnostics_mixin import DiagnosticsMixin
from porepy.applications.md_grids import fracture_sets
from porepy.models.geometry import ModelGeometry
from porepy.params.bc import BoundaryCondition, BoundaryConditionVectorial
from porepy.applications.material_values.fluid_values import water
from porepy.applications.material_values.solid_values import granite
from porepy.models.constitutive_laws import CubicLawPermeability
from porepy.applications.md_grids.domains import nd_cube_domain


class MyGeometry(ModelGeometry):
    def set_domain(self) -> None:
        """Set domain of the problem.

        Defaults to a 2d unit square domain.
        Override this method to define a geometry with a different domain.

        """
        self._domain = nd_cube_domain(2, 1.0)

    def set_fractures(self) -> None:
        # self._fractures = []
        # self._fractures = fracture_sets.benchmark_2d_case_3(size=0.9)
        # self._fractures = fracture_sets.orthogonal_fractures_2d(size=0.7)
        # self._fractures = [pp.LineFracture([[0.9, 0.1], [0.1, 0.9]])]
        # self._fractures = [pp.LineFracture([[0.1, 0.9], [0.1, 0.9]])]
        self._fractures = [pp.LineFracture([[0.1, 0.5], [0.1, 0.5]])]


class MyBC(BoundaryConditionsPoromechanics):
    def bc_type_mechanics(self, sd: Grid) -> BoundaryConditionVectorial:
        boundary_faces = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, boundary_faces.south + boundary_faces.west, "dir"
        )
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: BoundaryGrid) -> ndarray:
        stress = np.zeros((self.nd, boundary_grid.num_cells))
        boundary_faces = self.domain_boundary_sides(boundary_grid)
        stress[0, boundary_faces.north] = self.solid.convert_units(2 * 1e8, 'kg * m^-1 * s^-2')
        stress[1, boundary_faces.east] = self.solid.convert_units(1 * 1e8, 'kg * m^-1 * s^-2')
        return stress.ravel("F")

    def bc_type_fluid_flux(self, sd: Grid) -> BoundaryCondition:
        boundary_faces = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(
            sd, boundary_faces.west + boundary_faces.east + boundary_faces.north, "dir"
        )

    def bc_type_darcy_flux(self, sd: Grid) -> BoundaryCondition:
        boundary_faces = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(
            sd, boundary_faces.east + boundary_faces.north, "dir"
        )

    def bc_values_darcy_flux(self, boundary_grid: BoundaryGrid) -> ndarray:
        values = np.zeros(boundary_grid.num_cells)
        boundary_faces = self.domain_boundary_sides(boundary_grid)
        values[boundary_faces.west] = self.solid.convert_units(-1e6, "Pa*m^-1")
        return values


class FracPoroMechModel(
    DiagnosticsMixin, CubicLawPermeability, MyGeometry, MyBC, Poromechanics
):
    def after_nonlinear_convergence(
        self, solution: ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        self.time_manager.compute_time_step(iteration_counter, recompute_solution=False)

    def after_nonlinear_failure(
        self, solution: ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.time_manager.compute_time_step(recompute_solution=True)


def make_model():
    solid_constants = pp.SolidConstants(
        {"residual_aperture": 1e-4, "normal_permeability": 1e-4} | granite
    )

    fluid_constants = pp.FluidConstants(water)

    time_manager = pp.TimeManager(
        dt_init=1e0, dt_min_max=(1e-2, 1e2), schedule=[0, 100]
    )
    units = pp.Units(m=1e-6, kg=1e6)
    # units = pp.Units()

    params = {
        "material_constants": {"solid": solid_constants, "fluid": fluid_constants},
        "grid_type": "simplex",
        "meshing_arguments": {"cell_size": (1 / 16)},
        "time_manager": time_manager,
        'units': units,
    }
    return FracPoroMechModel(params)


if __name__ == "__main__":
    model = make_model()
    model.prepare_simulation()

# %%
    # pp.run_time_dependent_model(
    #     model, {"prepare_simulation": False, "progressbars": True}
    # )

    pp.plot_grid(
        model.mdg,
        cell_value=model.pressure_variable,
        vector_value=model.displacement_variable,
        # plot_2d=False,
        # plot_2d=True,
        fracturewidth_1d=3,
        rgb=[0.5, 0.5, 1],
    )

    model.assemble_linear_system()
    diagnostics = model.run_diagnostics(
        grouping=None, additional_handlers={"shape": lambda x, a, b: x.shape[0]}
    )
    model.plot_diagnostics(diagnostics, key="max")
