# %%
import numpy as np
import porepy as pp
from porepy.applications.md_grids import fracture_sets
from porepy.examples.flow_benchmark_2d_case_3 import Permeability
from porepy.models.poromechanics import Poromechanics
from porepy.viz.diagnostics_mixin import DiagnosticsMixin

from mat_utils import *


class TimeStepping:
    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        self.time_manager.compute_time_step(iteration_counter, recompute_solution=False)

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.time_manager.compute_time_step(recompute_solution=True)


class PoroMech(
    MyAwesomeSolver,
    TimeStepping,
    Permeability,
    DiagnosticsMixin,
    Poromechanics,
):

    def set_domain(self) -> None:
        self._domain = pp.Domain({"xmax": 2.2, "ymax": 1})

    def set_fractures(self) -> None:
        # self._fractures = fracture_sets.benchmark_2d_case_1()
        # self._fractures = fracture_sets.benchmark_2d_case_3()
        self._fractures = fracture_sets.seven_fractures_one_L_intersection()

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryCondition:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.north, "dir")
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.north + bounds.south, "dir")
        return bc

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        values[bounds.north] = self.fluid.convert_units(4e5, "Pa")
        # values[bounds.south] = self.fluid.convert_units(0.01, "Pa")
        return values


def make_model():
    water = {
        "compressibility": 4.559 * 1e-10,  # [Pa^-1], isentropic compressibility
        "density": 998.2,  # [kg m^-3]
        "specific_heat_capacity": 4182.0,  # [J kg^-1 K^-1], isochoric specific heat
        "thermal_conductivity": 0.5975,  # [kg m^-3]
        "thermal_expansion": 2.068e-4,  # [K^-1]
        "viscosity": 1.002e-3,  # [Pa s], absolute viscosity
    }

    granite = {
        "biot_coefficient": 0.47,  # [-]
        "density": 2683.0,  # [kg * m^-3]
        "friction_coefficient": 0.6,  # [-]
        "lame_lambda": 7020826106,  # [Pa]
        "permeability": 5.0e-18,  # [m^2]
        "porosity": 1.3e-2,  # [-]
        "shear_modulus": 1.485472195e10,  # [Pa]
        "specific_heat_capacity": 720.7,  # [J * kg^-1 * K^-1]
        "specific_storage": 4.74e-10,  # [Pa^-1]
        "thermal_conductivity": 3.1,  # [W * m^-1 * K^-1]
        "thermal_expansion": 9.66e-6,  # [K^-1]
    }

    dt = 1e-2
    time_manager = pp.TimeManager(
        dt_init=dt, dt_min_max=(1e-10, 1e2), schedule=[0, 100 * dt], constant_dt=False
    )
    params = {
        "material_constants": {
            "solid": pp.SolidConstants(
                {"residual_aperture": 1e-4, "normal_permeability": 1e4} | granite
            ),
            "fluid": pp.FluidConstants(water),
        },
        "grid_type": "simplex",
        "time_manager": time_manager,
        "units": pp.Units(
            # m=1e-6,
            # m=1
            # m=1e6
            kg=1e6
        ),
        "meshing_arguments": {
            "cell_size": (1 / 20),
        },
        "iterative_solver": False,
    }
    return PoroMech(params)


# %%
if __name__ == "__main__":
    model = make_model()
    model.prepare_simulation()

    pp.plot_grid(
        model.mdg,
        plot_2d=True,
        fracturewidth_1d=3,
        rgb=[0.5, 0.5, 1],
    )
    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": True,
            # "max_iterations": 25,
        },
    )

    pp.plot_grid(
        model.mdg,
        cell_value=model.pressure_variable,
        plot_2d=True,
        # vector_value=model.displacement_variable,
        # alpha=0.5,
    )

    pp.plot_grid(
        model.mdg,
        cell_value=model.pressure_variable,
        vector_value=model.displacement_variable,
        alpha=0.5,
    )

    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
