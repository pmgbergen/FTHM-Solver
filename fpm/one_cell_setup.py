from typing import Literal
import numpy as np
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.examples.flow_benchmark_2d_case_3 import Permeability
from porepy.models.poromechanics import Poromechanics
from porepy.viz.diagnostics_mixin import DiagnosticsMixin

from pp_utils import (
    BCFlow,
    BCMechanicsOpen,
    BCMechanicsSliding,
    BCMechanicsSticking,
    MyPetscSolver,
    TimeStepping,
)


class PoroMech(
    MyPetscSolver,
    TimeStepping,
    # BCMechanicsOpen,
    # BCMechanicsSticking,
    BCMechanicsSliding,
    BCFlow,
    Permeability,
    DiagnosticsMixin,
    Poromechanics,
):

    def set_domain(self) -> None:
        m = self.solid.units.m
        self._domain = nd_cube_domain(2, 1 / m)

    def set_fractures(self) -> None:
        m = self.solid.units.m
        # self._fractures = [pp.LineFracture(np.array([[0.25, 0.5], [0.5, 0.5]]) / m)]
        self._fractures = [pp.LineFracture(np.array([[0.25, 0.35], [0.25, 0.35]]) / m)]

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return "simplex"
        # return 'cartesian'


def make_model() -> PoroMech:
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

    dt = 1e-3
    time_manager = pp.TimeManager(
        dt_init=dt, dt_min_max=(1e-10, 1e2), schedule=[0, 10 * dt], constant_dt=False
    )
    units = pp.Units(kg=1e9)
    m = units.m
    params = {
        "material_constants": {
            "solid": pp.SolidConstants(
                {"residual_aperture": 1e-4, "normal_permeability": 1e4} | granite
            ),
            "fluid": pp.FluidConstants(water),
        },
        "grid_type": "simplex",
        "time_manager": time_manager,
        "units": units,
        "meshing_arguments": {
            "cell_size": 0.5 / m,
        },
        "solver_type": "1",
        "simulation_name": "one_cell_setup",
        # "iterative_solver": False,
    }
    return PoroMech(params)


if __name__ == "__main__":
    model = make_model()
    model.prepare_simulation()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    sol = model.linear_system[1]
    model.after_nonlinear_iteration(sol)
    model.after_nonlinear_convergence(sol, [0], 1)
    model.after_simulation()
