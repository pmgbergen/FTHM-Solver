# %%
# 2D model, 8 fractures, 1 intersection.
# Flow inlet boundary condition
# Water and granite


from typing import Literal
import numpy as np
import porepy as pp
from porepy.applications.md_grids import fracture_sets
from porepy.examples.flow_benchmark_2d_case_3 import Permeability
from porepy.models.poromechanics import Poromechanics
from porepy.viz.diagnostics_mixin import DiagnosticsMixin

from iterative_solver import (
    BCFlow,
    BCMechanicsOpen,
    BCMechanicsSliding,
    BCMechanicsSticking,
    IterativeHMSolver,
    DymanicTimeStepping,
    StatisticsSavingMixin,
)


# MEGA = 1e9
MEGA = 1
# MEGA = 1e5


class PoroMech(
    IterativeHMSolver,
    DymanicTimeStepping,
    StatisticsSavingMixin,
    # BCMechanicsOpen,
    # BCMechanicsSticking,
    BCMechanicsSliding,
    BCFlow,
    Permeability,
    DiagnosticsMixin,
    Poromechanics,
):

    @property
    def fracture_permeabilities(self) -> np.ndarray:
        """Permeability of the fractures.

        Ordering corresponds to definition of fractures in the geometry.

        """
        return np.array([1, 1, 1, 1e-8, 1e-8, 1, 1, 1, 1, 1]) * MEGA

    def set_domain(self) -> None:
        self._domain = pp.Domain({"xmax": 2.2, "ymax": 1})

    def set_fractures(self) -> None:
        self._fractures = fracture_sets.seven_fractures_one_L_intersection()
        # self._fractures = []

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return "simplex"


def make_model(cell_size=(1 / 20)):
    water = {
        "compressibility": 4.559 * 1e-10 * MEGA,  # [Pa^-1], isentropic compressibility
        "density": 998.2,  # [kg m^-3]
        "specific_heat_capacity": 4182.0,  # [J kg^-1 K^-1], isochoric specific heat
        "thermal_conductivity": 0.5975,  # [kg m^-3]
        "thermal_expansion": 2.068e-4,  # [K^-1]
        "viscosity": 1.002e-3,  # [Pa s], absolute viscosity
    }

    granite = {
        "biot_coefficient": 0.47 * MEGA,  # [-]
        "density": 2683.0,  # [kg * m^-3]
        "friction_coefficient": 0.6,  # [-]
        "lame_lambda": 7020826106 / MEGA,  # [Pa]
        "permeability": 5.0e-18,  # [m^2]
        "porosity": 1.3e-2,  # [-]
        "shear_modulus": 1.485472195e10,  # [Pa]
        "specific_heat_capacity": 720.7,  # [J * kg^-1 * K^-1]
        "specific_storage": 4.74e-10 * MEGA,  # [Pa^-1]
        "thermal_conductivity": 3.1,  # [W * m^-1 * K^-1]
        "thermal_expansion": 9.66e-6,  # [K^-1]
    }

    dt = 1e-3
    time_manager = pp.TimeManager(
        dt_init=dt, dt_min_max=(1e-10, 1e2), schedule=[0, 10 * dt], constant_dt=False
    )

    # cell_size = 1 / 40

    units = pp.Units()
    # units = pp.Units(kg=1e6)
    units = pp.Units(kg=1e9)
    # units = pp.Units(m=1e6, kg=1e9)
    m = units.m
    # m = 1
    params = {
        "material_constants": {
            "solid": pp.SolidConstants(
                {"residual_aperture": 1e-4, "normal_permeability": 1e4 * MEGA} | granite
            ),
            "fluid": pp.FluidConstants(water),
        },
        "grid_type": "simplex",
        "time_manager": time_manager,
        "units": units,
        "meshing_arguments": {
            "cell_size": cell_size / m,
        },
        'solver_type': '1',
        "simulation_name": "fpm_1",
        # "iterative_solver": False,
    }
    return PoroMech(params)


# %%
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    model = make_model()
    print(model.simulation_name)

    model.prepare_simulation()

    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    # model.plot_diagnostics(model.run_diagnostics(), "max")
    # plt.show()

    # pp.plot_grid(
    #     model.mdg,
    #     plot_2d=True,
    #     fracturewidth_1d=3,
    #     rgb=[0.5, 0.5, 1],
    # )

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

    # pp.plot_grid(
    #     model.mdg,
    #     cell_value=model.pressure_variable,
    #     vector_value=model.displacement_variable,
    #     alpha=0.5,
    # )

    print(model.simulation_name)

# %%
