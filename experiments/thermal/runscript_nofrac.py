import porepy as pp
import numpy as np
from experiments.models import Physics
from hm_solver import IterativeHMSolver as Solver
from experiments.thermal.thm_models import (
    ConstraintLineSearchNonlinearSolver,
    # Physics,
    get_barton_bandis_config,
    get_friction_coef_config,
)
from stats import StatisticsSavingMixin
# from experiments.thermal.thm_solver import ThermalSolver

XMAX = 1
YMAX = 1
ZMAX = 1


class Geometry2D0F(pp.SolutionStrategy):

    def set_domain(self) -> None:
        self._domain = pp.Domain({"xmin": 0, "xmax": XMAX, "ymin": 0, "ymax": YMAX})

    def _fluid_source(self, sd: pp.GridLike) -> np.ndarray:
        src = np.zeros(sd.num_cells)
        if sd.dim == self.nd:
            x, y, z = sd.cell_centers
            xmean = x.mean()
            ymean = y.mean()
            distance = np.sqrt((x - xmean) ** 2 + (y - ymean) ** 2 + (z - 0.1) ** 2)
            loc = np.where(distance == distance.min())[0][0]
            src[loc] = self.get_source_intensity(t=self.time_manager.time)
        return src


class Setup(Geometry2D0F, Solver, StatisticsSavingMixin, Physics):
    pass


def make_model(setup: dict):
    dt = 0.5

    cell_size_multiplier = setup["grid_refinement"]

    HOUR = 60 * 60

    params = {
        "setup": setup,
        "material_constants": {
            "solid": pp.SolidConstants(
                (
                    {
                        "shear_modulus": 1.2e10,  # [Pa]
                        "lame_lambda": 1.2e10,  # [Pa]
                        "dilation_angle": 5 * np.pi / 180,  # [rad]
                        "residual_aperture": 1e-4,  # [m]
                        "normal_permeability": 1e-4,
                        "permeability": 1e-14,  # [m^2]
                        # granite
                        "biot_coefficient": 0.47,  # [-]
                        "density": 2683.0,  # [kg * m^-3]
                        "porosity": 1.3e-2,  # [-]
                        "specific_storage": 4.74e-10,  # [Pa^-1]
                        # Thermal
                        "specific_heat_capacity": 720.7,
                        "thermal_conductivity": 0.1,  # Diffusion coefficient
                        "thermal_expansion": 9.66e-6,
                        "temperature": 350,
                    }
                    | get_barton_bandis_config(setup)
                    | get_friction_coef_config(setup)
                )
            ),
            "fluid": pp.FluidConstants(
                {
                    "pressure": 1e6,  # [Pa]
                    "compressibility": 4.559 * 1e-10,  # [Pa^-1], fluid compressibility
                    "density": 998.2,  # [kg m^-3]
                    "viscosity": 1.002e-3,  # [Pa s], absolute viscosity
                    # Thermal
                    "specific_heat_capacity": 4182.0,  # Вместимость
                    "thermal_conductivity": 0.5975,  # Diffusion coefficient
                    "temperature": 350,
                    "thermal_expansion": 2.068e-4,  # Density(T)
                }
            ),
        },
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=dt * HOUR,
            # dt_min_max=(0.01, 0.5),
            schedule=[0, 3 * HOUR, 6 * HOUR],
            iter_max=25,
            constant_dt=True,
        ),
        "units": pp.Units(kg=1e10),
        "meshing_arguments": {
            "cell_size": (0.1 * XMAX / cell_size_multiplier),
        },
        # experimental
        "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
    }
    return Setup(params)


def run_model(setup: dict):
    model = make_model(setup)
    model.prepare_simulation()
    print(model.simulation_name())

    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": False,
            "nl_convergence_tol": float("inf"),
            "nl_convergence_tol_res": 1e-7,
            "nl_divergence_tol": 1e8,
            "max_iterations": 25,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 1,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )

    print(model.simulation_name())


if __name__ == "__main__":
    run_model(
        {
            "physics": 1,
            "geometry": 0,
            "barton_bandis_stiffness_type": 2,
            "friction_type": 1,
            "grid_refinement": 1,
            "solver": 2,
            "save_matrix": True,
        }
    )
