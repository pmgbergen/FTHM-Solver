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

from porepy.examples.mandel_biot import MandelSetup
from porepy.viz.data_saving_model_mixin import DataSavingMixin

XMAX = 1000
YMAX = 1000
ZMAX = 1000


class Geometry2D0F(pp.SolutionStrategy):

    def save_data_time_step(self) -> None:
        DataSavingMixin.save_data_time_step(self)

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


class Setup(Geometry2D0F, Solver, StatisticsSavingMixin, MandelSetup):
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
                        "lame_lambda": 1.65e9,  # [Pa]
                        "shear_modulus": 2.475e9,  # [Pa]
                        "specific_storage": 6.0606e-11,  # [Pa^-1]
                        "permeability": 9.869e-14,  # [m^2]
                        "biot_coefficient": 1.0,  # [-]
                    }
                    # | get_barton_bandis_config(setup)
                    # | get_friction_coef_config(setup)
                )
            ),
            "fluid": pp.FluidConstants(
                {
                    "density": 1e3,  # [kg * m^-3]
                    "viscosity": 1e-3,  # [Pa * s]
                }
            ),
        },
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=1e5,
            # dt_min_max=(0.01, 0.5),
            schedule=[0, 1e6],
            iter_max=25,
            constant_dt=True,
        ),
        "units": pp.Units(kg=1e9),
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
            "grid_refinement": 3,
            "solver": 2,
            "save_matrix": True,
        }
    )
