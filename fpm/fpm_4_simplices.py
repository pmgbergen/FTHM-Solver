# %%
import numpy as np
import porepy as pp
from porepy.models.poromechanics import Poromechanics
from porepy.models.momentum_balance import MomentumBalance
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.constitutive_laws import CubicLawPermeability

from pp_utils import (
    CheckStickingSlidingOpen,
    MyPetscSolver,
    DymanicTimeStepping,
    NewtonBacktracking,
    NewtonBacktrackingSimple,
    StatisticsSavingMixin,
)

XMAX = 1.0
YMAX = 1.0
ZMAX = 1.0

fluid_material = {
    "compressibility": 4.559 * 1e-10,  # [Pa^-1], isentropic compressibility
    "density": 998.2,  # [kg m^-3]
    "viscosity": 1.002e-3,  # [Pa s], absolute viscosity
}
solid_material = {
    # match the paper
    "shear_modulus": 1.2e10,  # [Pa]
    "lame_lambda": 1.2e10,  # [Pa]
    "dilation_angle": 5 * np.pi / 180,  # [rad]
    "friction_coefficient": 0.577,  # [-]
    # try to match the paper but not exactly
    "residual_aperture": 1e-4,  # [m]
    "normal_permeability": 1e-4,
    "permeability": 1e-14,  # [m^2]
    # granite
    "biot_coefficient": 0.47,  # [-]
    "density": 2683.0,  # [kg * m^-3]
    "porosity": 1.3e-2,  # [-]
    "specific_storage": 4.74e-10,  # [Pa^-1]
    # other
    "maximum_fracture_closure": 0,  # Barton-Bandis elastic deformation. Defaults to 0.
}


class Fpm4(
    NewtonBacktracking,
    # NewtonBacktrackingSimple,
    MyPetscSolver,
    StatisticsSavingMixin,
    CheckStickingSlidingOpen,
    DymanicTimeStepping,
    CubicLawPermeability,
    Poromechanics,
    # MomentumBalance,
    # SinglePhaseFlow,
):

    def simulation_name(self):
        try:
            name = super().simulation_name()
        except Exception:
            name = "direct"
        cell_size = self.params["cell_size_multiplier"]
        return f"{name}_x{cell_size}"

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        st, sl, op, tr = self.sticking_sliding_open_transition()
        print()
        print("num sticking:", sum(st))
        print("num sliding:", sum(sl))
        print("num open:", sum(op))
        print("num trans:", sum(tr))

    # Geometry

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {"xmin": 0, "xmax": XMAX, "ymin": 0, "ymax": YMAX, "zmin": 0, "zmax": ZMAX}
        )

    def set_fractures(self) -> None:
        x = 0.3
        y = 0.3
        pts_list = [
            np.array(
                [
                    [x * XMAX, (1 - x) * XMAX, (1 - x) * XMAX, x * XMAX],  # x
                    [y * YMAX, (1 - y) * YMAX, y * YMAX, (1 - y) * YMAX],  # y
                    [z * ZMAX, z * ZMAX, z * ZMAX, z * ZMAX],  # z
                ]
            )
            for z in [
                # 0.2,
                # 0.4,
                0.5
                # 0.6,
                # 0.8,
            ]
        ]
        self._fractures = [pp.PlaneFracture(pts) for pts in pts_list]

    # Source

    fluid_source_key = "fluid_source"

    def locate_source(self, cell_centers: np.ndarray) -> np.ndarray:
        x = cell_centers[0]
        y = cell_centers[1]
        distance = np.sqrt((x - 0.499 * XMAX) ** 2 + (y - 0.499 * YMAX) ** 2)
        loc = distance == distance.min()
        assert loc.sum() == 1
        return loc

    def get_source_intensity(self, t):
        t_max = 6
        peak_intensity = self.fluid.convert_units(1e-3, "m^3 * s^-1")
        t_mid = t_max / 2
        if t <= t_mid:
            return t / t_mid * peak_intensity
        else:
            return (2 - t / t_mid) * peak_intensity * -1

    def _fluid_source(self, sd: pp.GridLike) -> np.ndarray:
        src = np.zeros(sd.num_cells)
        if sd.dim == (self.nd - 1):
            source_loc = self.locate_source(sd.cell_centers)
            src[source_loc] = self.get_source_intensity(t=self.time_manager.time)
        return src

    def update_time_dependent_ad_arrays(self) -> None:
        super().update_time_dependent_ad_arrays()
        for sd, data in self.mdg.subdomains(return_data=True):
            vals = self._fluid_source(sd)
            pp.set_solution_values(
                name=self.fluid_source_key, values=vals, data=data, iterate_index=0
            )

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        intf_source = super().fluid_source(subdomains)
        source = pp.ad.TimeDependentDenseArray(
            self.fluid_source_key, domains=subdomains
        )
        rho = self.fluid_density(subdomains)
        return intf_source + rho * source

    # Boundary conditions

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.bottom, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))
        # 10 Mpa
        val = self.solid.convert_units(1e7, units="Pa")
        x = 0.5  # 1
        bc_values[2, sides.top] = -val * boundary_grid.cell_volumes[sides.top]
        # bc_values[1, sides.top] = -val * boundary_grid.cell_volumes[sides.top] * 0.8

        bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west] * x

        # bc_values[2, sides.bottom] = val * boundary_grid.cell_volumes[sides.bottom]
        return bc_values.ravel("F")


def make_model(cell_size_multiplier=1, save_matrices=True):
    print(f"{cell_size_multiplier = }")
    dt = 0.5
    time_manager = pp.TimeManager(
        dt_init=dt,
        dt_min_max=(0.1, 0.5),
        schedule=[0, 3, 6],
        constant_dt=False,
        iter_max=25,
    )

    units = pp.Units(kg=1e10)
    params = {
        "material_constants": {
            "solid": pp.SolidConstants(solid_material),
            "fluid": pp.FluidConstants(fluid_material),
        },
        # "grid_type": "cartesian",
        "grid_type": "simplex",
        "time_manager": time_manager,
        "units": units,
        "cell_size_multiplier": cell_size_multiplier,
        "meshing_arguments": {
            "cell_size": (0.1 * XMAX / cell_size_multiplier),
        },
        # "iterative_solver": False,
        "solver_type": "2",
        "simulation_name": "fpm_4_simplices",
        "save_arrays": save_matrices,
    }
    return Fpm4(params)


def run(cell_size_multiplier: int, save_matrices: bool):

    model = make_model(
        cell_size_multiplier=cell_size_multiplier, save_matrices=save_matrices
    )
    model.prepare_simulation()
    print(model.simulation_name())

    # pp.plot_grid(
    #     model.mdg.subdomains(dim=2)[0],
    #     alpha=0.5,
    #     rgb=[0.5, 0.5, 1],
    # )
    # plt.show()

    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": True,
            "nl_convergence_tol": 1e-6,
            "nl_divergence_tol": 1e8,
            "max_iterations": 10,
        },
    )

    # pp.plot_grid(
    #     model.mdg,
    #     cell_value=model.pressure_variable,
    #     vector_value=model.displacement_variable,
    #     alpha=0.5,
    # )

    print(model.simulation_name())


# %%
if __name__ == "__main__":
    # for i in [0.5, 1, 2]:
    #     run(cell_size_multiplier=i, save_matrices=True)
    run(cell_size_multiplier=3, save_matrices=False)
