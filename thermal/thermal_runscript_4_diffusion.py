import porepy as pp
import numpy as np
import time
from thermal.models import (
    Physics,
    ConstraintLineSearchNonlinearSolver,
    get_barton_bandis_config,
    get_friction_coef_config,
)
from thermal.thm_solver import THMSolver
from plot_utils import write_dofs_info
from stats import StatisticsSavingMixin
from porepy.applications.md_grids.fracture_sets import benchmark_2d_case_3

XMAX = 1000
YMAX = 1000


class Geometry(pp.SolutionStrategy):
    def initial_condition(self) -> None:
        super().initial_condition()
        # vals = np.load("stats_thermal_geo4x2_sol3_bb2_fr1_h.npy")
        vals = np.load("stats_thermal_geo4x2_sol3_bb2_fr1_p.npy")
        self.equation_system.set_variable_values(vals, time_step_index=0)
        self.equation_system.set_variable_values(vals, iterate_index=0)

        # num_cells = sum([sd.num_cells for sd in self.mdg.subdomains()])
        # val = self.reference_variable_values.pressure * np.ones(num_cells)
        # for time_step_index in self.time_step_indices:
        #     self.equation_system.set_variable_values(
        #         val,
        #         variables=[self.pressure_variable],
        #         time_step_index=time_step_index,
        #     )

        # for iterate_index in self.iterate_indices:
        #     self.equation_system.set_variable_values(
        #         val,
        #         variables=[self.pressure_variable],
        #         iterate_index=iterate_index,
        #     )

        # val = self.reference_variable_values.temperature * np.ones(num_cells)
        # for time_step_index in self.time_step_indices:
        #     self.equation_system.set_variable_values(
        #         val,
        #         variables=[self.temperature_variable],
        #         time_step_index=time_step_index,
        #     )

        # for iterate_index in self.iterate_indices:
        #     self.equation_system.set_variable_values(
        #         val,
        #         variables=[self.temperature_variable],
        #         iterate_index=iterate_index,
        #     )

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_values_pressure(self, boundary_grid):
        vals = super().bc_values_pressure(boundary_grid)
        sides = self.domain_boundary_sides(boundary_grid)
        mul = 1.1  # maybe too much
        vals[sides.east] *= mul
        # gradient
        vals[sides.north] += (
            (vals[sides.north] * mul - vals[sides.north])
            / (XMAX * 1.1 - XMAX * -0.1)
            * (boundary_grid.cell_centers[0, sides.north] - XMAX * -0.1)
        )
        vals[sides.south] += (
            (vals[sides.south] * mul - vals[sides.south])
            / (XMAX * 1.1 - XMAX * -0.1)
            * (boundary_grid.cell_centers[0, sides.south] - XMAX * -0.1)
        )
        return vals

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))
        # rho * g * h
        # 2683 * 10 * 3000
        val = self.units.convert_units(8e7, units="Pa")
        bc_values[1, sides.north] = -val * boundary_grid.cell_volumes[sides.north]
        #  make the gradient
        bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west] * 1.2
        bc_values[0, sides.east] = -val * boundary_grid.cell_volumes[sides.east] * 1.2

        return bc_values.ravel("F")

    def locate_source(self, subdomains):
        source_loc_x = XMAX * 0.9
        source_loc_y = YMAX * 0.5
        ambient = [sd for sd in subdomains if sd.dim == self.nd]
        fractures = [sd for sd in subdomains if sd.dim == self.nd - 1]
        lower = [sd for sd in subdomains if sd.dim <= self.nd - 2]

        x, y, z = np.concatenate([sd.cell_centers for sd in fractures], axis=1)
        source_loc = np.argmin((x - source_loc_x) ** 2 + (y - source_loc_y) ** 2)
        src_frac = np.zeros(x.size)
        src_frac[source_loc] = 1

        zeros_ambient = np.zeros(sum(sd.num_cells for sd in ambient))
        zeros_lower = np.zeros(sum(sd.num_cells for sd in lower))
        return np.concatenate([zeros_ambient, src_frac, zeros_lower])

    def fluid_source_mass_rate(self):
        return self.units.convert_units(3e-1, "kg * s^-1")  # very high
        # maybe inject and then stop injecting?

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        return super().fluid_source(subdomains) + pp.ad.DenseArray(src)

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        cv = self.fluid.components[0].specific_heat_capacity
        t_inj = (
            self.units.convert_units(273 + 40, "K")
            - self.reference_variable_values.temperature
        )
        src *= cv * t_inj
        # src *= self.units.convert_units(1e6, "J * s^-1")
        return super().energy_source(subdomains) + pp.ad.DenseArray(src)

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros(boundary_grid.num_cells)
        bc_values[:] = self.reference_variable_values.temperature
        # bc_values[sides.east] = self.units.convert_units(600, units="K")
        return bc_values

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": -0.1 * XMAX,
                "xmax": 1.1 * XMAX,
                "ymin": -0.1 * YMAX,
                "ymax": 1.1 * YMAX,
            }
        )

    def set_fractures(self) -> None:
        # self._fractures = []
        self._fractures = benchmark_2d_case_3(size=XMAX)

    def after_simulation(self):
        super().after_simulation()
        vals = self.equation_system.get_variable_values(time_step_index=0)
        # np.save(f'{self.simulation_name()}_equilibrium.npy', vals)
        name = f"{self.simulation_name()}_endstate_{int(time.time() * 1000)}.npy"
        print("Saving", name)
        np.save(name, vals)


class Setup(Geometry, THMSolver, StatisticsSavingMixin, Physics):
    pass


def make_model(setup: dict):

    cell_size_multiplier = setup["grid_refinement"]

    DAY = 24 * 60 * 60

    shear = 1.2e10
    lame = 1.2e10
    biot = 0.47
    # biot = 0
    porosity = 1.3e-2  # probably on the low side
    specific_storage = 1 / (lame + 2 / 3 * shear) * (biot - porosity) * (1 - biot)

    params = {
        "setup": setup,
        "material_constants": {
            "solid": pp.SolidConstants(
                # IMPORTANT
                permeability=1e-14,  # [m^2]
                residual_aperture=1e-4 * 1e1,  # [m]
                # LESS IMPORTANT
                shear_modulus=shear,  # [Pa]
                lame_lambda=lame,  # [Pa]
                dilation_angle=5 * np.pi / 180,  # [rad]
                normal_permeability=1e-4,
                # granite
                biot_coefficient=biot,  # [-]
                density=2683.0,  # [kg * m^-3]
                porosity=porosity,  # [-]
                specific_storage=specific_storage,  # [Pa^-1]
                # **get_barton_bandis_config(setup),
                **get_friction_coef_config(setup),
                # Thermal
                specific_heat_capacity=720.7,
                thermal_conductivity=0.1,  # Diffusion coefficient
                thermal_expansion=9.66e-6,
            ),
            "fluid": pp.FluidComponent(
                compressibility=4.559 * 1e-10,  # [Pa^-1], fluid compressibility
                density=998.2,  # [kg m^-3]
                viscosity=1.002e-3,  # [Pa s], absolute viscosity
                # Thermal
                specific_heat_capacity=4182.0,  # Вместимость
                thermal_conductivity=0.5975,  # Diffusion coefficient
                thermal_expansion=2.068e-4,  # Density(T)
            ),
            "numerical": pp.NumericalConstants(
                characteristic_displacement=1e-1,  # [m]
            ),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            pressure=3.5e7,  # [Pa]
            # temperature=350,  # [K]
            temperature=273 + 120,
        ),
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=1e1 * DAY,
            # schedule=[0, 50 * DAY],
            schedule=[0, 1e4 * DAY],
            iter_max=25,
            constant_dt=False,
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
            "max_iterations": 30,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 1,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )

    write_dofs_info(model)
    print(model.simulation_name())


if __name__ == "__main__":

    # for s in [
    #     1,
    #     # 1.1,
    #     # 1.2,
    # ]:
    for g in [2]:
        run_model(
            {
                "physics": 1,
                "geometry": 4,
                "barton_bandis_stiffness_type": 2,
                "friction_type": 1,
                "grid_refinement": g,
                "solver": 3,
                "save_matrix": False,
            }
        )
