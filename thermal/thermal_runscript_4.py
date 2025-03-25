import porepy as pp
import numpy as np
import time
from thermal.models import Physics, ConstraintLineSearchNonlinearSolver
from thermal.thm_solver import THMSolver
from plot_utils import write_dofs_info
from stats import StatisticsSavingMixin
from porepy.applications.md_grids.fracture_sets import benchmark_2d_case_3

XMAX = 2000
YMAX = 2000


class Geometry(pp.PorePyModel):

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

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
        source_loc_x = XMAX * 0.5
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
        if self.params["linear_solver_config"]["steady_state"]:
            return 0
        else:
            return self.units.convert_units(1e1, "kg * s^-1")
            # maybe inject and then stop injecting?

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        return super().fluid_source(subdomains) + pp.ad.DenseArray(src)

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        cv = self.fluid.components[0].specific_heat_capacity
        t_inj = 40
        if self.params["linear_solver_config"].get("isothermal", False):
            t_inj = self.reference_variable_values.temperature - 273

        t_inj = (
            self.units.convert_units(273 + t_inj, "K")
            - self.reference_variable_values.temperature
        )
        src *= cv * t_inj
        return super().energy_source(subdomains) + pp.ad.DenseArray(src)

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": 0,
                "xmax": XMAX,
                "ymin": 0,
                "ymax": YMAX,
            }
        )

    def set_fractures(self) -> None:
        # self._fractures = benchmark_2d_case_3(size=XMAX)
        points = np.array(
            [
                [[0.0500, 0.2200], [0.4160, 0.0624]],
                [[0.0500, 0.2500], [0.2750, 0.1350]],
                [[0.1500, 0.4500], [0.6300, 0.0900]],
                [[0.1500, 0.4000], [0.9167, 0.5000]],
                [[0.6500, 0.849723], [0.8333, 0.167625]],
                [[0.7000, 0.849723], [0.2350, 0.167625]],
                [[0.6000, 0.8500], [0.3800, 0.2675]],
                [[0.3500, 0.8000], [0.9714, 0.7143]],
                [[0.7500, 0.9500], [0.9574, 0.8155]],
                [[0.1500, 0.4000], [0.8363, 0.9727]],
            ]
        )
        xscale = XMAX / 2
        yscale = YMAX / 2
        points[:, 0] = xscale / 2 + points[:, 0] * xscale
        points[:, 1] = yscale / 2 + points[:, 1] * yscale
        self._fractures = [pp.LineFracture(pts) for pts in points]

    def after_simulation(self):
        super().after_simulation()
        vals = self.equation_system.get_variable_values(time_step_index=0)
        name = f"{self.simulation_name()}_endstate_{int(time.time() * 1000)}.npy"
        print("Saving", name)
        self.params["linear_solver_config"]["end_state_filename"] = name
        np.save(name, vals)


class Setup(Geometry, THMSolver, StatisticsSavingMixin, Physics):

    def simulation_name(self):
        name = super().simulation_name()
        if self.params["linear_solver_config"].get("isothermal", False):
            name = f"{name}_isothermal"
        if (x := self.params["linear_solver_config"].get("thermal_conductivity_multiplier", 1)) != 1:
            name = f"{name}_diffusion={x}"
        if (x := self.params['setup'].get('friction_coef', None)):
            name = f"{name}_friction={x}"
        return name


def make_model(setup: dict):

    cell_size_multiplier = setup["grid_refinement"]

    DAY = 24 * 60 * 60

    shear = 1.2e10
    lame = 1.2e10
    if setup["steady_state"]:
        biot = 0
        dt_init = 1e0
        end_time = 1e1
    else:
        biot = 0.47
        dt_init = 1e-3
        if setup["grid_refinement"] >= 33:
            dt_init = 1e-4  # Is this necessary?
        end_time = 5e2
    porosity = 1.3e-2  # probably on the low side

    thermal_conductivity_multiplier = setup.get("thermal_conductivity_multiplier", 1)
    friction_coef = setup.get('friction_coef', 0.577)

    params = {
        "linear_solver_config": setup,
        "folder_name": "visualization_2d",
        "material_constants": {
            "solid": pp.SolidConstants(
                # IMPORTANT
                permeability=1e-13,  # [m^2]
                residual_aperture=1e-3,  # [m]
                # LESS IMPORTANT
                shear_modulus=shear,  # [Pa]
                lame_lambda=lame,  # [Pa]
                dilation_angle=5 * np.pi / 180,  # [rad]
                normal_permeability=1e-4,
                # granite
                biot_coefficient=biot,  # [-]
                density=2683.0,  # [kg * m^-3]
                porosity=porosity,  # [-]
                friction_coefficient=friction_coef,  # [-]
                # Thermal
                specific_heat_capacity=720.7,
                thermal_conductivity=0.1
                * thermal_conductivity_multiplier,  # Diffusion coefficient
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
                characteristic_displacement=2e0,  # [m]
            ),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            pressure=3.5e7,  # [Pa]
            temperature=273 + 120,
        ),
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=dt_init * DAY,
            schedule=[0, end_time * DAY],
            # iter_max=6,
            # iter_optimal_range=(4, 5),
            constant_dt=False,
            # recomp_factor=0.1,
            # iter_relax_factors=(0.1, 1.3),
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
            "max_iterations": 10,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 1,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 0,  # Set to 0 to use turn off the tailored line search
        },
    )

    write_dofs_info(model)
    print(model.simulation_name())


if __name__ == "__main__":

    # for friction_coef in [0.1, 0.9]:
        # for thermal_conductivity_multiplier in [0.01, 100]:

    common_params = {
        "geometry": "4h_steady",
        "save_matrix": False,
    # "isothermal": False,
        # "friction_coef": friction_coef,
        # 'thermal_conductivity_multiplier': thermal_conductivity_multiplier,
    }

    for g in [
        1,
        2,
        5,
        # 25,
        # 33,
        # 40,
    ]:
        for s in [
            'FGMRES',
            # "SAMG",
            # "CPR",
            # "SAMG+ILU",
            # "S4_diag+ILU",
            # "AAMG+ILU",
            # "S4_diag",
        ]:
            print("Running steady state")
            params = {
                "grid_refinement": g,
                "steady_state": True,
                "solver": s,
            } | common_params
            run_model(params)
            end_state_filename = params["end_state_filename"]

            print("Running injection")
            params = {
                "grid_refinement": g,
                "steady_state": False,
                "initial_state": end_state_filename,
                "solver": s,
            } | common_params
            run_model(params)
