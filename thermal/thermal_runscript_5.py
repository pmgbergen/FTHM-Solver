import porepy as pp
import numpy as np
import time
from thermal.models import Physics, ConstraintLineSearchNonlinearSolver
from thermal.thm_solver import THMSolver
from plot_utils import write_dofs_info
from stats import StatisticsSavingMixin

XMAX = 1000
YMAX = 2000
ZMAX = 1000

class Geometry(pp.PorePyModel):
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.north, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))
        # rho * g * h
        # 2683 * 10 * 3000
        val = self.units.convert_units(8e7, units="Pa")
        bc_values[1, sides.south] = +val * boundary_grid.cell_volumes[sides.south]
        bc_values[2, sides.top] = -val * boundary_grid.cell_volumes[sides.top] * 0.9
        bc_values[2, sides.bottom] = (
            val * boundary_grid.cell_volumes[sides.bottom] * 0.9
        )
        bc_values[0, sides.west] = +val * boundary_grid.cell_volumes[sides.west] * 1.2
        bc_values[0, sides.east] = (-val * boundary_grid.cell_volumes[sides.east]) * 1.2
        return bc_values.ravel("F")

    def locate_source(self, subdomains):
        xmax = self._domain.bounding_box["xmax"]
        ymax = self._domain.bounding_box["ymax"]
        zmax = self._domain.bounding_box["zmax"]
        source_loc_x = xmax * 0.5
        source_loc_y = ymax * 0.1
        source_loc_z = zmax * 0.5
        ambient = [sd for sd in subdomains if sd.dim == self.nd]
        fractures = [sd for sd in subdomains if sd.dim == self.nd - 1]
        lower = [sd for sd in subdomains if sd.dim <= self.nd - 2]

        x, y, z = np.concatenate([sd.cell_centers for sd in fractures], axis=1)
        source_loc = np.argmin(
            (x - source_loc_x) ** 2 + (y - source_loc_y) ** 2 + (z - source_loc_z) ** 2
        )
        src_frac = np.zeros(x.size)
        src_frac[source_loc] = 1

        zeros_ambient = np.zeros(sum(sd.num_cells for sd in ambient))
        zeros_lower = np.zeros(sum(sd.num_cells for sd in lower))
        return np.concatenate([zeros_ambient, src_frac, zeros_lower])

    def fluid_source_mass_rate(self):
        if self.params["linear_solver_config"]["steady_state"]:
            return 0
        else:
            return self.units.convert_units(1e3, "kg * s^-1")
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
        return super().energy_source(subdomains) + pp.ad.DenseArray(src)

    def set_domain(self):
        #     self._domain = pp.Domain(
        #         {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000, "zmin": 0, "zmax": 1000}
        #     )
        # self._domain = pp.Domain(
        #     {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 2250, "zmin": 0, "zmax": 1000}
        # )
        self._domain = pp.Domain(
            {
                "xmin": 0,
                "ymin": 0,
                "zmin": 0,
                "xmax": XMAX,
                "ymax": YMAX,
                "zmax": ZMAX,
            }
        )

    def set_fractures(self) -> None:
        #     coords_a = [0.5, 0.5, 0.5, 0.5]
        #     coords_b = [0.2, 0.2, 0.8, 0.8]
        #     coords_c = [0.2, 0.8, 0.8, 0.2]
        #     pts = []
        #     pts.append(np.array([coords_a, coords_b, coords_c]) * 1000)
        #     pts.append(np.array([coords_b, coords_a, coords_c]) * 1000)
        #     pts.append(np.array([coords_b, coords_c, coords_a]) * 1000)
        #     self._fractures = [pp.PlaneFracture(pts[i]) for i in range(3)]

        fracs = np.array(
            [
                [
                    [0.05, 0.95, 0.95, 0.05],
                    [0.25, 0.25, 2.0, 2.0],
                    [0.5, 0.5, 0.5, 0.5],
                ],
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.05, 0.05, 0.3, 0.3],
                    [0.95, 0.05, 0.05, 0.95],
                ],
                [
                    [0.05, 0.95, 0.95, 0.05],
                    [1.0, 1.0, 2.2, 2.2],
                    [0.5, 0.5, 0.85, 0.85],
                ],
                [
                    [0.05, 0.95, 0.95, 0.05],
                    [1.0, 1.0, 2.2, 2.2],
                    [0.48, 0.48, 0.14, 0.14],
                ],
                [
                    [0.23, 0.23, 0.17, 0.17],
                    [1.9, 1.9, 2.2, 2.2],
                    [0.3, 0.7, 0.7, 0.3],
                ],
                [
                    [0.17, 0.17, 0.23, 0.23],
                    [1.9, 1.9, 2.2, 2.2],
                    [0.3, 0.7, 0.7, 0.3],
                ],
                [
                    [0.77, 0.77, 0.77, 0.77],
                    [1.9, 1.9, 2.2, 2.2],
                    [0.3, 0.7, 0.7, 0.3],
                ],
                [
                    [0.83, 0.83, 0.83, 0.83],
                    [1.9, 1.9, 2.2, 2.2],
                    [0.3, 0.7, 0.7, 0.3],
                ],
            ]
        )
        xscale = XMAX / 2
        yscale = YMAX / 2
        zscale = ZMAX / 2
        fracs[:, 0] /= fracs[:, 0].max()
        fracs[:, 1] /= fracs[:, 1].max()
        fracs[:, 2] /= fracs[:, 2].max()
        fracs[:, 0] = xscale / 2 + fracs[:, 0] * xscale
        fracs[:, 1] = yscale / 2 + fracs[:, 1] * yscale
        fracs[:, 2] = zscale / 2 + fracs[:, 2] * zscale
        # fracs *= 1000
        self._fractures = [
            pp.PlaneFracture(frac, check_convexity=True) for frac in fracs
        ]

    # def set_geometry(self) -> None:
    #     """Create mixed-dimensional grid and fracture network."""

    #     # Create mixed-dimensional grid and fracture network.
    #     self.mdg, self.fracture_network = benchmark_3d_case_3(
    #         refinement_level=self.params["refinement_level"]
    #     )
    #     self.nd: int = self.mdg.dim_max()

    #     # Obtain domain and fracture list directly from the fracture network.
    #     self._domain = self.fracture_network.domain
    #     self._domain.bounding_box["xmin"] = -0.2 * self._domain.bounding_box["xmax"]
    #     self._domain.bounding_box["ymin"] = -0.2 * self._domain.bounding_box["ymax"]
    #     self._domain.bounding_box["zmin"] = -0.2 * self._domain.bounding_box["zmax"]
    #     self._domain.bounding_box["xmax"] *= 1.2
    #     self._domain.bounding_box["ymax"] *= 1.2
    #     self._domain.bounding_box["zmax"] *= 1.2
    #     self._fractures = self.fracture_network.fractures

    #     # Create projections between local and global coordinates for fracture grids.
    #     pp.set_local_coordinate_projections(self.mdg)

    def after_simulation(self):
        super().after_simulation()
        vals = self.equation_system.get_variable_values(time_step_index=0)
        name = f"{self.simulation_name()}_endstate_{int(time.time() * 1000)}.npy"
        print("Saving", name)
        self.params["linear_solver_config"]["end_state_filename"] = name
        np.save(name, vals)


class Setup(Geometry, THMSolver, StatisticsSavingMixin, Physics):
    pass


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
        end_time = 2e3
    porosity = 1.3e-2  # probably on the low side

    params = {
        "linear_solver_config": setup,
        "folder_name": "visualization_3d",
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
                friction_coefficient=0.577,  # [-]
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
            iter_max=10,
            constant_dt=False,
            recomp_max=1,
        ),
        "units": pp.Units(kg=1e10),
        "meshing_arguments": {
            "cell_size": (0.33 * 2250 / cell_size_multiplier),
        },
        # "refinement_level": cell_size_multiplier,
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
            "Global_line_search": 0,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )

    write_dofs_info(model)
    print(model.simulation_name())


if __name__ == "__main__":

    common_params = {
        "geometry": "5",
        "save_matrix": False,
    }
    for g in [
        # 1,
        # 2,
        # 5,
        # 10,
        15,
        20
    ]:
        for s in [
            'FGMRES',
            # "CPR",
            # "SAMG",
            # "S4_diag",
            # "SAMG+ILU",
            # "S4_diag+ILU",
            # "AAMG+ILU",
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
