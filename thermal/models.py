from functools import cached_property
import porepy as pp
import numpy as np
from porepy.models.constitutive_laws import (
    ConstantPermeability,
    CubicLawPermeability,
    DimensionDependentPermeability,
    SpecificStorage,
)
from experiments.models import (
    ConstraintLineSearchNonlinearSolver,
    get_friction_coef_config,
    get_barton_bandis_config,
)
from porepy.models.thermoporomechanics import Thermoporomechanics


SETUP_REFERENCE = {
    "physics": 1,  # 0 - simplified; 1 - full
    "geometry": 1,  # 1 - 2D 1 fracture, 2 - 2D 7 fractures, 3 - 3D (TODO);
    "barton_bandis_stiffness_type": 1,  # 0 - off; 1 - small; 2 - medium, 3 - large
    "friction_type": 1,  # 0 - small, 1 - medium, 2 - large
    "grid_refinement": 1,  # 1 - coarsest level
    "solver": 2,  # 0 - Direct solver, 1 - Richardson + fixed stress splitting scheme, 2 - GMRES + scalable prec.
    "save_matrix": False,  # Save each matrix and state array. Might take space.
}


from functools import cache


def cache_ad_tree(func):
    func = cache(func)

    def inner(subdomains):
        return func(tuple(subdomains))

    return inner


class Physics(CubicLawPermeability, Thermoporomechanics):

    def prepare_simulation(self):
        # This speeds up Porepy by caching the once-built AD trees.
        self.aperture = cache_ad_tree(self.aperture)
        self.specific_volume = cache_ad_tree(self.specific_volume)
        self.cubic_law_permeability = cache_ad_tree(self.cubic_law_permeability)
        self.combine_boundary_operators_darcy_flux = cache_ad_tree(
            self.combine_boundary_operators_darcy_flux
        )
        self.porosity = cache_ad_tree(self.porosity)
        self.opening_indicator = cache_ad_tree(self.opening_indicator)
        self.sliding_indicator = cache_ad_tree(self.sliding_indicator)
        super().prepare_simulation()

    def initial_condition(self) -> None:
        super().initial_condition()
        if self.params["setup"]["steady_state"]:
            num_cells = sum([sd.num_cells for sd in self.mdg.subdomains()])
            val = self.reference_variable_values.pressure * np.ones(num_cells)
            for time_step_index in self.time_step_indices:
                self.equation_system.set_variable_values(
                    val,
                    variables=[self.pressure_variable],
                    time_step_index=time_step_index,
                )

            for iterate_index in self.iterate_indices:
                self.equation_system.set_variable_values(
                    val,
                    variables=[self.pressure_variable],
                    iterate_index=iterate_index,
                )

            val = self.reference_variable_values.temperature * np.ones(num_cells)
            for time_step_index in self.time_step_indices:
                self.equation_system.set_variable_values(
                    val,
                    variables=[self.temperature_variable],
                    time_step_index=time_step_index,
                )

            for iterate_index in self.iterate_indices:
                self.equation_system.set_variable_values(
                    val,
                    variables=[self.temperature_variable],
                    iterate_index=iterate_index,
                )
        else:
            initial_state = self.params["setup"]["initial_state"]
            if initial_state != "ignore":
                vals = np.load(initial_state)
                self.equation_system.set_variable_values(vals, time_step_index=0)
                self.equation_system.set_variable_values(vals, iterate_index=0)

    @cached_property
    def __enthalpy(self):
        return self.enthalpy_flux(self.mdg.subdomains())

    @cached_property
    def __fourier(self):
        return self.fourier_flux(self.mdg.subdomains())

    @cached_property
    def cfl_flux(self):
        return (
            self.darcy_flux(self.mdg.subdomains())
            / self.fluid.reference_component.viscosity
        )

    def compute_convection_diffusion_transport(self):
        enthalpy = abs(self.__enthalpy.value(self.equation_system))
        fourier = abs(self.__fourier.value(self.equation_system))
        return enthalpy.max(), enthalpy.mean(), fourier.max(), fourier.mean()

    def compute_cfl(self):
        flux = abs(self.cfl_flux.value(self.equation_system)).max()
        length = np.concatenate(
            [sd.cell_volumes for sd in self.mdg.subdomains()]
        ).mean()
        time_step = self.time_manager.dt
        CFL = flux * time_step / length
        return CFL

    def before_nonlinear_iteration(self):
        t = self.temperature(self.mdg.subdomains()).value(self.equation_system)
        tmin, tmax = min(t), max(t)
        self._linear_solve_stats.temp_min = tmin
        self._linear_solve_stats.temp_max = tmax
        print(f"Temperature: {tmin:.2f}, {tmax:.2f}")

        cfl = self.compute_cfl()
        enthalpy_max, enthalpy_mean, fourier_max, fourier_mean = (
            self.compute_convection_diffusion_transport()
        )
        print(f"Peclet: {enthalpy_max/fourier_max:.1e}, CFL: {cfl:.1e}")
        self._linear_solve_stats.cfl = cfl
        self._linear_solve_stats.enthalpy_max = enthalpy_max
        self._linear_solve_stats.enthalpy_mean = enthalpy_mean
        self._linear_solve_stats.fourier_max = fourier_max
        self._linear_solve_stats.fourier_mean = fourier_mean
        if len(self.nonlinear_solver_statistics.nonlinear_increment_norms) != 0:
            incr_norm = self.nonlinear_solver_statistics.nonlinear_increment_norms[-1]
            res_norm = self.nonlinear_solver_statistics.residual_norms[-1]
            print(f"Increment: {incr_norm:.1e}, residual: {res_norm:.1e}")
        super().before_nonlinear_iteration()
