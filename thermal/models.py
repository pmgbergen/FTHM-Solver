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


class Physics(CubicLawPermeability, Thermoporomechanics):

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
            if initial_state != 'ignore':
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

    def compute_peclet(self):
        enthalpy = self.__enthalpy.value(self.equation_system)
        fourier = self.__fourier.value(self.equation_system)
        fourier_zero = abs(fourier) < 1e-10
        fourier[fourier_zero] = 1
        peclet = enthalpy / fourier
        peclet[fourier_zero] = 0
        peclet_max = abs(peclet).max()
        peclet_mean = abs(peclet).mean()
        return peclet_max, peclet_mean

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
        peclet_max, peclet_mean = self.compute_peclet()
        print(f"Peclet: {peclet_max:.1e}, CFL: {cfl:.1e}")
        self._linear_solve_stats.cfl = cfl
        self._linear_solve_stats.peclet_max = peclet_max
        self._linear_solve_stats.peclet_mean = peclet_mean
        super().before_nonlinear_iteration()
