import porepy as pp
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

    def before_nonlinear_iteration(self):
        t = self.temperature(self.mdg.subdomains()).value(self.equation_system)
        tmin, tmax = min(t), max(t)
        self._linear_solve_stats.temp_min = tmin
        self._linear_solve_stats.temp_max = tmax
        print(f"Temperature: {tmin:.2f}, {tmax:.2f}")
        super().before_nonlinear_iteration()

    pass
