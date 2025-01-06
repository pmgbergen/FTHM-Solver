import porepy as pp
from porepy.models.constitutive_laws import (
    ConstantPermeability,
    CubicLawPermeability,
    DimensionDependentPermeability,
    SpecificStorage,
)
from porepy.numerics.nonlinear import line_search
from porepy.models.poromechanics import Poromechanics


SETUP_REFERENCE = {
    "physics": 1,  # 0 - simplified; 1 - full
    "geometry": 1,  # 1 - 2D 1 fracture, 2 - 2D 7 fractures, 3 - 3D (TODO);
    "barton_bandis_stiffness_type": 1,  # 0 - off; 1 - small; 2 - medium, 3 - large
    "friction_type": 1,  # 0 - small, 1 - medium, 2 - large
    "grid_refinement": 1,  # 1 - coarsest level
    "solver": 2,  # 0 - Direct solver, 1 - Richardson + fixed stress splitting scheme, 2 - GMRES + scalable prec.
    "save_matrix": False,  # Save each matrix and state array. Might take space.
}


class Physics(DimensionDependentPermeability, SpecificStorage, Poromechanics):

    def simplified_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        matrix = [sd for sd in subdomains if sd.dim == self.nd]
        dp = self.perturbation_from_reference("pressure", matrix)
        phi_0 = self.reference_porosity(matrix)
        cf = self.fluid_compressibility(matrix)
        M_inv = self.specific_storage(matrix)  # 1 / M
        pressure_term = (M_inv + phi_0 * cf) * dp
        matrix_term = (
            pressure_term
            + self.porosity_change_from_displacement(matrix)
            + self._mpsa_consistency(matrix, self.darcy_keyword, self.pressure_variable)
        )
        # (1/M + phi_0 * cf) p + alpha ∇ * u

        fracture = [sd for sd in subdomains if sd.dim < self.nd]
        cfc = self.residual_aperture(fracture) * self.fluid_compressibility(fracture)
        dp = self.perturbation_from_reference("pressure", fracture)
        fracture_term = cfc * dp + self.aperture(fracture)
        # nu_0 * cf * p + nu

        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        result = (
            projection.cell_prolongation(matrix) @ matrix_term
            + projection.cell_prolongation(fracture) @ fracture_term
        )

        cell_volumes = self.wrap_grid_attribute(subdomains, "cell_volumes", dim=1)
        rho_ref = pp.ad.Scalar(self.fluid.reference_component.density, "reference_fluid_density")
        result *= cell_volumes * rho_ref

        result.set_name("simplified mass")
        return result

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        physics_type = self.params["setup"]["physics"]
        if physics_type == 1:
            return super().fluid_mass(subdomains)
        elif physics_type == 0:
            return self.simplified_mass(subdomains)
        raise ValueError(physics_type)

    def permeability(self, subdomains):
        physics_type = self.params["setup"]["physics"]
        permeability = self.params['setup'].get('permeability', 1)

        if permeability == 0:
            return ConstantPermeability.permeability(self, subdomains)
        elif permeability == 1:
            if physics_type == 1:
                return CubicLawPermeability.permeability(self, subdomains)
            elif physics_type == 0:
                const_perm = ConstantPermeability.permeability(self, subdomains)
                return const_perm


def get_barton_bandis_config(setup: dict):
    bb_type = setup["barton_bandis_stiffness_type"]
    if bb_type == 0:
        return {
            # Barton-Bandis elastic deformation. Defaults to 0.
            "fracture_gap": 1e-4,  # THIS SHOULD NOT CHANGE
            "maximum_elastic_fracture_opening": 0,
            # [Pa m^-1]  # increase this to make easier to converge (* 10) (1/10 of lame parameter)
            "fracture_normal_stiffness": 0,
        }
    elif bb_type == 1:
        return {
            "fracture_gap": 1e-4,
            "maximum_elastic_fracture_opening": 5e-5,
            "fracture_normal_stiffness": 1.2e5,
        }
    elif bb_type == 2:
        return {
            "fracture_gap": 1e-4,
            "maximum_elastic_fracture_opening": 5e-5,
            "fracture_normal_stiffness": 1.2e9,
        }
    elif bb_type == 3:
        return {
            "fracture_gap": 1e-4,
            "maximum_elastic_fracture_opening": 5e-5,
            "fracture_normal_stiffness": 1.2e13,
        }
    elif bb_type == 4:
        return {
            "fracture_gap": 1e-4,
            "maximum_elastic_fracture_opening": 5e-5,
            "fracture_normal_stiffness": 1.2e17,
        }
    elif bb_type == 5:
        return {
            "fracture_gap": 1e-4,
            "maximum_elastic_fracture_opening": 5e-5,
            "fracture_normal_stiffness": 1.2e19,
        }
    raise ValueError(bb_type)


def get_friction_coef_config(setup: dict):
    friction_type = setup["friction_type"]
    if friction_type == 0:
        return {
            "friction_coefficient": 0.1,  # [-]
        }
    elif friction_type == 1:
        return {
            "friction_coefficient": 0.577,  # [-]
        }
    elif friction_type == 2:
        return {"friction_coefficient": 0.8}
    raise ValueError(friction_type)


class ConstraintLineSearchNonlinearSolver(
    line_search.ConstraintLineSearch,  # The tailoring to contact constraints.
    line_search.SplineInterpolationLineSearch,  # Technical implementation of the actual search along given update direction
    line_search.LineSearchNewtonSolver,  # General line search.
):
    """Collect all the line search methods in one class."""
