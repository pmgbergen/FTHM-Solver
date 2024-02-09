# %%
from porepy.models.poromechanics import Poromechanics
from porepy.viz.diagnostics_mixin import DiagnosticsMixin
import porepy as pp
import numpy as np
from mat_utils import *
from porepy.applications.md_grids.domains import nd_cube_domain


class PoroMech(MyAwesomeSolver, DiagnosticsMixin, Poromechanics):

    def set_domain(self) -> None:
        self._domain = nd_cube_domain(2, 1.0)

    def set_fractures(self) -> None:
        self._fractures = [pp.LineFracture([[0.25, 2.0], [0.5, 0.5]])]

    def bc_type_mechanics(self, sd):
        boundary_faces = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, boundary_faces.south + boundary_faces.west + boundary_faces.east, "dir"
        )
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid):
        stress = np.zeros((self.nd, boundary_grid.num_cells))
        boundary_faces = self.domain_boundary_sides(boundary_grid)
        # stress[1, boundary_faces.north] = +self.solid.convert_units(1, 'kg * m^-1 * s^-2')  # open
        stress[0, boundary_faces.north] = -self.solid.convert_units(
            1, "kg * m^-1 * s^-2"
        )  # sticking
        # stress[0, boundary_faces.north] = +self.solid.convert_units(1, 'kg * m^-1 * s^-2')  # sliding
        return stress.ravel("F")

    def bc_type_fluid_flux(self, sd):
        boundary_faces = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, boundary_faces.north, "dir")

    def bc_type_darcy_flux(self, sd):
        boundary_faces = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, boundary_faces.north, "dir")

    def bc_values_pressure(self, boundary_grid):
        values = np.zeros(boundary_grid.num_cells)
        boundary_faces = self.domain_boundary_sides(boundary_grid)
        values[boundary_faces.north] = self.solid.convert_units(1, "Pa")
        return values


def make_model():
    granite = {}
    water = {}
    solid_constants = pp.SolidConstants(
        {"residual_aperture": 1e-1, "normal_permeability": 1e1} | granite
    )

    fluid_constants = pp.FluidConstants(water)

    dt = 1e-1

    time_manager = pp.TimeManager(dt_init=dt, dt_min_max=(1e-7, 1e2), schedule=[0, dt])
    # units = pp.Units(m=1e-6, kg=1e6)
    units = pp.Units()

    params = {
        "material_constants": {"solid": solid_constants, "fluid": fluid_constants},
        "grid_type": "simplex",
        "time_manager": time_manager,
        "units": units,
        "meshing_arguments": {
            # "cell_size": (1 / 32),
            "cell_size": (1 / 16),
            # "cell_size": (1 / 2),
        },
    }
    return PoroMech(params)


if __name__ == "__main__":
    model = make_model()
    model.prepare_simulation()

    pp.plot_grid(
        model.mdg,
        plot_2d=True,
        fracturewidth_1d=3,
        rgb=[0.5, 0.5, 1],
    )
    # pp.run_time_dependent_model(
    #     model, {"prepare_simulation": False, "progressbars": True}
    # )

    # pp.plot_grid(
    #     model.mdg,
    #     cell_value=model.pressure_variable,
    #     vector_value=model.displacement_variable,
    #     alpha=0.5,
    # )

    model.time_manager.increase_time()
    model.time_manager.increase_time_index()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()

    sd_1d = model.mdg.subdomains(dim=1)
    mass = model.mass_balance_equation(sd_1d)

    model.assemble_linear_system()
    mat, rhs = model.linear_system
    # spy(mat)
    # plt.show()
    # plot_mat(mat)
    # plt.show()

    block_matrix = make_block_mat(model, mat)
    A = block_matrix[0, 0]
    B = block_matrix[0, 3]
    C = block_matrix[3, 0]
    D = block_matrix[3, 3]
    # E = concatenate_blocks(block_matrix, [0], [1, 2, 4, 5])
    # F = concatenate_blocks(block_matrix, [3], [1, 2, 4, 5])
    # G = concatenate_blocks(block_matrix, [1, 2, 4, 5, 6], [0])
    # H = concatenate_blocks(block_matrix, [1, 2, 4, 5, 6], [3])
    # K = concatenate_blocks(block_matrix, [1, 2, 4, 5, 6], [1, 2, 4, 5])
    Omega = concatenate_blocks(block_matrix, [3, 0], [3, 0])
    # Phi = concatenate_blocks(block_matrix, [1, 2, 4, 5, 6], [3, 0])

    # plot_mat(Omega)
    # plot_mat(A)
    # spy(A)
    # plot_eigs(A, logx=True)
    # plt.show()
    assert (Omega - bmat([[D, C], [B, A]])).data.size == 0

    D_inv = inv(D)
    S_A = A - B @ D_inv @ C
    S_A_inv = inv(S_A)

    Omega_inv = inv(Omega)
    Omega_inv_op = OmegaInv(D_inv=D_inv, S_A_inv=S_A_inv, B=B, C=C)

    Omega_inv_fixed_strain = OmegaInv(D_inv=D_inv, S_A_inv=inv(A), B=B, C=C)

    fs = get_fixed_stress_stabilization(model)
    Omega_inv_fixed_stress = OmegaInv(D_inv=D_inv, S_A_inv=inv(A + fs), B=B, C=C)

    ones = np.ones(Omega.shape[0])
    expected = Omega_inv.dot(ones)
    result = Omega_inv_op.dot(ones)
    # print("error:", abs(expected - result).max())

    # print("cond:", cond(Omega))

    # cb_type = "pr_norm"
    # solve(Omega, label="noprec", callback_type=cb_type)
    # # solve(Omega, Omega_inv, label="exact inverse", callback_type=cb_type)
    # solve(Omega, Omega_inv_op, label="operator inverse", callback_type=cb_type)
    # solve(Omega, Omega_inv_fixed_strain, label="fixed strain", callback_type=cb_type)
    # solve(Omega, Omega_inv_fixed_stress, label="fixed stress", callback_type=cb_type)
    # plt.legend()
    # plt.show()
