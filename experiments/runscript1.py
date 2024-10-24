from models import run_model, make_model
from plot_utils import write_dofs_info


setup_reference = {
    "physics": 1,  # 0 - simplified; 1 - full
    "geometry": 1,  # 1 - 2D 1 fracture, 2 - 2D 7 fractures;
    "barton_bandis_stiffness_type": 1,  # 0 - off; 1 - small; 2 - medium, 3 - large
    "friction_type": 1,  # 0 - small, 1 - medium, 2 - large
    "grid_refinement": 1,  # 1 - coarsest level
    "solver": 2,  # 0 - Direct solver, 1 - Richardson + fixed stress splitting scheme, 2 - GMRES + scalable prec.
    "save_matrix": False,  # Save each matrix and state array. Might take space.
}


def experiment_1_barton_bandis_friction():
    setups = []
    for barton_bandis in [
        0,
        1,
        2,
        3,
        4,
        5,
    ]:
        for friction in [0, 1, 2]:
            for solver in [1, 2]:
                setups.append(
                    {
                        "physics": 0,
                        "geometry": 1,
                        "barton_bandis_stiffness_type": barton_bandis,
                        "friction_type": friction,
                        "grid_refinement": 1,
                        "solver": solver,
                        "save_matrix": True,
                    }
                )
    for setup in setups:
        model = make_model(setup)
        run_model(setup)
        write_dofs_info(model)
        print(model.simulation_name())


def experiment_1_grid_refinement():
    setups = []
    for grid_refinement in [1, 2, 3, 4]:
        for solver in [1, 11, 12]:
            setups.append(
                {
                    "physics": 1,
                    "geometry": 1,
                    "barton_bandis_stiffness_type": 2,
                    "friction_type": 1,
                    "grid_refinement": grid_refinement,
                    "solver": solver,
                    "save_matrix": True,
                }
            )
    for grid_refinement in [1, 2, 3, 4, 5, 6, 10, 33]:
        setups.append(
            {
                "physics": 1,
                "geometry": 1,
                "barton_bandis_stiffness_type": 2,
                "friction_type": 1,
                "grid_refinement": grid_refinement,
                "solver": 2,
                "save_matrix": False,
            }
        )
    for setup in setups:
        model = make_model(setup)
        run_model(setup)
        write_dofs_info(model)
        print(model.simulation_name())


if __name__ == "__main__":
    experiment_1_barton_bandis_friction()
    experiment_1_grid_refinement()
