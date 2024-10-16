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


def experiment_1():
    setups = []
    for physics in [1]:
        for barton_bandis in [2]:
            for friction_type in [1]:
                for solver in [2]:
                    # for grid_refinement in [1, 2, 3, 4, 5, 6]:
                    for grid_refinement in [10, 33, 100]:
                        setups.append(
                            {
                                "physics": physics,
                                "geometry": 1,  # 1 - 2D 1 fracture;
                                "barton_bandis_stiffness_type": barton_bandis,
                                "friction_type": friction_type,
                                "grid_refinement": grid_refinement,
                                "solver": solver,
                                "save_matrix": False,
                            }
                        )

    for setup in setups:
        model = make_model(setup)
        run_model(setup)
        write_dofs_info(model)
        print(model.simulation_name())


def experiment_2():
    setups = []
    for grid_refinement in [1, 2, 3, 4, 5, 6, 10, 33, 100]:
        setups.append(
            {
                "physics": 1,
                "geometry": 2,
                "barton_bandis_stiffness_type": 2,
                "friction_type": 1,
                "grid_refinement": grid_refinement,
                "solver": 2,
                "save_matrix": False,
            }
        )

    for setup in setups:
        model = make_model(setup)
        # run_model(setup)
        write_dofs_info(model)
        print(model.simulation_name())


if __name__ == "__main__":
    # experiment_1()
    experiment_2()
