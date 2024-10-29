from models import run_model, make_model
from plot_utils import write_dofs_info


def experiment_2():
    setups = []
    for grid_refinement in [
        1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 10,
        # 33,
    ]:
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
        run_model(setup)
        write_dofs_info(model)
        print(model.simulation_name())


if __name__ == "__main__":
    experiment_2()
