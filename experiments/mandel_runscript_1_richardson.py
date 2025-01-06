from experiments.mandel_runscript_1 import run_model


if __name__ == "__main__":
    setups = []
    for barton_bandis in [0, 1, 2, 3, 4, 5]:
        for friction in [0, 1, 2]:
            for solver in [1]:
                setups.append(
                    {
                        "physics": 0,
                        "geometry": 0,
                        "barton_bandis_stiffness_type": barton_bandis,
                        "friction_type": friction,
                        "grid_refinement": 1,
                        "solver": solver,
                        "save_matrix": False,
                    }
                )
    for setup in setups:
        run_model(setup)
