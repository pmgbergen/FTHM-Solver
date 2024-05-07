from porepy.models.poromechanics import Poromechanics
import porepy as pp
import numpy as np


class MyModel(Poromechanics):

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        self.time_manager.compute_time_step(iteration_counter, recompute_solution=False)

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.time_manager.compute_time_step(recompute_solution=True)


time_manager = pp.TimeManager(
    dt_init=1, dt_min_max=(0.1, 1), schedule=[0, 5, 10], constant_dt=False
)
model = MyModel(
    {
        "time_manager": time_manager,
    }
)

pp.run_time_dependent_model(
    model,
    {
        "progressbars": True,
    },
)
