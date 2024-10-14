from dataclasses import dataclass, field


# @dataclass
# class LinearSolveStats:
#     time_invert_F: float = -1
#     time_prepare_mass: float = -1
#     time_prepare_momentum: float = -1
#     time_prepare_solver: float = -1
#     time_gmres: float = -1
#     gmres_iters: int = -1
#     time_solve_linear_system: float = -1

#     simulation_dt: float = -1
#     petsc_converged_reason: int = -100
#     sticking: list[int] = field(default_factory=list)
#     sliding: list[int] = field(default_factory=list)
#     open_: list[int] = field(default_factory=list)
#     transition: list[int] = field(default_factory=list)


@dataclass
class LinearSolveStats:
    simulation_dt: float = -1
    krylov_iters: int = -1
    petsc_converged_reason: int = -100
    error_matrix_contribution: float = -1
    num_sticking: int = -1
    num_sliding: int = -1
    num_open: int = -1
    # Assumptions
    coulomb_mismatch: float = -1
    sticking_u_mismatch: float = -1
    lambdan_max: float = -1
    lambdat_max: float = -1
    un_max: float = -1
    ut_max: float = -1
    # Matrix saving
    matrix_id: str = ""
    rhs_id: str = ""
    state_id: str = ""
    iterate_id: str = ""


@dataclass
class TimeStepStats:
    linear_solves: list[LinearSolveStats] = field(default_factory=list)
    nonlinear_convergence_status: int = 1  # 1 converged -1 diverged

    @classmethod
    def from_json(cls, json: str):
        data = cls(**json)
        tmp = []
        for x in data.linear_solves:
            payload = {
                k: v for k, v in x.items() if k in LinearSolveStats.__dataclass_fields__
            }
            tmp.append(LinearSolveStats(**payload))
        data.linear_solves = tmp
        return data
