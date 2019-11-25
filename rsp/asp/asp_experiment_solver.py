from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths

from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.asp.asp_solution_description import ASPSolutionDescription
from utils.data_types import ExperimentResults
from utils.experiment_render_utils import init_renderer_for_env, render_env, cleanup_renderer_for_env
from utils.experiment_solver import AbstractSolver
from utils.experiment_utils import solve_problem, replay_until_malfunction


class ASPExperimentSolver(AbstractSolver):
    """
    Implements `AbstractSolver` for ASP.

    Methods
    -------
    run_experiment_trial:
        Returns the correct data format to run tests on full research pipeline
    """

    def run_experiment_trial(
            self,
            static_rail_env: RailEnv,
            malfunction_rail_env: RailEnv,
            malfunction_env_reset,
            k: int = 10,
            disable_verification_by_replay: bool = False,
            verbose: bool = False,
            rendering: bool = False
    ) -> ExperimentResults:
        """
        Runs the experiment.

        Parameters
        ----------
        static_rail_env: RailEnv
            Rail environment without any malfunction
        malfunction_rail_env: RailEnv
            Rail environment with one single malfunction

        Returns
        -------
        ExperimentResults
        """
        # --------------------------------------------------------------------------------------
        # Generate k shortest paths
        # --------------------------------------------------------------------------------------
        # TODO add method to FLATland to create of k shortest paths for all agents
        agents_paths_dict = {
            i: get_k_shortest_paths(static_rail_env,
                                    agent.initial_position,
                                    agent.initial_direction,
                                    agent.target,
                                    k) for i, agent in enumerate(static_rail_env.agents)
        }

        # --------------------------------------------------------------------------------------
        # Produce a full schedule
        # --------------------------------------------------------------------------------------
        schedule_problem = ASPProblemDescription(env=static_rail_env,
                                                 agents_path_dict=agents_paths_dict)

        renderer = init_renderer_for_env(static_rail_env, rendering)

        def render(test_id: int, solver_name, i_step: int):
            render_env(renderer, test_id, solver_name, i_step)

        schedule_result = solve_problem(
            env=static_rail_env,
            problem=schedule_problem,
            agents_paths_dict=agents_paths_dict,
            rendering_call_back=render,
            debug=False)
        cleanup_renderer_for_env(renderer)

        schedule_solution: ASPSolutionDescription = schedule_result.solution

        if verbose:
            print("schedule_solution=")
            print(schedule_solution.get_trainruns_dict())

        # --------------------------------------------------------------------------------------
        # Generate malfuntion
        # --------------------------------------------------------------------------------------

        malfunction_env_reset()
        malfunction = replay_until_malfunction(solution=schedule_solution, env=malfunction_rail_env)
        malfunction_env_reset()
        if verbose:
            print(f"malfunction={malfunction}")

        if not malfunction:
            raise Exception("Could not produce a malfunction")

        if verbose:
            print(f"malfunction={malfunction}")

        # --------------------------------------------------------------------------------------
        # Re-schedule Full
        # --------------------------------------------------------------------------------------
        full_reschedule_problem: ASPProblemDescription = schedule_problem.get_freezed_copy_for_rescheduling(
            malfunction=malfunction,
            trainruns_dict=schedule_solution.get_trainruns_dict()
        )

        renderer = init_renderer_for_env(malfunction_rail_env, rendering)

        def render(test_id: int, solver_name, i_step: int):
            render_env(renderer, test_id, solver_name, i_step)

        full_reschedule_result = solve_problem(
            env=malfunction_rail_env,
            problem=full_reschedule_problem,
            agents_paths_dict=agents_paths_dict,
            rendering_call_back=render,
            debug=False,
            malfunction=malfunction
        )
        cleanup_renderer_for_env(renderer)
        malfunction_env_reset()
        full_reschedule_solution: ASPSolutionDescription = full_reschedule_result.solution

        # TODO assert that everything is the same up to freezing point

        if verbose:
            print("full re-schedule_solution=")
            print(full_reschedule_solution.get_trainruns_dict())

        # --------------------------------------------------------------------------------------
        # Re-schedule Delta
        # --------------------------------------------------------------------------------------
        # TODO SIM-105 finish/refactor delta, how to represent delta?
        delta_reschedule_problem: ASPProblemDescription = schedule_problem.get_freezed_copy_for_rescheduling(
            malfunction=malfunction,
            trainruns_dict=schedule_solution.get_trainruns_dict()
        )

        renderer = init_renderer_for_env(malfunction_rail_env, rendering)

        def render(test_id: int, solver_name, i_step: int):
            render_env(renderer, test_id, solver_name, i_step)

        delta_reschedule_result = solve_problem(
            env=malfunction_rail_env,
            problem=delta_reschedule_problem,
            agents_paths_dict=agents_paths_dict,
            rendering_call_back=render,
            debug=False,
            malfunction=malfunction)
        cleanup_renderer_for_env(renderer)
        malfunction_env_reset()
        delta_reschedule_solution: ASPSolutionDescription = delta_reschedule_result.solution

        if verbose:
            print("delta re-schedule_solution=")
            print(delta_reschedule_solution.get_trainruns_dict())

        # TODO analyse running times (grounding vs solving - etc.)
        # TODO display sum of delays in both approaches

        # --------------------------------------------------------------------------------------
        # Result
        # --------------------------------------------------------------------------------------
        current_results = ExperimentResults(time_full=schedule_result.solve_time,
                                            time_full_after_malfunction=full_reschedule_result.solve_time,
                                            time_delta_after_malfunction=delta_reschedule_result.solve_time,
                                            solution_full=schedule_solution.get_trainruns_dict(),
                                            solution_delta=[],
                                            delta=[])
        return current_results
