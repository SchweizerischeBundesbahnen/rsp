"""Utils used both in solve_envs.py and solve_tests.py."""

import os

import numpy as np
# ----------------------------- Flatland -----------------------------------------------------------
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_generators import rail_from_file, sparse_rail_generator
from flatland.envs.schedule_generators import schedule_from_file, sparse_schedule_generator
# ----------------------------- Helpers -----------------------------------------------------------
from numpy.random.mtrand import RandomState

from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.googleortools.cp_sat_solver import CPSATSolver
from rsp.googleortools.ortools_problem_description import ORToolsProblemDescription
from rsp.utils.experiment_render_utils import init_renderer_for_env, cleanup_renderer_for_env, render_env
from rsp.utils.experiment_utils import solve_problem, current_milli_time
from rsp.utils.general_utils import verification

HEADER = "test_id;fraction_done_agents;total_reward;build_model_time;solve_time;is_solution_optimal;is_solved;" \
         "max_path_len;steps;w;h;num_agents;Solver;model_latest_arrival_time;sum_running_times\n"


# ----------------------------- Auxiliary functions -----------------------------------------------------------
def load_flatland_environment_from_file_with_fixed_seed(file_name):
    rail_generator = rail_from_file(file_name)
    schedule_generator = schedule_from_file(file_name)

    environment = RailEnv(width=1,
                          height=1,
                          rail_generator=rail_generator,
                          number_of_agents=1,
                          schedule_generator=schedule_generator,
                          remove_agents_at_target=True
                          )

    environment.reset(False, False, False, random_seed=1001)

    # reset speed (set all values to one)
    set_agent_speeds_to_one(environment)

    return environment


def create_flatland_environment_sparse(number_of_agents: int,
                                       width: int,
                                       height: int,
                                       seed_value: int,
                                       max_num_cities: int = 5,
                                       grid_mode: bool = False,
                                       max_rails_between_cities: int = 4,
                                       max_rails_in_city: int = 4
                                       ) -> (RailEnv, int):
    rail_generator = sparse_rail_generator(max_num_cities=max_num_cities,
                                           grid_mode=grid_mode,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rails_in_city=max_rails_in_city,
                                           seed=seed_value  # Random seed
                                           )
    schedule_generator = sparse_schedule_generator()

    environment = RailEnv(width=width,
                          height=height,
                          rail_generator=rail_generator,
                          obs_builder_object=DummyObservationBuilder(),
                          number_of_agents=number_of_agents,
                          schedule_generator=schedule_generator,
                          remove_agents_at_target=True
                          )
    environment.reset(random_seed=seed_value)

    return environment


class Bound:
    def __init__(self, min_value: int, max_value: int):
        self.min = min_value
        self.max = max_value


def pull_sparse_env_parameters(width_bound: Bound,
                               height_bound: Bound,
                               number_of_agents_bound: Bound,
                               max_num_cities_bound: Bound,
                               max_rails_between_cities_bound: Bound,
                               max_rails_in_city_bound: Bound,
                               np_random: RandomState) -> (int, int, int, int, int, int):
    def get_parameter_value(bound_value: Bound) -> int:
        rnd_val = np_random.choice(bound_value.max + bound_value.min)
        return rnd_val % max(1, bound_value.max - bound_value.min) + bound_value.min

    number_of_agents = get_parameter_value(number_of_agents_bound)
    width = get_parameter_value(width_bound)
    height = get_parameter_value(height_bound)
    max_num_cities = get_parameter_value(max_num_cities_bound)
    max_rails_between_cities = get_parameter_value(max_rails_between_cities_bound)
    max_rails_in_city = get_parameter_value(max_rails_in_city_bound)

    return number_of_agents, width, height, max_num_cities, max_rails_between_cities, max_rails_in_city


def set_agent_speeds_to_one(env: RailEnv):
    for a in range(env.get_num_agents()):
        env.agents[a].speed_data['speed'] = 1.0


def get_slowest_agent_steps_per_cell(env: RailEnv):
    min_speed = np.inf
    for agent in env.agents:
        speed = agent.speed_data['speed']
        min_speed = min(min_speed, speed)
    return 1.0 / min_speed


def list_files(directory_name):
    r = []
    for root, dirs, files in os.walk(directory_name):

        for name in files:
            if name.endswith(".pkl"):
                r.append(os.path.join(root, name))
    return r


def create_environment_for_test_helper(loop_index):
    seed_value = loop_index + 1
    grid_mode = False
    width_bound = Bound(40, 100)
    height_bound = Bound(40, 100)
    number_of_agents_bound = Bound(1, 10)
    max_num_cities_bound = Bound(3, 10)
    max_rails_between_cities_bound = Bound(2, 4)
    max_rails_in_city_bound = Bound(2, 6)

    np_random_for_params = np.random.RandomState()
    np_random_for_params.seed(seed_value)

    params = \
        pull_sparse_env_parameters(width_bound,
                                   height_bound,
                                   number_of_agents_bound,
                                   max_num_cities_bound,
                                   max_rails_between_cities_bound,
                                   max_rails_in_city_bound,
                                   np_random=np_random_for_params)
    number_of_agents, width, height, max_num_cities, max_rails_between_cities, max_rails_in_city = params
    env = create_flatland_environment_sparse(number_of_agents=number_of_agents,
                                             width=width,
                                             height=height,
                                             seed_value=seed_value,
                                             max_num_cities=max_num_cities,
                                             grid_mode=grid_mode,
                                             max_rails_between_cities=max_rails_between_cities,
                                             max_rails_in_city=max_rails_in_city)

    return env, width, height, number_of_agents


# ----------------------------- Test Helper -----------------------------------------------------------

ORTOOLS_CPSAT = "ortools_CPSAT"
ASP = "ASP"
ASP_ALTERNATIVES = "ASP_ALTERNATIVES"


def test_helper(output_file_name, rendering, tests, debug=False):
    total_reward_agents_array = []
    fraction_done_agents_array = []
    nbr_of_tests = len(tests)
    sbb_results = open(output_file_name, "w")

    print(HEADER)
    sbb_results.writelines([HEADER])

    for loop_index, loop_item in enumerate(tests):

        for solver_name in [ORTOOLS_CPSAT, ASP, ASP_ALTERNATIVES]:
            # --------------------------------------------------------------------------------------
            # Load env
            # --------------------------------------------------------------------------------------
            env, width, height, number_of_agents = create_environment_for_test_helper(loop_index)

            verification("env_grid", env.rail.grid.tolist(), loop_index, solver_name)

            # --------------------------------------------------------------------------------------
            # Generate paths
            # --------------------------------------------------------------------------------------
            k = 1
            if solver_name == ASP_ALTERNATIVES:
                k = 10
            start_shortest_path = current_milli_time()
            agents_paths_dict = {
                i: get_k_shortest_paths(env,
                                        agent.initial_position,
                                        agent.initial_direction,
                                        agent.target,
                                        k) for i, agent in enumerate(env.agents)
            }

            shortest_path_time = (current_milli_time() - start_shortest_path) / 1000.0

            verification("shortest_path_dict", agents_paths_dict, loop_index, solver_name)

            # --------------------------------------------------------------------------------------
            # Generate model
            # --------------------------------------------------------------------------------------
            if solver_name == ORTOOLS_CPSAT:
                problem = ORToolsProblemDescription(env=env,
                                                    solver=CPSATSolver(),
                                                    agents_path_dict=agents_paths_dict)
            elif solver_name == ASP or solver_name == ASP_ALTERNATIVES:
                problem = ASPProblemDescription(env=env,
                                                agents_path_dict=agents_paths_dict)

            else:
                raise Exception("Unexpected solver_name {}".format(solver_name))

            if isinstance(problem, ASPProblemDescription):
                verification("asp_encoding", problem.asp_program, loop_index, solver_name)

            # --------------------------------------------------------------------------------------
            # Solve
            # --------------------------------------------------------------------------------------
            renderer = init_renderer_for_env(env, rendering)

            def render(test_id: int, solver_name, i_step: int):
                render_env(renderer, test_id, solver_name, i_step)

            total_reward, solve_time, build_problem_time, solution = solve_problem(
                problem=problem, loop_index=loop_index, env=env,
                agents_paths_dict=agents_paths_dict,
                rendering_call_back=render, debug=debug)
            cleanup_renderer_for_env(renderer)

            longest_shortest_paths_over_all_agents = np.max(
                [len(agents_paths[0]) for agents_paths in agents_paths_dict.values()])

            # --------------------------------------------------------------------------------------
            # Stats
            # --------------------------------------------------------------------------------------

            # fraction of done-agents
            fraction_done_agents = np.sum([env.dones[i] for i in range(len(env.agents))]) / max(1, len(env.agents))

            # data to add
            fraction_done_agents_array.append(fraction_done_agents)
            total_reward_agents_array.append(total_reward)
            time_step = env._elapsed_steps

            # write out results for test
            x = ['{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
                loop_index,
                fraction_done_agents,
                total_reward,
                solve_time,
                build_problem_time,
                solution.is_optimal_solution(),
                solution.is_solved(),
                longest_shortest_paths_over_all_agents,
                time_step,
                width,
                height,
                number_of_agents,
                solver_name,
                solution.get_model_latest_arrival_time(),
                solution.get_sum_running_times()

            )]
            sbb_results.writelines(x)
            sbb_results.flush()

            if isinstance(loop_item, int):
                format_string = "[{:4f}]"
            else:
                format_string = "[{:4s}]"

            print((format_string.format(loop_item) + "[{:6.1%}] Done: [{:6.2%}]\t"
                                                     "mean reward:{:8.1f}\t"
                                                     "fraction of done: {:6.2%}\t"
                                                     "build model time:{:10.3f}\t"
                                                     "solve time:{:10.3f}\t"
                                                     "shortest path:{:10.3f}"
                                                     "s\t"
                                                     "w:{:5d}\th:{:5d} #Agents:{:5d} \tSolved: [{}]\t Optimal: [{}]\t"
                                                     "Max path len: {}\t steps: {:4d} \t{}\t"
                                                     "model_latest_arrival_time:{:10.3f}\t"
                                                     "sum_running_times:{:10.3f}")
                  .format((1 + loop_index) / nbr_of_tests,
                          fraction_done_agents,
                          np.mean(total_reward_agents_array),
                          np.mean(fraction_done_agents_array),
                          build_problem_time,
                          solve_time,
                          shortest_path_time,
                          width, height,
                          number_of_agents,
                          solution.is_solved(),
                          solution.is_optimal_solution(),
                          longest_shortest_paths_over_all_agents,
                          time_step,
                          solver_name,
                          solution.get_model_latest_arrival_time(),
                          solution.get_sum_running_times()
                          ))

    sbb_results.close()
