import time

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.utils.simple_rail import make_simple_rail

from solver.googleortools.ortools_solution_description import ORToolsSolutionDescription
from solver.googleortools.ortools_utils import make_variable_name_agent_at_waypoint
from solver.abstract_problem_description import AbstractProblemDescription
from solver.googleortools.cp_sat_solver import CPSATSolver
from solver.googleortools.ortools_problem_description import ORToolsProblemDescription


def test_simple_rail_CPSAT(rendering=False):
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.reset()
    agents_paths_dict = {
        i: get_k_shortest_paths(env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                1) for i, agent in enumerate(env.agents)
    }
    print(agents_paths_dict)
    if rendering:
        renderer = RenderTool(env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=True,
                              clear_debug_text=True,
                              screen_height=1000,
                              screen_width=1000)
        renderer.render_env(show=True, show_observations=False, show_predictions=False)
    start_solver = time.time()

    problem = ORToolsProblemDescription(env=env,
                                        solver=CPSATSolver(),
                                        agents_path_dict=agents_paths_dict)

    solution: ORToolsSolutionDescription = problem.solve()
    solve_time = (time.time() - start_solver)
    print("solve_time={:5.3f}ms".format(solve_time))

    assert solution.is_solved()

    agent_id = 0
    agent_path = agents_paths_dict[agent_id][0]
    agent = env.agents[agent_id]
    actual = []
    for path_loop in range(len(agent_path)):

        wp: Waypoint = agent_path[path_loop]

        entry_waypoint = AbstractProblemDescription.convert_position_and_entry_direction_to_waypoint(
            *wp.position,
            wp.direction)

        if path_loop == 0:
            agent_entry, scheduled_at = solution._get_solver_variable_value(
                make_variable_name_agent_at_waypoint(agent_id, entry_waypoint))
            actual.append((str(agent_entry), scheduled_at))

        if Vec2d.is_equal(agent.target, wp.position):
            break

        next_waypoint: Waypoint = agent_path[path_loop + 1]

        exit_waypoint = AbstractProblemDescription.convert_position_and_entry_direction_to_waypoint(
            *next_waypoint.position, next_waypoint.direction)
        agent_exit, agent_exit_value = solution._get_solver_variable_value(
            make_variable_name_agent_at_waypoint(agent_id, exit_waypoint))
        actual.append((str(agent_exit), agent_exit_value))

    assert len(agent_path) == len(actual)
    expected = [('Var_Agent_ID_0_at_3_0_3_entry', 0),
                ('Var_Agent_ID_0_at_3_1_1_entry', 2),
                ('Var_Agent_ID_0_at_3_2_1_entry', 3),
                ('Var_Agent_ID_0_at_3_3_1_entry', 4),
                ('Var_Agent_ID_0_at_3_4_1_entry', 5),
                ('Var_Agent_ID_0_at_3_5_1_entry', 6),
                ('Var_Agent_ID_0_at_3_6_1_entry', 7),
                ('Var_Agent_ID_0_at_3_7_1_entry', 8),
                ('Var_Agent_ID_0_at_3_8_1_entry', 9)]

    assert actual == expected, "actual={}, expected={}".format(actual, expected)
