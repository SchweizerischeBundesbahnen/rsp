import time
from typing import Dict, Optional, List, Callable

from flatland.action_plan.action_plan import ActionPlanElement, ControllerFromTrainruns
from flatland.action_plan.action_plan_player import ControllerFromTrainrunsReplayer
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv, DummyPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail, make_simple_rail_with_alternatives

from solver.abstract_problem_description import AbstractProblemDescription
from solver.asp.asp_problem_description import ASPProblemDescription
from solver.googleortools.cp_sat_solver import CPSATSolver
from solver.googleortools.ortools_problem_description import ORToolsProblemDescription


def generate_ortools_cpsat_problem_description(env: RailEnv,
                                               agents_path_dict: Dict[int, Optional[List[Waypoint]]]):
    return ORToolsProblemDescription(env=env,
                                     solver=CPSATSolver(),
                                     agents_path_dict=agents_path_dict)


# ----- EXPECTATIONS (solver-specific) ----------------
def test_simple_rail_asp_two_agents_without_loop():
    # minimize sum of travel times over all agents!
    expected_action_plan = [[
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
        ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=8, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=9, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=10, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=11, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=12, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=13, action=RailEnvActions.STOP_MOVING),

    ], [
        # it takes one additional time step to enter the grid!
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
        # now, we're at the beginning of the cell
        ActionPlanElement(scheduled_at=1, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=2, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=3, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=4, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_RIGHT),
        ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=8, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=9, action=RailEnvActions.STOP_MOVING),
    ]]

    _simple_rail_two_agents_without_loop(ASPProblemDescription, expected_action_plan)


def test_simple_rail_ortools_two_agents_without_loop():
    expected_action_plan = [
        [

            # agent 0 enters the grid
            ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
            # now at the beginning of the cell
            ActionPlanElement(scheduled_at=1, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=2, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=3, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=4, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=8, action=RailEnvActions.STOP_MOVING),
        ],
        [
            ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
            # enter the grid (first agent removed at step 8 plus 1 release time because of agent ordering)
            # TODO  fix ortools, it should be 9!
            ActionPlanElement(scheduled_at=10, action=RailEnvActions.MOVE_FORWARD),
            # enter the cell
            ActionPlanElement(scheduled_at=11, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=12, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=13, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=14, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=15, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=16, action=RailEnvActions.MOVE_RIGHT),
            ActionPlanElement(scheduled_at=17, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=18, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=19, action=RailEnvActions.STOP_MOVING),
        ]]
    _simple_rail_two_agents_without_loop(generate_ortools_cpsat_problem_description, expected_action_plan)


def test_simple_rail_asp_two_agents_with_loop():
    # minimize sum of travel times over all agents!
    expected_action_plan = [
        [
            ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
            ActionPlanElement(scheduled_at=4, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=8, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=9, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=10, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=11, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=12, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=13, action=RailEnvActions.STOP_MOVING)

        ], [
            ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=1, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=2, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=3, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=4, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_RIGHT),
            ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=8, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=9, action=RailEnvActions.STOP_MOVING),

        ]]
    _simple_rail_wo_agents_with_loop(ASPProblemDescription, expected_action_plan)


def test_simple_rail_ortools_two_agents_with_loop():
    expected_action_plan = [[
        # enter the grid
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
        # at the beinning of the cell
        ActionPlanElement(scheduled_at=1, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=2, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=3, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=4, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=8, action=RailEnvActions.MOVE_FORWARD),
        # agent arrives at (3,8) at time 9 and is immediately removed
        ActionPlanElement(scheduled_at=9, action=RailEnvActions.STOP_MOVING)

    ], [
        # wait
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),

        # enter the grid (first agent removed at step 9 plus 1 release time because of agent ordering)
        # TODO fix ortools, it should be 10!!!
        ActionPlanElement(scheduled_at=11, action=RailEnvActions.MOVE_FORWARD),
        # at the beginning of the cell
        ActionPlanElement(scheduled_at=12, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=13, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=14, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=15, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=16, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=17, action=RailEnvActions.MOVE_RIGHT),
        ActionPlanElement(scheduled_at=18, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=19, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=20, action=RailEnvActions.STOP_MOVING),

    ]]
    _simple_rail_wo_agents_with_loop(generate_ortools_cpsat_problem_description, expected_action_plan)


def test_simple_rail_asp_two_agents_with_loop_multi_speed(rendering=True):
    # minimize sum of travel times over all agents!
    expected_action_plan = [[
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
        ActionPlanElement(scheduled_at=10, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=11, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=12, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=13, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=14, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=15, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=16, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=17, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=18, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=19, action=RailEnvActions.STOP_MOVING)

    ], [
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=1, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=3, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=9, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=11, action=RailEnvActions.MOVE_RIGHT),
        ActionPlanElement(scheduled_at=13, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=15, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=17, action=RailEnvActions.STOP_MOVING),

    ]]
    _simple_rail_wo_agents_with_loop_multi_speed(ASPProblemDescription, expected_action_plan)


def test_simple_rail_with_alternatives_one_agent(rendering=True):
    expected_action_plan = [[
        ActionPlanElement(scheduled_at=0, action=2),
        ActionPlanElement(scheduled_at=1, action=2),
        ActionPlanElement(scheduled_at=2, action=2),
        ActionPlanElement(scheduled_at=3, action=2),
        ActionPlanElement(scheduled_at=4, action=2),
        ActionPlanElement(scheduled_at=5, action=1),
        ActionPlanElement(scheduled_at=6, action=2),
        ActionPlanElement(scheduled_at=7, action=2),
        ActionPlanElement(scheduled_at=8, action=2),
        ActionPlanElement(scheduled_at=9, action=2),
        ActionPlanElement(scheduled_at=10, action=2),
        ActionPlanElement(scheduled_at=11, action=2),
        ActionPlanElement(scheduled_at=12, action=2),
        ActionPlanElement(scheduled_at=13, action=2),
        ActionPlanElement(scheduled_at=14, action=2),
        ActionPlanElement(scheduled_at=15, action=2),
        ActionPlanElement(scheduled_at=16, action=2),
        ActionPlanElement(scheduled_at=17, action=4), ]]
    _simple_rail_wo_agents_with_loop_multi_speed_alternative_routes(ASPProblemDescription, expected_action_plan)


# ----- SCENARIOS (solver indepenent, for all)  ----------------
def _simple_rail_two_agents_without_loop(problem_description, expected_action_plan, ):
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=77),
                  number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  remove_agents_at_target=True
                  )
    env.reset()
    env.agents[0].initial_position = (3, 1)
    env.agents[0].target = (3, 8)
    env.agents[0].initial_direction = Grid4TransitionsEnum.EAST
    env.agents[1].initial_position = (3, 8)
    env.agents[1].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].target = (0, 3)
    env.reset(False, False, False)
    for handle, agent in enumerate(env.agents):
        print("[{}] {} -> {}".format(handle, agent.initial_position, agent.target))

    actual_action_plan: ControllerFromTrainrunsReplayer = _extract_action_plan_replayer(env, problem_description)

    print("expected:")
    ControllerFromTrainruns.print_action_plan_dict(expected_action_plan)
    print("\n\nactual:")
    actual_action_plan.print_action_plan()

    ControllerFromTrainruns.assert_actions_plans_equal(expected_action_plan, actual_action_plan.action_plan)
    assert actual_action_plan.action_plan == expected_action_plan, \
        "expected {}, actual {}".format(expected_action_plan, actual_action_plan)

    ControllerFromTrainrunsReplayer.replay_verify(actual_action_plan, env)


def _simple_rail_wo_agents_with_loop(problem_description, expected_action_plan, ):
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=77),
                  number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  remove_agents_at_target=True
                  )
    env.reset()
    env.agents[0].initial_position = (3, 0)
    env.agents[0].target = (3, 8)
    env.agents[0].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].initial_position = (3, 8)
    env.agents[1].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].target = (0, 3)
    env.reset(False, False, False)
    for handle, agent in enumerate(env.agents):
        print("[{}] {} -> {}".format(handle, agent.initial_position, agent.target))

    actual_action_plan = _extract_action_plan_replayer(env, problem_description)

    print("expected:")
    ControllerFromTrainruns.print_action_plan_dict(expected_action_plan)
    print("\n\nactual:")
    actual_action_plan.print_action_plan()

    ControllerFromTrainruns.assert_actions_plans_equal(expected_action_plan, actual_action_plan.action_plan)
    assert actual_action_plan.action_plan == expected_action_plan, \
        "expected {}, found {}".format(expected_action_plan, actual_action_plan)

    ControllerFromTrainrunsReplayer.replay_verify(actual_action_plan, env)


def _simple_rail_wo_agents_with_loop_multi_speed(problem_description, expected_action_plan, ):
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=77),
                  number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  remove_agents_at_target=True
                  )
    env.reset()
    env.agents[0].initial_position = (3, 0)
    env.agents[0].target = (3, 8)
    env.agents[0].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].initial_position = (3, 8)
    env.agents[1].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].target = (0, 3)
    env.agents[1].speed_data['speed'] = 0.5  # two
    env.reset(False, False, False)
    for handle, agent in enumerate(env.agents):
        print("[{}] {} -> {}".format(handle, agent.initial_position, agent.target))

    actual_action_plan = _extract_action_plan_replayer(env, problem_description)

    print("expected:")
    ControllerFromTrainruns.print_action_plan_dict(expected_action_plan)
    print("\n\nactual:")
    actual_action_plan.print_action_plan()

    ControllerFromTrainruns.assert_actions_plans_equal(expected_action_plan, actual_action_plan.action_plan)
    assert actual_action_plan.action_plan == expected_action_plan, \
        "expected {}, found {}".format(expected_action_plan, actual_action_plan.action_plan)

    ControllerFromTrainrunsReplayer.replay_verify(actual_action_plan, env)


def _simple_rail_wo_agents_with_loop_multi_speed_alternative_routes(problem_description, expected_action_plan,
                                                                    ):
    rail, rail_map = make_simple_rail_with_alternatives()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=DummyPredictorForRailEnv(max_depth=10)),
                  )
    env.reset()

    initial_position = (3, 1)  # west dead-end
    initial_direction = Grid4TransitionsEnum.WEST  # west
    target_position = (3, 9)  # east

    # set the initial position
    agent = env.agents[0]
    agent.position = initial_position
    agent.initial_position = initial_position
    agent.initial_direction = initial_direction
    agent.target = target_position  # east dead-end
    agent.moving = True

    env.reset(False, False)
    for handle, agent in enumerate(env.agents):
        print("[{}] {} -> {}".format(handle, agent.initial_position, agent.target))

    actual_action_plan = _extract_action_plan_replayer(env, problem_description, 10)

    actual_action_plan.print_action_plan()
    ControllerFromTrainruns.assert_actions_plans_equal(expected_action_plan, actual_action_plan.action_plan)
    assert actual_action_plan.action_plan == expected_action_plan, \
        "expected {}, found {}".format(expected_action_plan, actual_action_plan.action_plan)

    ControllerFromTrainrunsReplayer.replay_verify(actual_action_plan, env)
    


# ----- HELPER ------------------------------------------------
ProblemDescriptionConstructor = Callable[
    [RailEnv, Optional[Dict[int, List[List[Waypoint]]]], int, bool], AbstractProblemDescription]


def _extract_action_plan_replayer(env: RailEnv, problem_description_constructor,
                                  k: int = 1) -> ControllerFromTrainrunsReplayer:
    agents_paths_dict = {
        i: get_k_shortest_paths(env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                k, debug=False) for i, agent in enumerate(env.agents)
    }
    problem = problem_description_constructor(env=env, agents_path_dict=agents_paths_dict)
    start_solver = time.time()
    solution = problem.solve()
    solve_time = (time.time() - start_solver)
    print("solve_time={:5.3f}ms".format(solve_time))
    actual_action_plan: ControllerFromTrainrunsReplayer = solution.create_action_plan()
    return actual_action_plan
