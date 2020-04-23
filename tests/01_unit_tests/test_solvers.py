import time
from typing import List

from flatland.action_plan.action_plan import ActionPlanDict
from flatland.action_plan.action_plan import ActionPlanElement
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.action_plan.action_plan_player import ControllerFromTrainrunsReplayer
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import DummyPredictorForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail
from flatland.utils.simple_rail import make_simple_rail_with_alternatives

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.route_dag.generators.route_dag_generator_schedule import schedule_problem_description_from_rail_env
from rsp.utils.flatland_replay_utils import create_controller_from_trainruns_and_malfunction
from rsp.utils.flatland_replay_utils import make_render_call_back_for_replay
# ----- EXPECTATIONS (solver-specific) ----------------


def test_simple_rail_asp_two_agents_without_loop():
    # minimize sum of travel times over all agents!
    expected_action_plan = [[
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=1, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=2, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=3, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=4, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=8, action=RailEnvActions.STOP_MOVING),

    ], [
        # it takes one additional time step to enter the grid!
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
        # now, we're at the beginning of the cell
        ActionPlanElement(scheduled_at=10, action=RailEnvActions.MOVE_FORWARD),
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

    other_expected_action_plan = [[
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
    ], [ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
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

    _simple_rail_two_agents_without_loop([expected_action_plan, other_expected_action_plan])


def test_simple_rail_asp_two_agents_with_loop():
    # minimize sum of travel times over all agents!
    expected_action_plan = [
        [
            ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=1, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=2, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=3, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=4, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=8, action=RailEnvActions.MOVE_FORWARD),
            ActionPlanElement(scheduled_at=9, action=RailEnvActions.STOP_MOVING)

        ], [
            ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
            ActionPlanElement(scheduled_at=11, action=RailEnvActions.MOVE_FORWARD),
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
    other_expected_action_plan = [[
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
        ActionPlanElement(scheduled_at=13, action=RailEnvActions.STOP_MOVING),

    ], [ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
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

    _simple_rail_wo_agents_with_loop([expected_action_plan, other_expected_action_plan])


def test_simple_rail_asp_two_agents_with_loop_multi_speed(rendering=True):
    # minimize sum of travel times over all agents!
    expected_action_plan = [[
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=1, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=2, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=3, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=4, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=5, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=6, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=7, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=8, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=9, action=RailEnvActions.STOP_MOVING)

    ], [
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
        ActionPlanElement(scheduled_at=11, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=12, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=14, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=16, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=18, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=20, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=22, action=RailEnvActions.MOVE_RIGHT),
        ActionPlanElement(scheduled_at=24, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=26, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=28, action=RailEnvActions.STOP_MOVING),

    ]]
    other_expected_action_plan = [[
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
        ActionPlanElement(scheduled_at=17, action=RailEnvActions.STOP_MOVING)
    ]]

    _simple_rail_wo_agents_with_loop_multi_speed(ASPProblemDescription,
                                                 [expected_action_plan, other_expected_action_plan])


def test_simple_rail_with_alternatives_one_agent(rendering=False):
    expected_action_plans = [
        [[
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
            ActionPlanElement(scheduled_at=17, action=4)
        ]],
        [[
            ActionPlanElement(scheduled_at=0, action=2),
            ActionPlanElement(scheduled_at=1, action=2),
            ActionPlanElement(scheduled_at=2, action=2),
            ActionPlanElement(scheduled_at=3, action=2),
            ActionPlanElement(scheduled_at=4, action=2),
            ActionPlanElement(scheduled_at=5, action=2),
            ActionPlanElement(scheduled_at=6, action=2),
            ActionPlanElement(scheduled_at=7, action=2),
            ActionPlanElement(scheduled_at=8, action=2),
            ActionPlanElement(scheduled_at=9, action=2),
            ActionPlanElement(scheduled_at=10, action=2),
            ActionPlanElement(scheduled_at=11, action=2),
            ActionPlanElement(scheduled_at=12, action=2),
            ActionPlanElement(scheduled_at=13, action=3),
            ActionPlanElement(scheduled_at=14, action=2),
            ActionPlanElement(scheduled_at=15, action=2),
            ActionPlanElement(scheduled_at=16, action=2),
            ActionPlanElement(scheduled_at=17, action=4)
        ]]
    ]
    _simple_rail_wo_agents_with_loop_multi_speed_alternative_routes(expected_action_plans,
                                                                    rendering=rendering)


# ----- SCENARIOS (solver indepenent, for all)  ----------------
def _simple_rail_two_agents_without_loop(expected_action_plans):
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

    controller: ControllerFromTrainruns = _extract_controller_from_train_runs(env)

    _verify(controller, env, expected_action_plans)


def _simple_rail_wo_agents_with_loop(expected_action_plans, ):
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

    controller = _extract_controller_from_train_runs(env)

    _verify(controller, env, expected_action_plans)


def _simple_rail_wo_agents_with_loop_multi_speed(problem_description, expected_action_plans, ):
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

    controller: ControllerFromTrainruns = _extract_controller_from_train_runs(env)

    _verify(controller, env, expected_action_plans)


def _simple_rail_wo_agents_with_loop_multi_speed_alternative_routes(expected_action_plans,
                                                                    rendering: bool = False):
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

    controller = _extract_controller_from_train_runs(env)

    _verify(controller, env, expected_action_plans, rendering=rendering)


# ----- HELPER ------------------------------------------------

def _extract_controller_from_train_runs(env: RailEnv,
                                        k: int = 10) -> ControllerFromTrainruns:
    problem = ASPProblemDescription.factory_scheduling(schedule_problem_description_from_rail_env(env=env, k=k), no_optimize=False)

    start_solver = time.time()
    solution = problem.solve()
    solve_time = (time.time() - start_solver)
    print("solve_time={:5.3f}ms".format(solve_time))
    print(f"solution={solution.get_trainruns_dict()}")
    actual_action_plan: ControllerFromTrainruns = create_controller_from_trainruns_and_malfunction(
        trainrun_dict=solution.get_trainruns_dict(),
        env=env)
    return actual_action_plan


def _verify(controller: ControllerFromTrainruns, env, expected_action_plans: List[ActionPlanDict],
            rendering: bool = False):
    print("expected one of:")
    for expected_action_plan in expected_action_plans:
        ControllerFromTrainruns.print_action_plan_dict(expected_action_plan)
    print("\n\nactual:")
    controller.print_action_plan()

    assert controller.action_plan in expected_action_plans, \
        "expected {}, actual {}".format(expected_action_plans, controller.action_plan)

    ControllerFromTrainrunsReplayer.replay_verify(
        controller, env, call_back=make_render_call_back_for_replay(env=env, rendering=rendering))
