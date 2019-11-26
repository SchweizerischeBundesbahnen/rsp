import time

import pytest
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail
from importlib_resources import path

from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.asp.asp_solver import _asp_helper, flux_helper


def test_asp_helper():
    with path('tests.data.asp.instances', 'dummy.lp') as instance_in:
        with path('res.asp.encodings', 'encoding.lp') as encoding_in:
            models, _, _, _ = _asp_helper([instance_in, encoding_in])

    print(models)
    assert len(models) == 1
    actual = list(models)[0]

    expected = set([
        'start(t1,1)',
        'm((1,4),1)',
        'visit(t1,4)',
        'e(t1,1,0)',
        'l(t1,1,6)',
        'visit(t1,1)',
        'edge(t1,1,4)',
        'route(t1,(1,4))',
        'w(t1,(1,4),0)',
        'e(t1,4,0)',
        'l(t1,4,6)',
        'dl((t1,4),1)',
        'end(t1,4)',
        'train(t1)',
        'dl((t1,1),0)'
    ])
    assert actual == expected, "actual {}, expected {}".format(actual, expected)


def test_mutual_exclusion():
    with path('tests.data.asp.instances', 'dummy_two_agents.lp') as instance_in:
        with path('res.asp.encodings', 'encoding.lp') as encoding_in:
            models, statistics, _, _ = _asp_helper([instance_in, encoding_in])
    assert int(statistics['summary']['models']['enumerated']) == 1
    first_actual = list(models)[0]
    expected_dl = set(['dl((t1,4),8)', 'dl((t2,4),0)', 'dl((t2,1),1)', 'dl((t1,1),6)'])
    assert first_actual.issuperset(expected_dl), "expected {} to be subset of {}".format(expected_dl, first_actual)


def test_simple_rail_asp_one_agent():
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.reset()
    env.agents[0].initial_position = (3, 1)
    env.agents[0].initial_direction = Grid4TransitionsEnum.EAST
    env.agents[0].target = (3, 5)
    env.reset(False, False, True)

    agents_paths_dict = {
        i: get_k_shortest_paths(env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                1) for i, agent in enumerate(env.agents)
    }

    start_solver = time.time()
    problem = ASPProblemDescription(env=env, agents_path_dict=agents_paths_dict)

    print(problem.asp_program)
    models, stats, _, _ = flux_helper(problem.asp_program)

    solve_time = (time.time() - start_solver)
    print("solve_time={:5.3f}ms".format(solve_time))

    assert int(stats['summary']['models']['enumerated']) == 1
    actual_answer_set = list(models)[0]

    expected = set(["dl((t0,((3,1),1)),0)",
                    "dl((t0,((3,2),1)),2)",
                    "dl((t0,((3,3),1)),3)",
                    "dl((t0,((3,4),1)),4)",
                    "dl((t0,((3,5),1)),5)",
                    "dl((t0,((3,5),5)),6)"
                    ])
    agent_path = agents_paths_dict[0][0]
    assert agent_path == (Waypoint(position=(3, 1), direction=1), Waypoint(position=(3, 2), direction=1),
                          Waypoint(position=(3, 3), direction=1), Waypoint(position=(3, 4), direction=1),
                          Waypoint(position=(3, 5), direction=1))

    assert actual_answer_set.issuperset(expected), \
        "expected={} should be subset of actual={}".format(expected, actual_answer_set)

    solution: ASPSolutionDescription = problem.solve()
    print(solution.asp_solution.answer_sets)


def test_asp_helper_forcing():
    """Case study to freeze variables by adding facts."""
    with path('tests.data.asp.instances', 'dummy_forced.lp') as instance_in:
        with path('res.asp.encodings', 'encoding.lp') as encoding_in:
            models, _, _, _ = _asp_helper([instance_in, encoding_in])

    print(models)
    assert len(models) == 1
    actual = list(models)[0]

    expected = set([
        'e(t1,1,0)',
        'l(t1,1,6)',
        'l(t1,4,6)',
        'start(t1,1)',
        'dl((t1,1),2)',
        'w(t1,(1,4),0)',
        'visit(t1,1)',
        'edge(t1,1,4)',
        'e(t1,4,0)',
        'train(t1)',
        'm((1,4),2)',
        'route(t1,(1,4))',
        'dl((t1,4),4)',
        'visit(t1,4)',
        'end(t1,4)'
    ])
    assert actual == expected, "actual {}, expected {}".format(actual, expected)


def test_minimize_sum_of_running_times_scheduling():
    """Case Study how to model minimizing sum of running times in a non optimal way."""
    encodings = []
    with path('tests.data.asp.instances', 'dummy_two_agents_minimize_sum_of_running_times.lp') as instance_in:
        encodings.append(instance_in)
    with path('res.asp.encodings', 'encoding.lp') as encoding_in:
        encodings.append(encoding_in)
    with path('res.asp.encodings', 'minimize_total_sum_of_running_times.lp') as encoding_in:
        encodings.append(encoding_in)
    models, all_statistics, _, _ = _asp_helper(encodings)

    assert len(models) == 1
    actual = list(models)[0]
    dls = list(filter(lambda x: x.startswith("dl"), actual))
    print(actual)
    print(dls)

    assert all_statistics['summary']['costs'][0] == 0, "found {}".format(all_statistics[0]['summary']['costs'][0])
    assert all_statistics['summary']['models']['enumerated'] == 1

    expected = set([
        'dl((t1,1),2)',
        'dl((t1,2),4)',
        'dl((t1,3),6)',
        'dl((t2,4),0)',
        'dl((t2,2),2)',
        'dl((t2,3),4)'
    ])
    assert expected.issubset(actual), "actual {}, expected {}".format(actual, expected)


@pytest.mark.skip(reason="Currently disable, does not work under Linux, see https://issues.sbb.ch/browse/SIM-149")
def test_minimize_delay_rescheduling():
    """Case Study how to model minimizing delay with respect to given schedule and a malfunction delay."""
    encodings = []
    with path('tests.data.asp.instances', 'dummy_two_agents_rescheduling.lp') as instance_in:
        encodings.append(instance_in)
    with path('res.asp.encodings', 'encoding.lp') as encoding_in:
        encodings.append(encoding_in)
    with path('res.asp.encodings', 'delay_linear.lp') as encoding_in:
        encodings.append(encoding_in)
    with path('res.asp.encodings', 'minimize_delay.lp') as encoding_in:
        encodings.append(encoding_in)
    models, all_statistics, _, _ = _asp_helper(encoding_files=encodings)
    assert len(models) == 1
    actual = list(models)[0]
    lates = list(filter(lambda x: "late" in x, actual))
    print(lates)
    dls = list(filter(lambda x: x.startswith("dl"), actual))
    print(dls)

    assert all_statistics['summary']['costs'][0] == 6, "found {}".format(all_statistics['summary']['costs'][0])
    assert all_statistics['summary']['models']['enumerated'] == 1

    expected = set([
        'dl((t1,1),0)',
        'dl((t1,2),10)',
        'dl((t1,3),14)',
        'dl((t2,4),0)',
        'dl((t2,2),8)',
        'dl((t2,3),10)'
    ])
    assert expected.issubset(actual), "actual {}, expected {}".format(actual, expected)
