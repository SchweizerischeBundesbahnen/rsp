import pprint

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.trainrun_utils import verify_trainrun_dict_simple
from rsp.utils.flatland_replay_utils import extract_trainrun_dict_from_flatland_positions


def test_extract_trainrun_dict_from_flatland_positions():
    initial_directions = {
        0: 2,
        1: 4}
    initial_positions = {
        0: (0, 0),
        1: (4, 0)}
    targets = {
        0: (2, 0),
        1: (2, 0)
    }
    actual_trainrun_dict = extract_trainrun_dict_from_flatland_positions(
        initial_directions,
        initial_positions=initial_positions,
        schedule={
            0: {
                0: Waypoint(position=None, direction=2),
                1: Waypoint(position=None, direction=4),
            },
            1: {
                0: Waypoint(position=(0, 0), direction=2),
                1: Waypoint(position=(4, 0), direction=4),
            },
            2: {
                0: Waypoint(position=(1, 0), direction=2),
                1: Waypoint(position=(3, 0), direction=4),
            },
            3: {
                0: Waypoint(position=None, direction=2),
                1: Waypoint(position=(3, 0), direction=4),
            },
            4: {
                0: Waypoint(position=None, direction=2),
                1: Waypoint(position=None, direction=4),
            }

        },
        targets=targets

    )
    expected_trainrun_dict = {
        0: [
            TrainrunWaypoint(waypoint=Waypoint(position=(0, 0), direction=5), scheduled_at=0),
            TrainrunWaypoint(waypoint=Waypoint(position=(0, 0), direction=2), scheduled_at=1),
            TrainrunWaypoint(waypoint=Waypoint(position=(1, 0), direction=2), scheduled_at=2),
            TrainrunWaypoint(waypoint=Waypoint(position=(2, 0), direction=2), scheduled_at=3)
        ],
        1: [
            TrainrunWaypoint(waypoint=Waypoint(position=(4, 0), direction=5), scheduled_at=0),
            TrainrunWaypoint(waypoint=Waypoint(position=(4, 0), direction=4), scheduled_at=1),
            TrainrunWaypoint(waypoint=Waypoint(position=(3, 0), direction=4), scheduled_at=2),
            TrainrunWaypoint(waypoint=Waypoint(position=(2, 0), direction=4), scheduled_at=4)
        ]
    }
    _pp = pprint.PrettyPrinter(indent=4)
    assert expected_trainrun_dict == actual_trainrun_dict, f"\nactual={_pp.pformat(actual_trainrun_dict)},\nexpected={_pp.pformat(expected_trainrun_dict)}"
    if False:
        # TODO in rsp, we spend one tick too long in target cell - will we have a problem when taking FLATland solutions?
        #   - entering the target cell at t -> release time blocks until t+1 (next entry possible at t+1)
        #   - entering the dummy node in the target cell at t+1 -> release time blocks again, next entry only possible at t+2 instead of t+1!
        verify_trainrun_dict_simple(
            trainrun_dict=actual_trainrun_dict,
            minimum_runningtime_dict={
                0: 1,
                1: 1
            },
            initial_directions=initial_directions,
            initial_positions=initial_positions,
            targets=targets
        )
