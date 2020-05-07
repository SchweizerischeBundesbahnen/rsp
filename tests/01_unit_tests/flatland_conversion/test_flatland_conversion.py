import pprint

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.flatland_integration.flatland_conversion import extract_trainrun_dict_from_flatland_positions


def test_extract_trainrun_dict_from_flatland_positions():
    actual_trainrun_dict = extract_trainrun_dict_from_flatland_positions(
        initial_directions={0: 2,
                            1: 4},
        initial_positions={0: (0, 0),
                           1: (4, 0)},
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
        targets={
            0: (2, 0),
            1: (2, 0)
        }

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

    # TODO SIM-434 verify trainrundict
