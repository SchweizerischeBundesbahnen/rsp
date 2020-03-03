from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.utils.experiment_render_utils import convert_trainrundict_to_positions_for_all_timesteps


def test_convert_trainrundict_to_positions_for_all_timesteps():
    trainrun_dict: TrainrunDict = {
        0: [
            TrainrunWaypoint(waypoint=Waypoint(position=(0, 0), direction=0), scheduled_at=3),
            TrainrunWaypoint(waypoint=Waypoint(position=(0, 1), direction=0), scheduled_at=4),
            TrainrunWaypoint(waypoint=Waypoint(position=(0, 2), direction=0), scheduled_at=6),
        ]
    }
    expected = {
        0: {
            0: None,
            1: None,
            2: None,
            3: Waypoint(position=(0, 0), direction=0),
            4: Waypoint(position=(0, 1), direction=0),
            5: Waypoint(position=(0, 1), direction=0),
            6: Waypoint(position=(0, 2), direction=0)
        }
    }
    actual = convert_trainrundict_to_positions_for_all_timesteps(trainrun_dict)

    assert actual == expected, f"actual={actual}, expected={expected}"
