from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.step_04_analysis.detailed_experiment_analysis.flatland_rendering import convert_trainrun_dict_to_train_schedule_dict


def test_convert_trainrundict_to_positions_for_flatland_timesteps():
    trainrun_dict: TrainrunDict = {
        0: [
            # before None
            TrainrunWaypoint(waypoint=Waypoint(position=(0, 0), direction=0), scheduled_at=3),
            # after (0,1) since next scheduled_at is 4 == 3+1
            # before (0,1) # NOQA
            TrainrunWaypoint(waypoint=Waypoint(position=(0, 1), direction=0), scheduled_at=4),
            # after (0,2) since next scheduled_at only at 6 > 4+1
            # before (0,2)  # NOQA
            TrainrunWaypoint(waypoint=Waypoint(position=(0, 2), direction=0), scheduled_at=6),
            # after None since no next scheduled_at
        ]
    }
    expected = {
        0: {
            0: None,
            1: None,
            2: None,
            # moved to next cell after time step 3!
            3: Waypoint(position=(0, 1), direction=0),
            # not moved further after time step 4 yet!
            4: Waypoint(position=(0, 1), direction=0),
            # reached target at next step -> FLATland already has None after step 5!
            5: None,
        }
    }
    actual = convert_trainrun_dict_to_train_schedule_dict(trainrun_dict)

    assert actual == expected, f"actual={actual}, expected={expected}"

    print(
        convert_trainrun_dict_to_train_schedule_dict(
            {
                11: [
                    TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(23, 23), direction=5)),
                    TrainrunWaypoint(scheduled_at=1, waypoint=Waypoint(position=(23, 23), direction=1)),
                    TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 24), direction=1)),
                    TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 25), direction=1)),
                    TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 26), direction=1)),
                    TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(23, 27), direction=1)),
                    TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(23, 28), direction=1)),
                ]
            }
        )
    )
