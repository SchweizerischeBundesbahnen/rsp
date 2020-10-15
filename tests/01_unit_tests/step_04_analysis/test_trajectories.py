from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import explode_trajectories
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import get_difference_in_time_space_trajectories


def test_explode_trajectories():
    trajectories = {82: [(230, 812), (230, 814), (None, None)]}
    actual = explode_trajectories(trajectories)
    expected = {82: set([(230, 812), (230, 813), (230, 814)])}
    assert actual == expected, f"actual={actual}, expected={expected}"


def test_get_difference_in_time_space_trajectories():
    target_trajectories = {82: [(230, 812), (230, 814), (None, None)]}
    base_trajectories = {82: [(230, 812), (230, 816), (None, None)]}
    actual = get_difference_in_time_space_trajectories(base_trajectories=base_trajectories, target_trajectories=target_trajectories)
    actual_diff = actual.changed_agents
    expected_diff = {82: [(230, 815), (230, 815), (None, None), (230, 816), (230, 816), (None, None)]}
    assert actual_diff == expected_diff, f"actual_diff={actual_diff}, expected_diff={expected_diff}"
