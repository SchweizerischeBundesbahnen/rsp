import numpy as np
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from rsp.scheduling.asp.asp_problem_description import ASPProblemDescription

conv = ASPProblemDescription.convert_position_and_entry_direction_to_waypoint


def test_convert_transition():
    def _helper(actual, expected):
        assert np.array_equal(actual, expected), "found {}, expected {}".format(actual, expected)

    actual = conv(2, 2, Grid4TransitionsEnum.NORTH)
    expected = ((2, 2), 0)
    _helper(actual, expected)

    actual = conv(2, 2, Grid4TransitionsEnum.EAST)
    expected = ((2, 2), 1)
    _helper(actual, expected)

    actual = conv(2, 2, Grid4TransitionsEnum.SOUTH)
    expected = ((2, 2), 2)
    _helper(actual, expected)

    actual = conv(2, 2, Grid4TransitionsEnum.WEST)
    expected = ((2, 2), 3)
    _helper(actual, expected)
