from enum import Enum


class ASPObjective(Enum):
    """enum value (key arbitrary) must be the same as encoding to be
    included."""

    # minimize_total_sum_of_running_times.lp
    MINIMIZE_SUM_RUNNING_TIMES = "minimize_total_sum_of_running_times"

    # minimize delay with respect to earliest constraints
    MINIMIZE_DELAY = "minimize_delay"

    # minimize route section penalties
    MINIMIZE_ROUTES = "minimize_routes"  # noqa: F841

    # minimize linear combination of route section penalties and delay
    MINIMIZE_DELAY_ROUTES_COMBINED = "minimize_delay_and_routes_combined"


class ASPHeuristics(Enum):
    """enum value (key arbitrary) must be the same as encoding to be
    included."""

    # avoiding delay at earlier nodes in the paths.
    # NOT USED YET (we do not give the data in re-scheduling yet)
    HEURISTIC_DELAY = "heuristic_DELAY"  # noqa: F841

    # tries to avoid routes where there is a penalty.
    # NOT USED YET (we do not give the data in re-scheduling yet)
    HEURISIC_ROUTES = "heuristic_ROUTES"  # noqa: F841

    # attempts to order conflicting trains by their possible arrival times at the edges where the conflict is located.
    # NOT USED YET (we do not give the data in re-scheduling yet)
    HEURISTIC_SEQ = "heuristic_SEQ"
