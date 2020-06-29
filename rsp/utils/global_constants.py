# TODO add these constants to experiment_agenda?
from rsp.experiment_solvers.asp.data_types import ASPHeuristics

RELEASE_TIME = 1

SCHEDULE_HEURISTICS = [ASPHeuristics.HEURISTIC_SEQ]
RESCHEDULE_HEURISTICS = []
DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY = 60
DELAY_MODEL_PENALTY_AFTER_LINEAR = 5000000
DELAY_MODEL_RESOLUTION = 1

DL_PROPAGATE_PARTIAL = True
