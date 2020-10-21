from typing import List
from typing import NamedTuple

from frozenlist import FrozenList
from rsp.scheduling.asp.asp_data_types import ASPHeuristics

GlobalConstants = NamedTuple(
    "GlobalConstants",
    [
        ("RELEASE_TIME", int),
        ("SCHEDULE_HEURISTICS", List[ASPHeuristics]),
        ("RESCHEDULE_HEURISTICS", List[ASPHeuristics]),
        ("DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY", int),
        ("DELAY_MODEL_PENALTY_AFTER_LINEAR", int),
        ("DELAY_MODEL_RESOLUTION", int),
        ("DL_PROPAGATE_PARTIAL", bool),
        ("NB_RANDOM", int),
    ],
)

DEFAULT_SCHEDULE_HEURISTICS = FrozenList([ASPHeuristics.HEURISTIC_SEQ])
DEFAULT_RESCHEDULE_HEURISTICS = FrozenList([])


def get_defaults(
    release_time=1,
    schedule_heuristics=DEFAULT_SCHEDULE_HEURISTICS,
    reschedule_heuristics=DEFAULT_RESCHEDULE_HEURISTICS,
    delay_model_upper_bound_linear_penalty=60,
    delay_model_penalty_after_linear=5000000,
    delay_model_resolution=1,
    dl_propagate_partial=True,
    nb_random=5,
):
    return GlobalConstants(
        RELEASE_TIME=release_time,
        SCHEDULE_HEURISTICS=list(schedule_heuristics),
        RESCHEDULE_HEURISTICS=list(reschedule_heuristics),
        DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY=delay_model_upper_bound_linear_penalty,
        DELAY_MODEL_PENALTY_AFTER_LINEAR=delay_model_penalty_after_linear,
        DELAY_MODEL_RESOLUTION=delay_model_resolution,
        DL_PROPAGATE_PARTIAL=dl_propagate_partial,
        NB_RANDOM=nb_random,
    )


class GlobalConstantsCls(object):
    _instance = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def set_defaults(cls, constants: GlobalConstants):
        cls._constants = constants
        for key, value in cls._constants._asdict().items():
            setattr(cls, key, value)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            # Put any initialization here.
            cls.set_defaults(constants=get_defaults())
        return cls._instance


# N.B. we need a singleton to be able to access the same object even when the module is imported multiple times!
# N.B. a different singleton lives in every process, so we must reset it appropriately within every process, see `run_experiment_from_to_file`!
GLOBAL_CONSTANTS = GlobalConstantsCls.instance()
