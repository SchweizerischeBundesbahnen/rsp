"""
Data types used in the experiment for the real time rescheduling research project

"""
from typing import NamedTuple, List, Dict

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint, TrainrunDict

ExperimentParameters = NamedTuple('ExperimentParameters',
                                  [('experiment_id', int),
                                   ('trials_in_experiment', int),
                                   ('number_of_agents', int),
                                   ('speed_data', Dict[float]),
                                   ('width', int),
                                   ('height', int),
                                   ('seed_value', int),
                                   ('max_num_cities', int),
                                   ('grid_mode', bool),
                                   ('max_rail_between_cities', int),
                                   ('max_rail_in_city', int),
                                   ('earliest_malfunction', int),
                                   ('malfunction_duration', int)])

ExperimentAgenda = NamedTuple('ExperimentAgenda', [('experiments', List[ExperimentParameters])])

ExperimentResults = NamedTuple('ExperimentResults', [
    ('time_full', float),
    ('time_full_after_malfunction', float),
    ('time_delta_after_malfunction', float),
    ('solution_full', TrainrunDict),
    ('solution_full_after_malfunction', TrainrunDict),
    ('solution_delta_after_malfunction', TrainrunDict),
    ('costs_full', float),  # sum of travelling times in scheduling solution
    ('costs_full_after_malfunction', float),  # total delay at target over all agents with respect to schedule
    ('costs_delta_after_malfunction', float),  # total delay at target over all agents with respect to schedule
    ('delta', Dict[int, List[TrainrunWaypoint]])
    # train run way points in the full re-schedule that are not in the full schedule
])

ParameterRanges = NamedTuple('ParameterRanges', [('size_range', List[int]),
                                                 ('agent_range', List[int]),
                                                 ('in_city_rail_range', List[int]),
                                                 ('out_city_rail_range', List[int]),
                                                 ('city_range', List[int]),
                                                 ('earliest_malfunction', List[int]),
                                                 ('malfunction_duration', List[int])
                                                 ])

Malfunction = NamedTuple('Malfunction', [
    ('time_step', int),
    ('agent_id', int),
    ('malfunction_duration', int)
])
