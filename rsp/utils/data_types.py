"""Data types used in the experiment for the real time rescheduling research
project."""
import pprint
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from pandas import DataFrame
from pandas import Series

ExperimentFreeze = NamedTuple('ExperimentFreeze', [
    ('freeze_visit', List[TrainrunWaypoint]),
    ('freeze_earliest', Dict[Waypoint, int]),
    ('freeze_latest', Dict[Waypoint, int]),
    ('freeze_banned', List[Waypoint])
])
ExperimentFreezeDict = Dict[int, ExperimentFreeze]

AgentPaths = List[List[Waypoint]]
AgentsPathsDict = Dict[int, AgentPaths]


def experiment_freeze_dict_from_list_of_train_run_waypoint(l: List[TrainrunWaypoint]) -> Dict[TrainrunWaypoint, int]:
    """Generate dictionary of scheduled time at waypoint.

    Parameters
    ----------
    l train run waypoints

    Returns
    -------
    """
    return {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in l}


SpeedData = Mapping[float, float]
ExperimentParameters = NamedTuple('ExperimentParameters',
                                  [('experiment_id', int),
                                   ('experiment_group', int),
                                   ('trials_in_experiment', int),
                                   ('number_of_agents', int),
                                   ('speed_data', SpeedData),
                                   ('width', int),
                                   ('height', int),
                                   ('seed_value', int),
                                   ('max_num_cities', int),
                                   ('grid_mode', bool),
                                   ('max_rail_between_cities', int),
                                   ('max_rail_in_city', int),
                                   ('earliest_malfunction', int),
                                   ('malfunction_duration', int),
                                   ('number_of_shortest_paths_per_agent', int)
                                   ])

ExperimentAgenda = NamedTuple('ExperimentAgenda', [('experiment_name', str),
                                                   ('experiments', List[ExperimentParameters])])

ExperimentMalfunction = NamedTuple('ExperimentMalfunction', [
    ('time_step', int),
    ('agent_id', int),
    ('malfunction_duration', int)
])

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
    ('experiment_freeze_full', ExperimentFreezeDict),
    ('experiment_freeze_full_after_malfunction', ExperimentFreezeDict),
    ('experiment_freeze_delta_after_malfunction', ExperimentFreezeDict),
    ('malfunction', ExperimentMalfunction),
    ('agents_paths_dict', AgentsPathsDict),
    ('nb_resource_conflicts_full', int),
    ('nb_resource_conflicts_full_after_malfunction', int),
    ('nb_resource_conflicts_delta_after_malfunction', int)
])

ParameterRanges = NamedTuple('ParameterRanges', [('size_range', List[int]),
                                                 ('agent_range', List[int]),
                                                 ('in_city_rail_range', List[int]),
                                                 ('out_city_rail_range', List[int]),
                                                 ('city_range', List[int]),
                                                 ('earliest_malfunction', List[int]),
                                                 ('malfunction_duration', List[int]),
                                                 ('number_of_shortest_paths_per_agent', List[int])
                                                 ])

COLUMNS = ['experiment_id',
           'experiment_group',
           'time_full',
           'time_full_after_malfunction',
           'time_delta_after_malfunction',
           'solution_full',
           'solution_full_after_malfunction',
           'solution_delta_after_malfunction',
           'costs_full',
           'costs_full_after_malfunction',
           'costs_delta_after_malfunction',
           'experiment_freeze_full',
           'experiment_freeze_full_after_malfunction',
           'experiment_freeze_delta_after_malfunction',
           'nb_resource_conflicts_full',
           'nb_resource_conflicts_full_after_malfunction',
           'nb_resource_conflicts_delta_after_malfunction',
           'malfunction',
           'agents_paths_dict',
           'size',
           'n_agents',
           'max_num_cities',
           'max_rail_between_cities',
           'max_rail_in_city']


def convert_experiment_results_to_data_frame(experiment_results: ExperimentResults,
                                             experiment_parameters: ExperimentParameters) -> DataFrame:
    """Converts experiment results to data frame.

    Parameters
    ----------
    experiment_results: ExperimentResults
    experiment_parameters: ExperimentParameters

    Returns
    -------
    DataFrame
    """
    return {'experiment_id': experiment_parameters.experiment_id,
            'experiment_group': experiment_parameters.experiment_group,
            'time_full': experiment_results.time_full,
            'time_full_after_malfunction': experiment_results.time_delta_after_malfunction,
            'time_delta_after_malfunction': experiment_results.time_full_after_malfunction,
            'solution_full': experiment_results.solution_full,
            'solution_full_after_malfunction': experiment_results.solution_full_after_malfunction,
            'solution_delta_after_malfunction': experiment_results.solution_delta_after_malfunction,
            'costs_full': experiment_results.costs_full,
            'costs_full_after_malfunction': experiment_results.costs_full_after_malfunction,
            'costs_delta_after_malfunction': experiment_results.costs_delta_after_malfunction,
            'experiment_freeze_full': experiment_results.experiment_freeze_full,
            'experiment_freeze_full_after_malfunction': experiment_results.experiment_freeze_full_after_malfunction,
            'experiment_freeze_delta_after_malfunction': experiment_results.experiment_freeze_delta_after_malfunction,
            'nb_resource_conflicts_full': experiment_results.nb_resource_conflicts_full,
            'nb_resource_conflicts_full_after_malfunction': experiment_results.nb_resource_conflicts_full_after_malfunction,
            'nb_resource_conflicts_delta_after_malfunction': experiment_results.nb_resource_conflicts_delta_after_malfunction,
            'malfunction': experiment_results.malfunction,
            'agents_paths_dict': experiment_results.agents_paths_dict,
            'size': experiment_parameters.width,
            'n_agents': experiment_parameters.number_of_agents,
            'max_num_cities': experiment_parameters.max_num_cities,
            'max_rail_between_cities': experiment_parameters.max_rail_between_cities,
            'max_rail_in_city': experiment_parameters.max_rail_in_city}


def convert_data_frame_row_to_experiment_results(rows: DataFrame) -> ExperimentResults:
    """Converts data frame back to experiment results structure.

    Parameters
    ----------
    rows: DataFrame

    Returns
    -------
    ExperimentResults
    """
    return ExperimentResults(
        time_full=rows['time_full'].iloc[0],
        time_full_after_malfunction=rows['time_full_after_malfunction'].iloc[0],
        time_delta_after_malfunction=rows['time_delta_after_malfunction'].iloc[0],
        solution_full=rows['solution_full'].iloc[0],
        solution_full_after_malfunction=rows['solution_full_after_malfunction'].iloc[0],
        solution_delta_after_malfunction=rows['solution_delta_after_malfunction'].iloc[0],
        costs_full=rows['costs_full'].iloc[0],
        costs_full_after_malfunction=rows['costs_full_after_malfunction'].iloc[0],
        costs_delta_after_malfunction=rows['costs_delta_after_malfunction'].iloc[0],
        experiment_freeze_full=rows['experiment_freeze_full'].iloc[0],
        experiment_freeze_full_after_malfunction=rows['experiment_freeze_full_after_malfunction'].iloc[0],
        experiment_freeze_delta_after_malfunction=rows['experiment_freeze_delta_after_malfunction'].iloc[0],
        nb_resource_conflicts_full=rows['nb_resource_conflicts_full'].iloc[0],
        nb_resource_conflicts_full_after_malfunction=rows['nb_resource_conflicts_full_after_malfunction'].iloc[0],
        nb_resource_conflicts_delta_after_malfunction=rows['nb_resource_conflicts_delta_after_malfunction'].iloc[0],
        malfunction=rows['malfunction'].iloc[0],
        agents_paths_dict=rows['agents_paths_dict'].iloc[0],
    )


def convert_pandas_series_experiment_results(row: Series) -> ExperimentResults:
    """Converts data frame back to experiment results structure.

    Parameters
    ----------
    rows: DataFrame

    Returns
    -------
    ExperimentResults
    """
    return ExperimentResults(
        time_full=row['time_full'],
        time_full_after_malfunction=row['time_full_after_malfunction'],
        time_delta_after_malfunction=row['time_delta_after_malfunction'],
        solution_full=row['solution_full'],
        solution_full_after_malfunction=row['solution_full_after_malfunction'],
        solution_delta_after_malfunction=row['solution_delta_after_malfunction'],
        costs_full=row['costs_full'],
        costs_full_after_malfunction=row['costs_full_after_malfunction'],
        costs_delta_after_malfunction=row['costs_delta_after_malfunction'],
        experiment_freeze_full=row['experiment_freeze_full'],
        experiment_freeze_full_after_malfunction=row['experiment_freeze_full_after_malfunction'],
        experiment_freeze_delta_after_malfunction=row['experiment_freeze_delta_after_malfunction'],
        nb_resource_conflicts_full=row['nb_resource_conflicts_full'],
        nb_resource_conflicts_full_after_malfunction=row['nb_resource_conflicts_full_after_malfunction'],
        nb_resource_conflicts_delta_after_malfunction=row['nb_resource_conflicts_delta_after_malfunction'],
        malfunction=row['malfunction'],
        agents_paths_dict=row['agents_paths_dict'],
    )


_pp = pprint.PrettyPrinter(indent=4)


def experimentFreezeDictPrettyPrint(d: ExperimentFreezeDict):
    for agent_id, experiment_freeze in d.items():
        prefix = f"agent {agent_id} "
        experimentFreezePrettyPrint(experiment_freeze, prefix)


def experimentFreezePrettyPrint(experiment_freeze: ExperimentFreeze, prefix: str = ""):
    print(f"{prefix}freeze_visit={_pp.pformat(experiment_freeze.freeze_visit)}")
    print(f"{prefix}freeze_earliest={_pp.pformat(experiment_freeze.freeze_earliest)}")
    print(f"{prefix}freeze_latest={_pp.pformat(experiment_freeze.freeze_latest)}")
    print(f"{prefix}freeze_banned={_pp.pformat(experiment_freeze.freeze_banned)}")
