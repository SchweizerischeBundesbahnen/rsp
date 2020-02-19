"""Data types used in the experiment for the real time rescheduling research
project."""
import pprint
from typing import List
from typing import Mapping
from typing import NamedTuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame
from pandas import Series

from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.route_dag.route_dag import TopoDict

SpeedData = Mapping[float, float]
# experiment_group (future use): if we want use a range of values on the same infrastructure and want to identify them
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
    ('route_dag_constraints_full', RouteDAGConstraintsDict),
    ('route_dag_constraints_full_after_malfunction', RouteDAGConstraintsDict),
    ('route_dag_constraints_delta_after_malfunction', RouteDAGConstraintsDict),
    ('malfunction', ExperimentMalfunction),
    ('topo_dict', TopoDict),
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
           'route_dag_constraints_full',
           'route_dag_constraints_full_after_malfunction',
           'route_dag_constraints_delta_after_malfunction',
           'nb_resource_conflicts_full',
           'nb_resource_conflicts_full_after_malfunction',
           'nb_resource_conflicts_delta_after_malfunction',
           'malfunction',
           'topo_dict',
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
            'time_full_after_malfunction': experiment_results.time_full_after_malfunction,
            'time_delta_after_malfunction': experiment_results.time_delta_after_malfunction,
            'solution_full': experiment_results.solution_full,
            'solution_full_after_malfunction': experiment_results.solution_full_after_malfunction,
            'solution_delta_after_malfunction': experiment_results.solution_delta_after_malfunction,
            'costs_full': experiment_results.costs_full,
            'costs_full_after_malfunction': experiment_results.costs_full_after_malfunction,
            'costs_delta_after_malfunction': experiment_results.costs_delta_after_malfunction,
            'route_dag_constraints_full': experiment_results.route_dag_constraints_full,
            'route_dag_constraints_full_after_malfunction': experiment_results.route_dag_constraints_full_after_malfunction,
            'route_dag_constraints_delta_after_malfunction': experiment_results.route_dag_constraints_delta_after_malfunction,
            'nb_resource_conflicts_full': experiment_results.nb_resource_conflicts_full,
            'nb_resource_conflicts_full_after_malfunction': experiment_results.nb_resource_conflicts_full_after_malfunction,
            'nb_resource_conflicts_delta_after_malfunction': experiment_results.nb_resource_conflicts_delta_after_malfunction,
            'malfunction': experiment_results.malfunction,
            'topo_dict': experiment_results.topo_dict,
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
        route_dag_constraints_full=rows['route_dag_constraints_full'].iloc[0],
        route_dag_constraints_full_after_malfunction=rows['route_dag_constraints_full_after_malfunction'].iloc[0],
        route_dag_constraints_delta_after_malfunction=rows['route_dag_constraints_delta_after_malfunction'].iloc[0],
        nb_resource_conflicts_full=rows['nb_resource_conflicts_full'].iloc[0],
        nb_resource_conflicts_full_after_malfunction=rows['nb_resource_conflicts_full_after_malfunction'].iloc[0],
        nb_resource_conflicts_delta_after_malfunction=rows['nb_resource_conflicts_delta_after_malfunction'].iloc[0],
        malfunction=rows['malfunction'].iloc[0],
        topo_dict=rows['topo_dict'].iloc[0],
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
        route_dag_constraints_full=row['route_dag_constraints_full'],
        route_dag_constraints_full_after_malfunction=row['route_dag_constraints_full_after_malfunction'],
        route_dag_constraints_delta_after_malfunction=row['route_dag_constraints_delta_after_malfunction'],
        nb_resource_conflicts_full=row['nb_resource_conflicts_full'],
        nb_resource_conflicts_full_after_malfunction=row['nb_resource_conflicts_full_after_malfunction'],
        nb_resource_conflicts_delta_after_malfunction=row['nb_resource_conflicts_delta_after_malfunction'],
        malfunction=row['malfunction'],
        topo_dict=row['topo_dict'],
    )


_pp = pprint.PrettyPrinter(indent=4)


def experimentFreezeDictPrettyPrint(d: RouteDAGConstraintsDict):
    for agent_id, route_dag_constraints in d.items():
        prefix = f"agent {agent_id} "
        experimentFreezePrettyPrint(route_dag_constraints, prefix)


def experimentFreezePrettyPrint(route_dag_constraints: RouteDAGConstraints, prefix: str = ""):
    print(f"{prefix}freeze_visit={_pp.pformat(route_dag_constraints.freeze_visit)}")
    print(f"{prefix}freeze_earliest={_pp.pformat(route_dag_constraints.freeze_earliest)}")
    print(f"{prefix}freeze_latest={_pp.pformat(route_dag_constraints.freeze_latest)}")
    print(f"{prefix}freeze_banned={_pp.pformat(route_dag_constraints.freeze_banned)}")
