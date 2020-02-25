"""Data types used in the experiment for the real time rescheduling research
project."""
import pprint
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Tuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from pandas import DataFrame
from pandas import Series

from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.route_dag.route_dag import ScheduleProblemDescription

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
                                   ('number_of_shortest_paths_per_agent', int),
                                   ('weight_route_change', int),
                                   ('weight_lateness_seconds', int),
                                   ])

ExperimentAgenda = NamedTuple('ExperimentAgenda', [('experiment_name', str),
                                                   ('experiments', List[ExperimentParameters])])


ExperimentMalfunction = NamedTuple('ExperimentMalfunction', [
    ('time_step', int),
    ('agent_id', int),
    ('malfunction_duration', int)
])

FIELDS_EXPERIMENT_RESULTS = [
    ('time_full', float),
    ('time_full_after_malfunction', float),
    ('time_delta_after_malfunction', float),
    ('solution_full', TrainrunDict),
    ('solution_full_after_malfunction', TrainrunDict),
    ('solution_delta_after_malfunction', TrainrunDict),
    ('costs_full', float),  # sum of travelling times in scheduling solution
    ('costs_full_after_malfunction', float),
    # TODO SIM-325 total delay at target over all agents with respect to schedule
    ('costs_delta_after_malfunction', float),
    # TODO SIM-325 total delay at target over all agents with respect to schedule
    ('problem_full', ScheduleProblemDescription),
    ('problem_full_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_after_malfunction', ScheduleProblemDescription),
    ('malfunction', ExperimentMalfunction),
    ('nb_resource_conflicts_full', int),
    ('nb_resource_conflicts_full_after_malfunction', int),
    ('nb_resource_conflicts_delta_after_malfunction', int)
]

ExperimentResults = NamedTuple('ExperimentResults', FIELDS_EXPERIMENT_RESULTS)

ExperimentResultsAnalysis = NamedTuple('ExperimentResultsAnalyis', FIELDS_EXPERIMENT_RESULTS + [
    ('speed_up', float),
    ('factor_resource_conflicts', int),
    ('path_search_space_schedule', int),
    ('path_search_space_rsp_full', int),
    ('path_search_space_rsp_delta', int),
    ('factor_path_search_space', int),
    ('size_used', int),
    ('lateness_full_after_malfunction', int),
    ('sum_route_section_penalties_full_after_malfunction', int),
    ('lateness_delta_after_malfunction', int),
    ('sum_route_section_penalties_delta_after_malfunction', int),
    ('vertex_eff_lateness_full_after_malfunction', Dict[Waypoint, int]),
    ('edge_eff_route_penalties_full_after_malfunction', Dict[Tuple[Waypoint, Waypoint], int]),
    ('vertex_eff_lateness_delta_after_malfunction', Dict[Waypoint, int]),
    ('edge_eff_route_penalties_delta_after_malfunction', Dict[Tuple[Waypoint, Waypoint], int]),
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
           'problem_full',
           'problem_full_after_malfunction',
           'problem_delta_after_malfunction',
           'nb_resource_conflicts_full',
           'nb_resource_conflicts_full_after_malfunction',
           'nb_resource_conflicts_delta_after_malfunction',
           'malfunction',
           'size',
           'n_agents',
           'max_num_cities',
           'max_rail_between_cities',
           'max_rail_in_city']

COLUMNS_ANALYSIS = COLUMNS + [
    'speed_up',
    'factor_resource_conflicts',
    'path_search_space_schedule',
    'path_search_space_rsp_full',
    'path_search_space_rsp_delta',
    'size_used',
    'factor_path_search_space',
    'lateness_full_after_malfunction',
    'sum_route_section_penalties_full_after_malfunction',
    'lateness_delta_after_malfunction',
    'sum_route_section_penalties_delta_after_malfunction',
    'vertex_eff_lateness_full_after_malfunction',
    'edge_eff_route_penalties_full_after_malfunction',
    'vertex_eff_lateness_delta_after_malfunction',
    'edge_eff_route_penalties_delta_after_malfunction',
]


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
            'problem_full': experiment_results.problem_full,
            'problem_full_after_malfunction': experiment_results.problem_full_after_malfunction,
            'problem_delta_after_malfunction': experiment_results.problem_delta_after_malfunction,
            'nb_resource_conflicts_full': experiment_results.nb_resource_conflicts_full,
            'nb_resource_conflicts_full_after_malfunction': experiment_results.nb_resource_conflicts_full_after_malfunction,
            'nb_resource_conflicts_delta_after_malfunction': experiment_results.nb_resource_conflicts_delta_after_malfunction,
            'malfunction': experiment_results.malfunction,
            'size': experiment_parameters.width,
            'n_agents': experiment_parameters.number_of_agents,
            'max_num_cities': experiment_parameters.max_num_cities,
            'max_rail_between_cities': experiment_parameters.max_rail_between_cities,
            'max_rail_in_city': experiment_parameters.max_rail_in_city,
            }


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
        problem_full=rows['problem_full'].iloc[0],
        problem_full_after_malfunction=rows['problem_full_after_malfunction'].iloc[0],
        problem_delta_after_malfunction=rows['problem_delta_after_malfunction'].iloc[0],
        nb_resource_conflicts_full=rows['nb_resource_conflicts_full'].iloc[0],
        nb_resource_conflicts_full_after_malfunction=rows['nb_resource_conflicts_full_after_malfunction'].iloc[0],
        nb_resource_conflicts_delta_after_malfunction=rows['nb_resource_conflicts_delta_after_malfunction'].iloc[0],
        malfunction=rows['malfunction'].iloc[0]
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
        problem_full=row['problem_full'],
        problem_full_after_malfunction=row['problem_full_after_malfunction'],
        problem_delta_after_malfunction=row['problem_delta_after_malfunction'],
        nb_resource_conflicts_full=row['nb_resource_conflicts_full'],
        nb_resource_conflicts_full_after_malfunction=row['nb_resource_conflicts_full_after_malfunction'],
        nb_resource_conflicts_delta_after_malfunction=row['nb_resource_conflicts_delta_after_malfunction'],
        malfunction=row['malfunction'],
    )


def convert_pandas_series_experiment_results_analysis(row: Series) -> ExperimentResultsAnalysis:
    """Converts data frame back to experiment results structure.

    Parameters
    ----------
    rows: DataFrame

    Returns
    -------
    ExperimentResults
    """
    return ExperimentResultsAnalysis(
        time_full=row['time_full'],
        time_full_after_malfunction=row['time_full_after_malfunction'],
        time_delta_after_malfunction=row['time_delta_after_malfunction'],
        solution_full=row['solution_full'],
        solution_full_after_malfunction=row['solution_full_after_malfunction'],
        solution_delta_after_malfunction=row['solution_delta_after_malfunction'],
        costs_full=row['costs_full'],
        costs_full_after_malfunction=row['costs_full_after_malfunction'],
        costs_delta_after_malfunction=row['costs_delta_after_malfunction'],
        problem_full=row['problem_full'],
        problem_full_after_malfunction=row['problem_full_after_malfunction'],
        problem_delta_after_malfunction=row['problem_delta_after_malfunction'],
        nb_resource_conflicts_full=row['nb_resource_conflicts_full'],
        nb_resource_conflicts_full_after_malfunction=row['nb_resource_conflicts_full_after_malfunction'],
        nb_resource_conflicts_delta_after_malfunction=row['nb_resource_conflicts_delta_after_malfunction'],
        malfunction=row['malfunction'],
        speed_up=row['speed_up'],
        factor_resource_conflicts=row['factor_resource_conflicts'],
        path_search_space_schedule=row['path_search_space_schedule'],
        path_search_space_rsp_full=row['path_search_space_rsp_full'],
        path_search_space_rsp_delta=row['path_search_space_rsp_delta'],
        size_used=row['size_used'],
        factor_path_search_space=row['factor_path_search_space'],
        lateness_full_after_malfunction=row['lateness_full_after_malfunction'],
        sum_route_section_penalties_full_after_malfunction=row['sum_route_section_penalties_full_after_malfunction'],
        lateness_delta_after_malfunction=row['lateness_delta_after_malfunction'],
        sum_route_section_penalties_delta_after_malfunction=row['sum_route_section_penalties_delta_after_malfunction'],
        vertex_eff_lateness_full_after_malfunction=row['vertex_eff_lateness_full_after_malfunction'],
        edge_eff_route_penalties_full_after_malfunction=row['edge_eff_route_penalties_full_after_malfunction'],
        vertex_eff_lateness_delta_after_malfunction=row['vertex_eff_lateness_delta_after_malfunction'],
        edge_eff_route_penalties_delta_after_malfunction=row['edge_eff_route_penalties_delta_after_malfunction'],

    )


def extend_experiment_results_for_analysis(
        experiment_results: ExperimentResultsAnalysis,
        speed_up: float,
        factor_resource_conflicts: int,
        path_search_space_schedule: int,
        path_search_space_rsp_full: int,
        path_search_space_rsp_delta: int,
        size_used: int,
        factor_path_search_space: int,
        lateness_full_after_malfunction: int,
        sum_route_section_penalties_full_after_malfunction: int,
        lateness_delta_after_malfunction: int,
        sum_route_section_penalties_delta_after_malfunction: int,
        vertex_eff_lateness_full_after_malfunction: Dict[Waypoint, int],
        edge_eff_route_penalties_full_after_malfunction: Dict[Tuple[Waypoint, Waypoint], int],
        vertex_eff_lateness_delta_after_malfunction: Dict[Waypoint, int],
        edge_eff_route_penalties_delta_after_malfunction: Dict[Tuple[Waypoint, Waypoint], int]
) -> ExperimentResultsAnalysis:
    return ExperimentResultsAnalysis(
        *experiment_results,
        speed_up=speed_up,
        factor_resource_conflicts=factor_resource_conflicts,
        path_search_space_schedule=path_search_space_schedule,
        path_search_space_rsp_full=path_search_space_rsp_full,
        path_search_space_rsp_delta=path_search_space_rsp_delta,
        size_used=size_used,
        factor_path_search_space=factor_path_search_space,
        lateness_full_after_malfunction=lateness_full_after_malfunction,
        sum_route_section_penalties_full_after_malfunction=sum_route_section_penalties_full_after_malfunction,
        lateness_delta_after_malfunction=lateness_delta_after_malfunction,
        sum_route_section_penalties_delta_after_malfunction=sum_route_section_penalties_delta_after_malfunction,
        vertex_eff_lateness_full_after_malfunction=vertex_eff_lateness_full_after_malfunction,
        edge_eff_route_penalties_full_after_malfunction=edge_eff_route_penalties_full_after_malfunction,
        vertex_eff_lateness_delta_after_malfunction=vertex_eff_lateness_delta_after_malfunction,
        edge_eff_route_penalties_delta_after_malfunction=edge_eff_route_penalties_delta_after_malfunction,
    )


def convert_experiment_results_analysis_to_data_frame(experiment_results: ExperimentResultsAnalysis,
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
            'problem_full': experiment_results.problem_full,
            'problem_full_after_malfunction': experiment_results.problem_full_after_malfunction,
            'problem_delta_after_malfunction': experiment_results.problem_delta_after_malfunction,
            'nb_resource_conflicts_full': experiment_results.nb_resource_conflicts_full,
            'nb_resource_conflicts_full_after_malfunction': experiment_results.nb_resource_conflicts_full_after_malfunction,
            'nb_resource_conflicts_delta_after_malfunction': experiment_results.nb_resource_conflicts_delta_after_malfunction,
            'malfunction': experiment_results.malfunction,
            'size': experiment_parameters.width,
            'n_agents': experiment_parameters.number_of_agents,
            'max_num_cities': experiment_parameters.max_num_cities,
            'max_rail_between_cities': experiment_parameters.max_rail_between_cities,
            'max_rail_in_city': experiment_parameters.max_rail_in_city,
            'speed_up': experiment_results.speed_up,
            'factor_resource_conflicts': experiment_results.factor_resource_conflicts,
            'path_search_space_schedule': experiment_results.path_search_space_schedule,
            'path_search_space_rsp_full': experiment_results.path_search_space_rsp_full,
            'path_search_space_rsp_delta': experiment_results.path_search_space_rsp_delta,
            'factor_path_search_space': experiment_results.factor_path_search_space,
            'lateness_full_after_malfunction': experiment_results.lateness_full_after_malfunction,
            'sum_route_section_penalties_full_after_malfunction': experiment_results.sum_route_section_penalties_full_after_malfunction,
            'lateness_delta_after_malfunction': experiment_results.lateness_delta_after_malfunction,
            'sum_route_section_penalties_delta_after_malfunction': experiment_results.sum_route_section_penalties_delta_after_malfunction,
            'vertex_eff_lateness_full_after_malfunction': experiment_results.vertex_eff_lateness_full_after_malfunction,
            'edge_eff_route_penalties_full_after_malfunction': experiment_results.edge_eff_route_penalties_full_after_malfunction,
            'vertex_eff_lateness_delta_after_malfunction': experiment_results.vertex_eff_lateness_delta_after_malfunction,
            'edge_eff_route_penalties_delta_after_malfunction': experiment_results.edge_eff_route_penalties_delta_after_malfunction,

            }


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
