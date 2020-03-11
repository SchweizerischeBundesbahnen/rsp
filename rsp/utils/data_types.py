"""Data types used in the experiment for the real time rescheduling research
project."""
import pprint
import warnings
from functools import reduce
from operator import mul
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Set
from typing import Tuple

import pandas as pd
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from pandas import DataFrame
from pandas import Series

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.experiment_solvers.global_switches import COMPATIBILITY_MODE
from rsp.route_dag.route_dag import get_paths_for_route_dag_constraints
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.route_dag.route_dag import TopoDict

SpeedData = Mapping[float, float]

ParameterRanges = NamedTuple('ParameterRanges', [('size_range', List[int]),
                                                 ('agent_range', List[int]),
                                                 ('in_city_rail_range', List[int]),
                                                 ('out_city_rail_range', List[int]),
                                                 ('city_range', List[int]),
                                                 ('earliest_malfunction', List[int]),
                                                 ('malfunction_duration', List[int]),
                                                 ('number_of_shortest_paths_per_agent', List[int]),
                                                 ('max_window_size_from_earliest', List[int])
                                                 ])

# the experiment_id is unambiguous within the agenda for the full parameter set!
ExperimentParameters = NamedTuple('ExperimentParameters',
                                  [('experiment_id', int),
                                   ('grid_id', int),
                                   ('number_of_agents', int),
                                   ('speed_data', SpeedData),
                                   ('width', int),
                                   ('height', int),
                                   ('flatland_seed_value', int),
                                   ('asp_seed_value', int),
                                   ('max_num_cities', int),
                                   ('grid_mode', bool),
                                   ('max_rail_between_cities', int),
                                   ('max_rail_in_city', int),
                                   ('earliest_malfunction', int),
                                   ('malfunction_duration', int),
                                   ('number_of_shortest_paths_per_agent', int),
                                   ('weight_route_change', int),
                                   ('weight_lateness_seconds', int),
                                   ('max_window_size_from_earliest', int),
                                   ]
                                  )
if COMPATIBILITY_MODE:
    ExperimentParameters.__new__.__defaults__ = (None,) * len(ExperimentParameters._fields)

ExperimentAgenda = NamedTuple('ExperimentAgenda', [('experiment_name', str),
                                                   ('experiments', List[ExperimentParameters])])

ExperimentResults = NamedTuple('ExperimentResults', [
    ('experiment_parameters', ExperimentParameters),
    ('malfunction', ExperimentMalfunction),
    ('problem_full', ScheduleProblemDescription),
    ('problem_full_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_after_malfunction', ScheduleProblemDescription),
    ('results_full', SchedulingExperimentResult),
    ('results_full_after_malfunction', SchedulingExperimentResult),
    ('results_delta_after_malfunction', SchedulingExperimentResult),
])
if COMPATIBILITY_MODE:
    ExperimentResults.__new__.__defaults__ = (None,) * len(ExperimentResults._fields)

ExperimentResultsAnalysis = NamedTuple('ExperimentResultsAnalysis', [
    ('experiment_parameters', ExperimentParameters),
    ('malfunction', ExperimentMalfunction),
    ('problem_full', ScheduleProblemDescription),
    ('problem_full_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_after_malfunction', ScheduleProblemDescription),
    ('results_full', SchedulingExperimentResult),
    ('results_full_after_malfunction', SchedulingExperimentResult),
    ('results_delta_after_malfunction', SchedulingExperimentResult),
    ('experiment_id', int),
    ('grid_id', int),
    ('size', int),
    ('n_agents', int),
    ('max_num_cities', int),
    ('max_rail_between_cities', int),
    ('max_rail_in_city', int),
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
    ('nb_resource_conflicts_full', int),
    ('nb_resource_conflicts_full_after_malfunction', int),
    ('nb_resource_conflicts_delta_after_malfunction', int),
    ('speed_up', float),
    ('factor_resource_conflicts', int),
    ('path_search_space_schedule', int),
    ('path_search_space_rsp_full', int),
    ('path_search_space_rsp_delta', int),
    ('factor_path_search_space', int),
    ('size_used', int),
    ('lateness_full_after_malfunction', Dict[int, int]),
    ('sum_route_section_penalties_full_after_malfunction', int),
    ('lateness_delta_after_malfunction', Dict[int, int]),
    ('sum_route_section_penalties_delta_after_malfunction', int),
    ('vertex_eff_lateness_full_after_malfunction', Dict[Waypoint, int]),
    ('edge_eff_route_penalties_full_after_malfunction', Dict[Tuple[Waypoint, Waypoint], int]),
    ('vertex_eff_lateness_delta_after_malfunction', Dict[Waypoint, int]),
    ('edge_eff_route_penalties_delta_after_malfunction', Dict[Tuple[Waypoint, Waypoint], int]),
])
if COMPATIBILITY_MODE:
    ExperimentResults.__new__.__defaults__ = (None,) * len(ExperimentResultsAnalysis._fields)
COLUMNS = ExperimentResults._fields
COLUMNS_ANALYSIS = ExperimentResultsAnalysis._fields

TrainSchedule = Dict[int, Waypoint]
TrainScheduleDict = Dict[int, TrainSchedule]


def convert_experiment_results_to_data_frame(experiment_results: ExperimentResults,
                                             experiment_parameters: ExperimentParameters) -> Dict:
    """Converts experiment results to data frame.

    Parameters
    ----------
    experiment_results: ExperimentResults
    experiment_parameters: ExperimentParameters

    Returns
    -------
    DataFrame
    """
    return experiment_results._asdict()


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
        experiment_parameters=rows['experiment_parameters'].iloc[0],
        malfunction=rows['malfunction'].iloc[0],
        problem_full=rows['problem_full'].iloc[0],
        problem_full_after_malfunction=rows['problem_full_after_malfunction'].iloc[0],
        problem_delta_after_malfunction=rows['problem_delta_after_malfunction'].iloc[0],
        results_full=rows['results_full'].iloc[0],
        results_full_after_malfunction=rows['results_full_after_malfunction'].iloc[0],
        results_delta_after_malfunction=rows['results_delta_after_malfunction'].iloc[0],
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
    return ExperimentResults(**row)


def convert_pandas_series_experiment_results_analysis(row: Series) -> ExperimentResultsAnalysis:
    """Converts data frame back to experiment results structure.

    Parameters
    ----------
    rows: DataFrame

    Returns
    -------
    ExperimentResults
    """
    return ExperimentResultsAnalysis(**row)


def convert_list_of_experiment_results_analysis_to_data_frame(l: List[ExperimentResultsAnalysis]) -> DataFrame:
    return pd.DataFrame(columns=COLUMNS_ANALYSIS, data=[r._asdict() for r in l])


def convert_list_of_experiment_results_to_data_frame(l: List[ExperimentResults]) -> DataFrame:
    return pd.DataFrame(columns=COLUMNS, data=[r._asdict() for r in l])


def expand_experiment_results_list_for_analysis(l: List[ExperimentResults]) -> List[ExperimentResultsAnalysis]:
    return list(map(expand_experiment_results_for_analysis, l))


def expand_experiment_results_for_analysis(
        experiment_results: ExperimentResults,
        debug: bool = False
) -> ExperimentResultsAnalysis:
    if not isinstance(experiment_results, ExperimentResults):
        experiment_results_as_dict = dict(experiment_results[0])
        experiment_results = ExperimentResults(**experiment_results_as_dict)
    experiment_id = experiment_results.experiment_parameters.experiment_id

    # derive speed up
    time_full = experiment_results.results_full.solve_time
    time_full_after_malfunction = experiment_results.results_full_after_malfunction.solve_time
    time_delta_after_malfunction = experiment_results.results_delta_after_malfunction.solve_time
    nb_resource_conflicts_delta_after_malfunction = experiment_results.results_delta_after_malfunction.nb_conflicts
    nb_resource_conflicts_full_after_malfunction = experiment_results.results_full_after_malfunction.nb_conflicts
    speed_up = time_full_after_malfunction / time_delta_after_malfunction
    # search space indiciators
    factor_resource_conflicts = -1
    try:
        factor_resource_conflicts = \
            nb_resource_conflicts_delta_after_malfunction / \
            nb_resource_conflicts_full_after_malfunction
    except ZeroDivisionError:
        warnings.warn(f"no resource conflicts for experiment {experiment_id} -> set ratio to -1:\n{experiment_results}")
    path_search_space_rsp_delta, path_search_space_rsp_full, path_search_space_schedule = extract_path_search_space(
        experiment_results=experiment_results)
    factor_path_search_space = path_search_space_rsp_delta / path_search_space_rsp_full
    used_cells: Set[Waypoint] = {
        waypoint for agent_id, topo in
        experiment_results.problem_full.topo_dict.items()
        for waypoint in topo.nodes
    }
    # costs: lateness and routing penalties
    lateness_full_after_malfunction = {}
    sum_route_section_penalties_full_after_malfunction = {}
    lateness_delta_after_malfunction = {}
    sum_route_section_penalties_delta_after_malfunction = {}
    vertex_eff_lateness_full_after_malfunction = {}
    edge_eff_route_penalties_full_after_malfunction = {}
    vertex_eff_lateness_delta_after_malfunction = {}
    edge_eff_route_penalties_delta_after_malfunction = {}
    for agent_id in experiment_results.results_full.trainruns_dict.keys():
        # full re-scheduling
        train_run_full_after_malfunction_agent = experiment_results.results_full_after_malfunction.trainruns_dict[
            agent_id]
        dummy_target_vertex = Waypoint(
            position=train_run_full_after_malfunction_agent[-1].waypoint.position,
            direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)

        train_run_full_after_malfunction_constraints_agent = \
            experiment_results.problem_full_after_malfunction.route_dag_constraints_dict[agent_id]
        train_run_full_after_malfunction_dummy_target_earliest_agent = \
            train_run_full_after_malfunction_constraints_agent.freeze_earliest[dummy_target_vertex]
        train_run_full_after_malfunction_scheduled_at_dummy_target = \
            train_run_full_after_malfunction_agent[-1].scheduled_at + 1
        lateness_full_after_malfunction[agent_id] = \
            max(
                train_run_full_after_malfunction_scheduled_at_dummy_target -
                train_run_full_after_malfunction_dummy_target_earliest_agent,
                0)
        # TODO SIM-325 extend to all vertices
        vertex_eff_lateness_full_after_malfunction[agent_id] = {
            dummy_target_vertex: lateness_full_after_malfunction[agent_id]
        }
        edges_full_after_malfunction_agent = {
            (wp1.waypoint, wp2.waypoint)
            for wp1, wp2 in zip(train_run_full_after_malfunction_agent, train_run_full_after_malfunction_agent[1:])
        }
        edge_eff_route_penalties_full_after_malfunction_agent = {
            edge: penalty
            for edge, penalty in
            experiment_results.problem_full_after_malfunction.route_section_penalties[agent_id].items()
            if edge in edges_full_after_malfunction_agent
        }
        edge_eff_route_penalties_full_after_malfunction[
            agent_id] = edge_eff_route_penalties_full_after_malfunction_agent
        sum_route_section_penalties_full_after_malfunction[agent_id] = sum(
            edge_eff_route_penalties_full_after_malfunction_agent.values())

        # delta re-scheduling
        train_run_delta_after_malfunction_agent = experiment_results.results_delta_after_malfunction.trainruns_dict[
            agent_id]
        train_run_delta_after_malfunction_target_agent = train_run_delta_after_malfunction_agent[-1]
        train_run_delta_after_malfunction_constraints_agent = \
            experiment_results.problem_delta_after_malfunction.route_dag_constraints_dict[agent_id]
        train_run_delta_after_malfunction_dummy_target_earliest_agent = \
            train_run_delta_after_malfunction_constraints_agent.freeze_earliest[dummy_target_vertex]
        train_run_delta_after_malfunction_scheduled_at_dummy_target = \
            train_run_delta_after_malfunction_target_agent.scheduled_at + 1
        lateness_delta_after_malfunction[agent_id] = \
            max(
                train_run_delta_after_malfunction_scheduled_at_dummy_target - train_run_delta_after_malfunction_dummy_target_earliest_agent,
                0)
        # TODO SIM-325 extend to all vertices
        vertex_eff_lateness_delta_after_malfunction[agent_id] = {
            dummy_target_vertex: lateness_delta_after_malfunction[agent_id]
        }
        edges_delta_after_malfunction_agent = {
            (wp1.waypoint, wp2.waypoint)
            for wp1, wp2 in
            zip(train_run_delta_after_malfunction_agent, train_run_delta_after_malfunction_agent[1:])
        }
        edge_eff_route_penalties_delta_after_malfunction_agent = {
            edge: penalty
            for edge, penalty in
            experiment_results.problem_delta_after_malfunction.route_section_penalties[agent_id].items()
            if edge in edges_delta_after_malfunction_agent
        }
        edge_eff_route_penalties_delta_after_malfunction[
            agent_id] = edge_eff_route_penalties_delta_after_malfunction_agent
        sum_route_section_penalties_delta_after_malfunction[agent_id] = sum(
            edge_eff_route_penalties_delta_after_malfunction_agent.values())
    if debug:
        print(f"[{experiment_id}] lateness_full_after_malfunction={lateness_full_after_malfunction}")
        print(
            f"[{experiment_id}] sum_route_section_penalties_full_after_malfunction={sum_route_section_penalties_full_after_malfunction}")
        print(f"[{experiment_id}] lateness_delta_after_malfunction={lateness_delta_after_malfunction}")
        print(
            f"[{experiment_id}] sum_route_section_penalties_delta_after_malfunction={sum_route_section_penalties_delta_after_malfunction}")

    return ExperimentResultsAnalysis(
        **experiment_results._asdict(),
        experiment_id=experiment_results.experiment_parameters.experiment_id,
        grid_id=experiment_results.experiment_parameters.grid_id,
        size=experiment_results.experiment_parameters.width,
        n_agents=experiment_results.experiment_parameters.number_of_agents,
        max_num_cities=experiment_results.experiment_parameters.max_num_cities,
        max_rail_between_cities=experiment_results.experiment_parameters.max_rail_between_cities,
        max_rail_in_city=experiment_results.experiment_parameters.max_rail_in_city,
        time_full=time_full,
        time_full_after_malfunction=time_full_after_malfunction,
        time_delta_after_malfunction=time_delta_after_malfunction,
        solution_full=experiment_results.results_full.trainruns_dict,
        solution_full_after_malfunction=experiment_results.results_full_after_malfunction.trainruns_dict,
        solution_delta_after_malfunction=experiment_results.results_delta_after_malfunction.trainruns_dict,
        costs_full=experiment_results.results_full.optimization_costs,
        costs_full_after_malfunction=experiment_results.results_full_after_malfunction.optimization_costs,
        costs_delta_after_malfunction=experiment_results.results_delta_after_malfunction.optimization_costs,
        nb_resource_conflicts_full=experiment_results.results_full.nb_conflicts,
        nb_resource_conflicts_full_after_malfunction=experiment_results.results_full_after_malfunction.nb_conflicts,
        nb_resource_conflicts_delta_after_malfunction=experiment_results.results_delta_after_malfunction.nb_conflicts,
        speed_up=speed_up,
        factor_resource_conflicts=factor_resource_conflicts,
        path_search_space_schedule=path_search_space_schedule,
        path_search_space_rsp_full=path_search_space_rsp_full,
        path_search_space_rsp_delta=path_search_space_rsp_delta,
        size_used=len(used_cells),
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


def convert_experiment_results_analysis_to_data_frame(experiment_results: ExperimentResultsAnalysis) -> DataFrame:
    """Converts experiment results to data frame.

    Parameters
    ----------
    experiment_results: ExperimentResults
    experiment_parameters: ExperimentParameters

    Returns
    -------
    DataFrame
    """
    return experiment_results._asdict()


def extract_path_search_space(experiment_results: ExperimentResults) -> Tuple[int, int, int]:
    route_dag_constraints_delta_afer_malfunction = experiment_results.problem_delta_after_malfunction.route_dag_constraints_dict
    route_dag_constraints_full_after_malfunction = experiment_results.problem_delta_after_malfunction.route_dag_constraints_dict
    route_dag_constraints_schedule = experiment_results.problem_full.route_dag_constraints_dict
    topo_dict = experiment_results.problem_full.topo_dict
    all_nb_alternatives_rsp_delta, all_nb_alternatives_rsp_full, all_nb_alternatives_schedule = extract_number_of_path_alternatives(
        topo_dict,
        route_dag_constraints_schedule,
        route_dag_constraints_delta_afer_malfunction,
        route_dag_constraints_full_after_malfunction)
    path_search_space_schedule = _prod(all_nb_alternatives_schedule)
    path_search_space_rsp_full = _prod(all_nb_alternatives_rsp_full)
    path_search_space_rsp_delta = _prod(all_nb_alternatives_rsp_delta)
    return path_search_space_rsp_delta, path_search_space_rsp_full, path_search_space_schedule


def extract_number_of_path_alternatives(
        topo_dict: TopoDict,
        route_dag_constraints_schedule: RouteDAGConstraintsDict,
        route_dag_constraints_delta_afer_malfunction: RouteDAGConstraintsDict,
        route_dag_constraints_full_after_malfunction: RouteDAGConstraintsDict
) -> Tuple[List[int], List[int], List[int]]:
    """Extract number of path alternatives for schedule, rsp full and rsp delta
    for each agent."""
    all_nb_alternatives_schedule = []
    all_nb_alternatives_rsp_full = []
    all_nb_alternatives_rsp_delta = []

    for agent_id in route_dag_constraints_delta_afer_malfunction:
        alternatives_schedule = get_paths_for_route_dag_constraints(
            topo=topo_dict[agent_id],
            route_dag_constraints=route_dag_constraints_schedule[agent_id]
        )
        alternatives_rsp_full = get_paths_for_route_dag_constraints(
            topo=topo_dict[agent_id],
            route_dag_constraints=route_dag_constraints_full_after_malfunction[agent_id]
        )
        alternatives_rsp_delta = get_paths_for_route_dag_constraints(
            topo=topo_dict[agent_id],
            route_dag_constraints=route_dag_constraints_delta_afer_malfunction[agent_id]
        )
        all_nb_alternatives_schedule.append(len(alternatives_schedule))
        all_nb_alternatives_rsp_full.append(len(alternatives_rsp_full))
        all_nb_alternatives_rsp_delta.append(len(alternatives_rsp_delta))
    return all_nb_alternatives_rsp_delta, all_nb_alternatives_rsp_full, all_nb_alternatives_schedule


# numpy produces overflow -> python ints are unbounded,
# see https://stackoverflow.com/questions/2104782/returning-the-product-of-a-list
def _prod(l: List[int]):
    return reduce(mul, l, 1)


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
