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
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from pandas import DataFrame
from pandas import Series

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.schedule_problem_description.data_types_and_utils import get_paths_for_route_dag_constraints
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraintsDict
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import TopoDict
from rsp.utils.general_helpers import catch_zero_division_error_as_minus_one

SpeedData = Mapping[float, float]

SymmetricEncounterGraphDistance = NamedTuple('SymmetricEncounterGraphDistance', [
    ('proximity', float)
])

ParameterRanges = NamedTuple('ParameterRanges', [
    # infrastructure and agent placement
    # 0: size_range
    ('size_range', List[int]),
    # 1: agent_range
    ('agent_range', List[int]),
    # 2: in_city_rail_range
    ('in_city_rail_range', List[int]),
    # 3: out_city_rail_range
    ('out_city_rail_range', List[int]),
    # 4: city_range
    ('city_range', List[int]),


    # schedule

    # malfunction
    # 5: earliest_malfunction
    ('earliest_malfunction', List[int]),
    # 6: malfunction_duration
    ('malfunction_duration', List[int]),

    # rescheduling
    # 7: number_of_shortest_paths_per_agent
    ('number_of_shortest_paths_per_agent', List[int]),
    # 8: max_window_size_from_earliest
    ('max_window_size_from_earliest', List[int]),
    # 9: asp_seed_value
    ('asp_seed_value', List[int]),
    # 10: weight_route_change
    ('weight_route_change', List[int]),
    # 11: weight_lateness_seconds
    ('weight_lateness_seconds', List[int])
])
ParameterRangesAndSpeedData = NamedTuple('ParameterRangesAndSpeedData', [
    ('parameter_ranges', ParameterRanges),
    ('speed_data', SpeedData)
])

InfrastructureParameters = NamedTuple('InfrastructureParameters', [
    ('infra_id', int),

    ('width', int),
    ('height', int),
    ('flatland_seed_value', int),
    ('max_num_cities', int),
    ('grid_mode', bool),
    ('max_rail_between_cities', int),
    ('max_rail_in_city', int),
    ('number_of_agents', int),
    ('speed_data', SpeedData),
    ('number_of_shortest_paths_per_agent', int),
])

InfrastructureParametersRange = NamedTuple('InfrastructureParameters', [
    ('width', List[int]),
    ('height', List[int]),
    ('flatland_seed_value', List[int]),
    ('max_num_cities', List[int]),
    ('max_rail_between_cities', List[int]),
    ('max_rail_in_city', List[int]),
    ('number_of_agents', List[int]),
    ('number_of_shortest_paths_per_agent', List[int]),
    # TODO SIM-650: flatland_seed in range?!
])

ScheduleParameters = NamedTuple('ScheduleParameters', [
    ('infra_id', int),
    ('schedule_id', int),

    ('asp_seed_value', int),
    # TODO SIM-650 use this value
    ('number_of_shortest_paths_per_agent_schedule', int),
])

ScheduleParametersRange = NamedTuple('ScheduleParametersRange', [
    ('asp_seed_value', List[int]),
    # TODO SIM-650 use this value
    ('number_of_shortest_paths_per_agent_schedule', List[int]),
])

ReScheduleParametersRange = NamedTuple('ReScheduleParametersRange', [
    # 5: earliest_malfunction
    ('earliest_malfunction', List[int]),
    # 6: malfunction_duration
    ('malfunction_duration', List[int]),

    # rescheduling
    # 7: number_of_shortest_paths_per_agent
    ('number_of_shortest_paths_per_agent', List[int]),
    # 8: max_window_size_from_earliest
    ('max_window_size_from_earliest', List[int]),
    # 9: asp_seed_value
    ('asp_seed_value', List[int]),
    # 10: weight_route_change
    ('weight_route_change', List[int]),
    # 11: weight_lateness_seconds
    ('weight_lateness_seconds', List[int])
])

ReScheduleParameters = NamedTuple('ReScheduleParameters', [
    # 5: earliest_malfunction
    ('earliest_malfunction', int),
    # 6: malfunction_duration
    ('malfunction_duration', int),

    # rescheduling
    # 7: number_of_shortest_paths_per_agent
    ('number_of_shortest_paths_per_agent', int),
    # 8: max_window_size_from_earliest
    ('max_window_size_from_earliest', int),
    # 9: asp_seed_value
    ('asp_seed_value', int),
    # 10: weight_route_change
    ('weight_route_change', int),
    # 11: weight_lateness_seconds
    ('weight_lateness_seconds', int)
])

# the experiment_id is unambiguous within the agenda for the full parameter set!
ExperimentParameters = NamedTuple('ExperimentParameters', [
    ('experiment_id', int),  # unique per execution (there may be multiple `experiment_id`s for the same `grid_id`
    ('grid_id', int),  # same if all params are the same

    ('infra_parameters', InfrastructureParameters),
    ('schedule_parameters', ScheduleParameters),

    ('earliest_malfunction', int),
    ('malfunction_duration', int),
    ('weight_route_change', int),
    ('weight_lateness_seconds', int),
    ('max_window_size_from_earliest', int)
])

ExperimentAgenda = NamedTuple('ExperimentAgenda', [
    ('experiment_name', str),
    ('experiments', List[ExperimentParameters])
])


def parameter_ranges_and_speed_data_to_hiearchical(
        parameter_ranges_and_speed_data: ParameterRangesAndSpeedData,
        flatland_seed_value: int = 12
) -> Tuple[InfrastructureParametersRange, SpeedData, ScheduleParametersRange, ReScheduleParametersRange]:
    return InfrastructureParametersRange(
        width=parameter_ranges_and_speed_data.parameter_ranges.size_range,
        height=parameter_ranges_and_speed_data.parameter_ranges.size_range,
        flatland_seed_value=[flatland_seed_value, flatland_seed_value, 1],
        max_num_cities=parameter_ranges_and_speed_data.parameter_ranges.city_range,
        max_rail_in_city=parameter_ranges_and_speed_data.parameter_ranges.in_city_rail_range,
        max_rail_between_cities=parameter_ranges_and_speed_data.parameter_ranges.out_city_rail_range,
        number_of_agents=parameter_ranges_and_speed_data.parameter_ranges.agent_range,
        number_of_shortest_paths_per_agent=parameter_ranges_and_speed_data.parameter_ranges.number_of_shortest_paths_per_agent,
    ), parameter_ranges_and_speed_data.speed_data, \
           ScheduleParametersRange(
               asp_seed_value=parameter_ranges_and_speed_data.parameter_ranges.asp_seed_value,
               # TODO SIM-622 hard-code to 1/evaluate
               number_of_shortest_paths_per_agent_schedule=parameter_ranges_and_speed_data.parameter_ranges.number_of_shortest_paths_per_agent,
           ), ReScheduleParametersRange(
        earliest_malfunction=parameter_ranges_and_speed_data.parameter_ranges.earliest_malfunction,
        malfunction_duration=parameter_ranges_and_speed_data.parameter_ranges.malfunction_duration,
        number_of_shortest_paths_per_agent=parameter_ranges_and_speed_data.parameter_ranges.number_of_shortest_paths_per_agent,
        max_window_size_from_earliest=parameter_ranges_and_speed_data.parameter_ranges.max_window_size_from_earliest,
        asp_seed_value=parameter_ranges_and_speed_data.parameter_ranges.asp_seed_value,
        weight_route_change=parameter_ranges_and_speed_data.parameter_ranges.weight_route_change,
        weight_lateness_seconds=parameter_ranges_and_speed_data.parameter_ranges.weight_lateness_seconds,
    )


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
    ('solve_total_ratio_full', float),
    ('solve_time_full', float),
    ('total_time_full', float),
    ('choice_conflict_ratio_full', float),
    ('choices_full', float),
    ('conflicts_full', float),
    ('user_accu_propagations_full', float),
    ('user_step_propagations_full', float),
    ('solve_total_ratio_full_after_malfunction', float),
    ('solve_time_full_after_malfunction', float),
    ('total_time_full_after_malfunction', float),
    ('choice_conflict_ratio_full_after_malfunction', float),
    ('choices_full_after_malfunction', float),
    ('conflicts_full_after_malfunction', float),
    ('user_accu_propagations_full_after_malfunction', float),
    ('user_step_propagations_full_after_malfunction', float),
    ('solve_total_ratio_delta_after_malfunction', float),
    ('solve_time_delta_after_malfunction', float),
    ('total_time_delta_after_malfunction', float),
    ('choice_conflict_ratio_delta_after_malfunction', float),
    ('choices_delta_after_malfunction', float),
    ('conflicts_delta_after_malfunction', float),
    ('user_accu_propagations_delta_after_malfunction', float),
    ('user_step_propagations_delta_after_malfunction', float),
])

COLUMNS = ExperimentResults._fields
COLUMNS_ANALYSIS = ExperimentResultsAnalysis._fields

LeftClosedInterval = NamedTuple('LeftClosedInterval', [
    ('from_incl', int),
    ('to_excl', int)])
Resource = NamedTuple('Resource', [
    ('row', int),
    ('column', int)])
ResourceOccupation = NamedTuple('ResourceOccupation', [
    ('interval', LeftClosedInterval),
    ('resource', Resource),
    ('direction', int),
    ('agent_id', int)
])

# sorted list of non-overlapping resource occupations per resource
SortedResourceOccupationsPerResource = Dict[Resource, List[ResourceOccupation]]

# sorted list of resource occupations per agent; resource occupations overlap by release time!
SortedResourceOccupationsPerAgent = Dict[int, List[ResourceOccupation]]

# list of resource occupations per agent and time-step (there are multiple resource occupations if the previous resource is not released yet)
ResourceOccupationPerAgentAndTimeStep = Dict[Tuple[int, int], List[ResourceOccupation]]

ScheduleAsResourceOccupations = NamedTuple('ScheduleAsResourceOccupations', [
    ('sorted_resource_occupations_per_resource', SortedResourceOccupationsPerResource),
    ('sorted_resource_occupations_per_agent', SortedResourceOccupationsPerAgent),
    ('resource_occupations_per_agent_and_time_step', ResourceOccupationPerAgentAndTimeStep),
])

TimeWindow = ResourceOccupation
# list of time windows per resource sorted by lower bound; time windows may overlap!
TimeWindowsPerResourceAndTimeStep = Dict[Tuple[Resource, int], List[TimeWindow]]

# sorted list of time windows per agent
TimeWindowsPerAgentSortedByLowerBound = Dict[int, List[TimeWindow]]

SchedulingProblemInTimeWindows = NamedTuple('SchedulingProblemInTimeWindows', [
    ('time_windows_per_resource_and_time_step', TimeWindowsPerResourceAndTimeStep),
    ('time_windows_per_agent_sorted_by_lower_bound', TimeWindowsPerAgentSortedByLowerBound),
])


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
    row: DataFrame

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
        debug: bool = False,
        nonify_problem_and_results: bool = False
) -> ExperimentResultsAnalysis:
    """

    Parameters
    ----------
    experiment_results:
        experiment_results to expand into to experiment_results_analysis
        TODO SIM-418 cleanup of this workaround: what would be a good compromise between typing and memory usage?
    debug: bool
    nonify_problem_and_results: bool
        in order to save space, set results_* and problem_* fields to None. This may cause not all code to work any more.
        TODO SIM-418 cleanup of this workaround: what would be a good compromise between typing and memory usage?

    Returns
    -------

    """
    if not isinstance(experiment_results, ExperimentResults):
        experiment_results_as_dict = dict(experiment_results[0])
        experiment_results = ExperimentResults(**experiment_results_as_dict)
    experiment_parameters: ExperimentParameters = experiment_results.experiment_parameters
    experiment_id = experiment_parameters.experiment_id

    # derive speed up
    time_full = experiment_results.results_full.solver_statistics["summary"]["times"]["total"]
    time_full_after_malfunction = \
        experiment_results.results_full_after_malfunction.solver_statistics["summary"]["times"]["total"]
    time_delta_after_malfunction = \
        experiment_results.results_delta_after_malfunction.solver_statistics["summary"]["times"]["total"]
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
        warnings.warn(f"no resource conflicts for experiment {experiment_id} -> set ratio to -1")
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
        train_run_full_after_malfunction_agent: Trainrun = experiment_results.results_full_after_malfunction.trainruns_dict[
            agent_id]
        target_full_after_malfunction_agent: Waypoint = train_run_full_after_malfunction_agent[-1].waypoint

        train_run_full_after_malfunction_constraints_agent = \
            experiment_results.problem_full_after_malfunction.route_dag_constraints_dict[agent_id]
        train_run_full_after_malfunction_target_earliest_agent = \
            train_run_full_after_malfunction_constraints_agent.freeze_earliest[target_full_after_malfunction_agent]
        train_run_full_after_malfunction_scheduled_at_target = \
            train_run_full_after_malfunction_agent[-1].scheduled_at
        lateness_full_after_malfunction[agent_id] = \
            max(
                train_run_full_after_malfunction_scheduled_at_target -
                train_run_full_after_malfunction_target_earliest_agent,
                0)
        # TODO SIM-325 extend to all vertices
        vertex_eff_lateness_full_after_malfunction[agent_id] = {
            target_full_after_malfunction_agent: lateness_full_after_malfunction[agent_id]
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
        target_delta_after_malfunction_agent: Waypoint = train_run_delta_after_malfunction_agent[-1].waypoint
        train_run_delta_after_malfunction_target_agent = train_run_delta_after_malfunction_agent[-1]
        train_run_delta_after_malfunction_constraints_agent = \
            experiment_results.problem_delta_after_malfunction.route_dag_constraints_dict[agent_id]
        train_run_delta_after_malfunction_target_earliest_agent = \
            train_run_delta_after_malfunction_constraints_agent.freeze_earliest[target_delta_after_malfunction_agent]
        train_run_delta_after_malfunction_scheduled_at_target = \
            train_run_delta_after_malfunction_target_agent.scheduled_at
        lateness_delta_after_malfunction[agent_id] = \
            max(
                train_run_delta_after_malfunction_scheduled_at_target - train_run_delta_after_malfunction_target_earliest_agent,
                0)
        # TODO SIM-325 extend to all vertices
        vertex_eff_lateness_delta_after_malfunction[agent_id] = {
            target_delta_after_malfunction_agent: lateness_delta_after_malfunction[agent_id]
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
        **dict(
            experiment_results._asdict(),
            **({
                   'problem_full': None,
                   'problem_full_after_malfunction': None,
                   'problem_delta_after_malfunction': None,
                   'results_full': None,
                   'results_full_after_malfunction': None,
                   'results_delta_after_malfunction': None,
               } if nonify_problem_and_results else {}),
            **_expand_asp_solver_statistics_for_asp_plausi(r=experiment_results.results_full, suffix="full"),
            **_expand_asp_solver_statistics_for_asp_plausi(r=experiment_results.results_full_after_malfunction,
                                                           suffix="full_after_malfunction"),
            **_expand_asp_solver_statistics_for_asp_plausi(r=experiment_results.results_delta_after_malfunction,
                                                           suffix="delta_after_malfunction"),

        ),
        experiment_id=experiment_parameters.experiment_id,
        grid_id=experiment_parameters.grid_id,
        size=experiment_parameters.infra_parameters.width,
        n_agents=experiment_parameters.infra_parameters.number_of_agents,
        max_num_cities=experiment_parameters.infra_parameters.max_num_cities,
        max_rail_between_cities=experiment_parameters.infra_parameters.max_rail_between_cities,
        max_rail_in_city=experiment_parameters.infra_parameters.max_rail_in_city,
        time_full=time_full,
        time_full_after_malfunction=time_full_after_malfunction,
        time_delta_after_malfunction=time_delta_after_malfunction,
        solution_full=experiment_results.results_full.trainruns_dict,
        solution_full_after_malfunction=experiment_results.results_full_after_malfunction.trainruns_dict,
        solution_delta_after_malfunction=experiment_results.results_delta_after_malfunction.trainruns_dict,
        # scheduling without optimization has no empty costs array
        costs_full=(experiment_results.results_full.solver_statistics["summary"]["costs"][0]
                    if len(experiment_results.results_full.solver_statistics["summary"]["costs"]) > 0 else -1),
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


def _expand_asp_solver_statistics_for_asp_plausi(r: SchedulingExperimentResult, suffix: str):
    return {
        f'solve_total_ratio_{suffix}':
            catch_zero_division_error_as_minus_one(
                lambda:
                r.solver_statistics["summary"]["times"]["solve"] /
                r.solver_statistics["summary"]["times"]["total"]
            ),
        f'solve_time_{suffix}':
            r.solver_statistics["summary"]["times"]["solve"],
        f'total_time_{suffix}':
            r.solver_statistics["summary"]["times"]["total"],
        f'choice_conflict_ratio_{suffix}':
            catch_zero_division_error_as_minus_one(
                lambda:
                r.solver_statistics["solving"]["solvers"]["choices"] /
                r.solver_statistics["solving"]["solvers"]["conflicts"]
            ),
        f'choices_{suffix}':
            r.solver_statistics["solving"]["solvers"]["choices"],
        f'conflicts_{suffix}':
            r.solver_statistics["solving"]["solvers"]["conflicts"],
        f'user_accu_propagations_{suffix}':
            sum(map(lambda d: d["Propagation(s)"],
                    r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])) /
            len(r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"]),
        f'user_step_propagations_{suffix}':
            sum(map(lambda d: d["Propagation(s)"],
                    r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])) /
            len(r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"]),
    }


def convert_experiment_results_analysis_to_data_frame(experiment_results: ExperimentResultsAnalysis) -> DataFrame:
    """Converts experiment results to data frame.

    Parameters
    ----------
    experiment_results: ExperimentResults

    Returns
    -------
    DataFrame
    """
    return experiment_results._asdict()


def extract_path_search_space(experiment_results: ExperimentResults) -> Tuple[int, int, int]:
    route_dag_constraints_delta_afer_malfunction = experiment_results.problem_delta_after_malfunction.route_dag_constraints_dict
    route_dag_constraints_full_after_malfunction = experiment_results.problem_delta_after_malfunction.route_dag_constraints_dict
    route_dag_constraints_schedule = experiment_results.problem_full.route_dag_constraints_dict
    topo_dict_schedule = experiment_results.problem_full.topo_dict
    topo_dict_full_after_malfunction = experiment_results.problem_full_after_malfunction.topo_dict
    topo_dict_delta_after_malfunction = experiment_results.problem_delta_after_malfunction.topo_dict
    all_nb_alternatives_rsp_delta, all_nb_alternatives_rsp_full, all_nb_alternatives_schedule = extract_number_of_path_alternatives(
        topo_dict_full_after_malfunction=topo_dict_full_after_malfunction,
        topo_dict_schedule=topo_dict_schedule,
        topo_dict_delta_afer_malfunction=topo_dict_delta_after_malfunction,
        route_dag_constraints_schedule=route_dag_constraints_schedule,
        route_dag_constraints_delta_afer_malfunction=route_dag_constraints_delta_afer_malfunction,
        route_dag_constraints_full_after_malfunction=route_dag_constraints_full_after_malfunction
    )
    path_search_space_schedule = _prod(all_nb_alternatives_schedule)
    path_search_space_rsp_full = _prod(all_nb_alternatives_rsp_full)
    path_search_space_rsp_delta = _prod(all_nb_alternatives_rsp_delta)
    return path_search_space_rsp_delta, path_search_space_rsp_full, path_search_space_schedule


def extract_number_of_path_alternatives(
        topo_dict_schedule: TopoDict,
        topo_dict_delta_afer_malfunction: TopoDict,
        topo_dict_full_after_malfunction: TopoDict,
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
            topo=topo_dict_schedule[agent_id],
            route_dag_constraints=route_dag_constraints_schedule[agent_id]
        )
        alternatives_rsp_full = get_paths_for_route_dag_constraints(
            topo=topo_dict_full_after_malfunction[agent_id],
            route_dag_constraints=route_dag_constraints_full_after_malfunction[agent_id]
        )
        alternatives_rsp_delta = get_paths_for_route_dag_constraints(
            topo=topo_dict_delta_afer_malfunction[agent_id],
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


def experiment_freeze_dict_pretty_print(d: RouteDAGConstraintsDict):
    for agent_id, route_dag_constraints in d.items():
        prefix = f"agent {agent_id} "
        experiment_freeze_pretty_print(route_dag_constraints, prefix)


def experiment_freeze_pretty_print(route_dag_constraints: RouteDAGConstraints, prefix: str = ""):
    print(f"{prefix}freeze_visit={_pp.pformat(route_dag_constraints.freeze_visit)}")
    print(f"{prefix}freeze_earliest={_pp.pformat(route_dag_constraints.freeze_earliest)}")
    print(f"{prefix}freeze_latest={_pp.pformat(route_dag_constraints.freeze_latest)}")
    print(f"{prefix}freeze_banned={_pp.pformat(route_dag_constraints.freeze_banned)}")
