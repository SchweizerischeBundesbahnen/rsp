"""Data types used in the experiment for the real time rescheduling research
project."""
import pprint
import warnings
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Set
from typing import Tuple

import pandas as pd
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint
from numpy import inf
from pandas import DataFrame

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraintsDict
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.utils.general_helpers import catch_zero_division_error_as_minus_one
from rsp.utils.global_constants import DELAY_MODEL_RESOLUTION
from rsp.utils.global_constants import DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY
from rsp.utils.rsp_logger import rsp_logger

SpeedData = Mapping[float, float]

# @deprecated(reason="You should use hiearchy level ranges.")
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
# @deprecated(reason="You should use hiearchy level ranges.")
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
])

ScheduleParameters = NamedTuple('ScheduleParameters', [
    ('infra_id', int),
    ('schedule_id', int),

    ('asp_seed_value', int),
    ('number_of_shortest_paths_per_agent_schedule', int),
])

ScheduleParametersRange = NamedTuple('ScheduleParametersRange', [
    ('asp_seed_value', List[int]),
    ('number_of_shortest_paths_per_agent_schedule', List[int]),
])

ReScheduleParametersRange = NamedTuple('ReScheduleParametersRange', [
    # 5: earliest_malfunction
    ('earliest_malfunction', List[int]),
    # 6: malfunction_duration
    ('malfunction_duration', List[int]),

    #
    ('malfunction_agent_id', List[int]),

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

    ('malfunction_agent_id', int),

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
    ('malfunction_agent_id', int),
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
        malfunction_agent_id=[0, 0, 1],
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
    ('problem_delta_trivially_perfect_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_perfect_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_no_rerouting_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_online_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_random_after_malfunction', ScheduleProblemDescription),
    ('results_full', SchedulingExperimentResult),
    ('results_full_after_malfunction', SchedulingExperimentResult),
    ('results_delta_trivially_perfect_after_malfunction', SchedulingExperimentResult),
    ('results_delta_perfect_after_malfunction', SchedulingExperimentResult),
    ('results_delta_no_rerouting_after_malfunction', SchedulingExperimentResult),
    ('results_delta_online_after_malfunction', SchedulingExperimentResult),
    ('results_delta_random_after_malfunction', SchedulingExperimentResult),
    ('predicted_changed_agents_online_after_malfunction', Set[int]),
    ('predicted_changed_agents_random_after_malfunction', Set[int]),
])

speed_up_scopes = [f"delta_{infix}_after_malfunction" for infix in ['trivially_perfect', 'perfect', 'no_rerouting', 'online', 'random']]
prediction_scopes = [f"delta_{infix}_after_malfunction" for infix in ['online', 'random']]

after_malfunction_scopes = ['full_after_malfunction', ] + speed_up_scopes
all_scopes = ['full'] + after_malfunction_scopes


def solver_statistics_costs_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["costs"][0]


def solver_statistics_times_total_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["times"]["total"]


def solver_statistics_times_solve_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["times"]["solve"]


def trainrun_dict_from_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> TrainrunDict:
    return results.trainruns_dict


# TODO SIM-672 duplicate with solver_statistics
def costs_from_results(results_schedule: SchedulingExperimentResult,
                       results_reschedule: SchedulingExperimentResult,
                       problem_reschedule: ScheduleProblemDescription) -> float:
    return results_reschedule.solver_statistics["summary"]["costs"][0]


def nb_resource_conflicts_from_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.nb_conflicts


def solve_total_ratio_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return catch_zero_division_error_as_minus_one(
        lambda:
        r.solver_statistics["summary"]["times"]["solve"] /
        r.solver_statistics["summary"]["times"]["total"])


def choice_conflict_ratio_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return catch_zero_division_error_as_minus_one(
        lambda:
        r.solver_statistics["solving"]["solvers"]["choices"] /
        r.solver_statistics["solving"]["solvers"]["conflicts"]
    )


def size_used_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> int:
    used_cells: Set[Waypoint] = {
        waypoint for agent_id, topo in
        p.topo_dict.items()
        for waypoint in topo.nodes
    }
    return len(used_cells)


def solver_statistics_choices_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return r.solver_statistics["solving"]["solvers"]["choices"]


def solver_statistics_conflicts_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return r.solver_statistics["solving"]["solvers"]["conflicts"]


def summed_user_accu_propagations_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return sum(map(lambda d: d["Propagation(s)"],
                   r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])) / len(r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])


def summed_user_step_propagations_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return sum(map(lambda d: d["Propagation(s)"],
                   r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])) / len(r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])


def lateness_per_agent_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription):
    return {
        agent_id: results_reschedule.trainruns_dict[agent_id][-1].scheduled_at - results_schedule.trainruns_dict[agent_id][-1].scheduled_at
        for agent_id in results_reschedule.trainruns_dict
    }


def costs_from_lateness_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription):
    # TODO SIM-672 runs twice, optimize
    lateness_dict = lateness_per_agent_from_results(
        results_schedule=results_schedule,
        results_reschedule=results_reschedule,
        problem_reschedule=problem_reschedule
    )
    return lateness_to_effective_cost(
        weight_lateness_seconds=problem_reschedule.weight_lateness_seconds,
        lateness_dict=lateness_dict
    )


def lateness_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription
) -> float:
    # TODO SIM-672 runs twice, optimize
    return sum(lateness_per_agent_from_results(results_schedule, results_reschedule, problem_reschedule).values())


def changed_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription):
    return len([
        agent_id
        for agent_id in results_reschedule.trainruns_dict.keys()
        if set(results_schedule.trainruns_dict[agent_id]) != set(results_reschedule.trainruns_dict[agent_id])
    ])


def changed_percentage_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription):
    return sum([
        1 if set(results_schedule.trainruns_dict[agent_id]) != set(results_reschedule.trainruns_dict[agent_id]) else 0
        for agent_id in results_reschedule.trainruns_dict.keys()
    ]) / len(results_reschedule.trainruns_dict)


def vertex_lateness_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription) -> Dict[int, Dict[Waypoint, float]]:
    schedule = {
        agent_id: {
            trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
            for trainrun_waypoint in trainrun
        }
        for agent_id, trainrun in results_schedule.trainruns_dict.items()
    }
    return {
        agent_id: {
            trainrun_waypoint.waypoint: max(trainrun_waypoint.scheduled_at - schedule[agent_id].get(trainrun_waypoint.waypoint, inf), 0)
            for trainrun_waypoint in reschedule_trainrun
        }
        for agent_id, reschedule_trainrun in results_reschedule.trainruns_dict.items()
    }


def costs_from_route_section_penalties_per_agent_and_edge_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription) -> Dict[int, Dict[Tuple[Waypoint, Waypoint], int]]:
    edges_rescheduling = {
        agent_id: {
            (wp1.waypoint, wp2.waypoint)
            for wp1, wp2 in zip(reschedule_trainrun, reschedule_trainrun[1:])
        }
        for agent_id, reschedule_trainrun in results_reschedule.trainruns_dict.items()
    }

    return {
        agent_id: {
            edge: penalty
            for edge, penalty in problem_reschedule.route_section_penalties[agent_id].items()
            if edge in edges_rescheduling[agent_id]
        }
        for agent_id in results_reschedule.trainruns_dict.keys()
    }


def costs_from_route_section_penalties_per_agent_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription) -> Dict[int, float]:
    # TODO we should avoid redundant computations?
    edge_penalties = costs_from_route_section_penalties_per_agent_and_edge_from_results(
        results_schedule=results_schedule,
        results_reschedule=results_reschedule,
        problem_reschedule=problem_reschedule)
    return {agent_id: sum(edge_penalties.values()) for agent_id, edge_penalties in edge_penalties.items()}


def costs_from_route_section_penalties(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription) -> float:
    # TODO we should avoid redundant computations?
    edge_penalties = costs_from_route_section_penalties_per_agent_and_edge_from_results(
        results_schedule=results_schedule,
        results_reschedule=results_reschedule,
        problem_reschedule=problem_reschedule)
    return sum([sum(edge_penalties.values()) for agent_id, edge_penalties in edge_penalties.items()])


def speed_up_from_results(results_full_reschedule: SchedulingExperimentResult,
                          results_other_reschedule: SchedulingExperimentResult,
                          experiment_results: ExperimentResults) -> float:
    return results_full_reschedule.solver_statistics["summary"]["times"]["total"] / results_other_reschedule.solver_statistics["summary"]["times"]["total"]


def costs_ratio_from_results(results_full_reschedule: SchedulingExperimentResult,
                             results_other_reschedule: SchedulingExperimentResult,
                             experiment_results: ExperimentResults) -> float:
    costs_full_reschedule = costs_from_results(
        results_schedule=experiment_results.results_full,
        results_reschedule=results_full_reschedule,
        problem_reschedule=experiment_results.problem_full_after_malfunction
    )
    # TODO SIM-324 pull out verification
    assert costs_full_reschedule >= experiment_results.malfunction.malfunction_duration, \
        f"costs_full_reschedule {costs_full_reschedule} should be greater than malfunction duration, " \
        f"{experiment_results.malfunction} {experiment_results.experiment_parameters}"
    costs_other_reschedule = costs_from_results(
        results_schedule=experiment_results.results_full,
        results_reschedule=results_other_reschedule,
        problem_reschedule=experiment_results.problem_full_after_malfunction
    )
    # TODO SIM-324 pull out verification
    assert costs_other_reschedule >= experiment_results.malfunction.malfunction_duration
    try:
        return costs_full_reschedule / costs_other_reschedule,
    except ZeroDivisionError as e:
        rsp_logger.error(f"{costs_full_reschedule} / {costs_other_reschedule}")
        raise e


experiment_results_analysis_all_scopes_fields = {
    'solver_statistics_costs': (float, solver_statistics_costs_from_experiment_results),
    'solver_statistics_times_total': (float, solver_statistics_times_total_from_experiment_results),
    'solver_statistics_times_solve': (float, solver_statistics_times_solve_from_experiment_results),
    'solver_statistics_choices': (float, solver_statistics_choices_from_results),
    'solver_statistics_conflicts': (float, solver_statistics_conflicts_from_results),
    'summed_user_accu_propagations': (float, summed_user_accu_propagations_from_results),
    'summed_user_step_propagations': (float, summed_user_step_propagations_from_results),
    'solution': (TrainrunDict, trainrun_dict_from_results),
    'nb_resource_conflicts': (float, nb_resource_conflicts_from_results),
    'solve_total_ratio': (float, solve_total_ratio_from_results),
    'choice_conflict_ratio': (float, choice_conflict_ratio_from_results),
    'size_used': (int, size_used_from_results),
}

experiment_results_analysis_after_malfunction_scopes_fields = {
    'costs': (float, costs_from_results),
    'costs_from_route_section_penalties': (float, costs_from_route_section_penalties),
    'costs_from_route_section_penalties_per_agent': (Dict[int, float], costs_from_route_section_penalties_per_agent_from_results),
    'costs_from_route_section_penalties_per_agent_and_edge':
        (Dict[int, Dict[Tuple[Waypoint, Waypoint], float]], costs_from_route_section_penalties_per_agent_and_edge_from_results),
    'costs_from_lateness': (float, costs_from_lateness_from_results),

    # lateness unweighted
    'lateness': (float, lateness_from_results),
    'lateness_per_agent': (Dict[int, int], lateness_per_agent_from_results),
    'vertex_lateness': (Dict[int, Dict[Waypoint, float]], vertex_lateness_from_results),
    'changed_agents': (float, changed_from_results),
    'changed_agents_percentage': (float, changed_percentage_from_results),
}

speedup_scopes_fields = {
    'speed_up': (float, speed_up_from_results),
    'costs_ratio': (float, costs_ratio_from_results),
}

prediction_scopes_fields = {
    'predicted_changed_agents_number': int,
    'predicted_changed_agents_percentage': float,
    'predicted_changed_agents_false_positives_percentage': float,
    'predicted_changed_agents_false_negatives_percentage': float,
    'predicted_changed_agents_false_positives': int,
    'predicted_changed_agents_false_negatives': int,
}

ExperimentResultsAnalysis = NamedTuple('ExperimentResultsAnalysis', [
    ('experiment_parameters', ExperimentParameters),
    ('malfunction', ExperimentMalfunction),
    ('problem_full', ScheduleProblemDescription),
    ('problem_full_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_trivially_perfect_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_perfect_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_no_rerouting_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_online_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_random_after_malfunction', ScheduleProblemDescription),
    ('results_full', SchedulingExperimentResult),
    ('results_full_after_malfunction', SchedulingExperimentResult),
    ('results_delta_trivially_perfect_after_malfunction', SchedulingExperimentResult),
    ('results_delta_perfect_after_malfunction', SchedulingExperimentResult),
    ('results_delta_no_rerouting_after_malfunction', SchedulingExperimentResult),
    ('results_delta_online_after_malfunction', SchedulingExperimentResult),
    ('results_delta_random_after_malfunction', SchedulingExperimentResult),
    ('predicted_changed_agents_online_after_malfunction', Set[int]),
    ('predicted_changed_agents_random_after_malfunction', Set[int]),
    ('experiment_id', int),
    ('grid_id', int),
    ('infra_id', int),
    ('schedule_id', int),
    ('size', int),
    ('n_agents', int),
    ('max_num_cities', int),
    ('max_rail_between_cities', int),
    ('max_rail_in_city', int),
    ('earliest_malfunction', int),
    ('malfunction_duration', int),
    ('malfunction_agent_id', int),
    ('weight_route_change', int),
    ('weight_lateness_seconds', int),
    ('max_window_size_from_earliest', int),

    # ========================================
    # lower_bound / upper_bound
    # ========================================

    ('factor_resource_conflicts', int),

] + [
                                           (f'{prefix}_{scope}', type_)
                                           for prefix, (type_, _) in experiment_results_analysis_all_scopes_fields.items()
                                           for scope in all_scopes
                                       ] + [
                                           (f'{prefix}_{scope}', type_)
                                           for prefix, (type_, _) in experiment_results_analysis_after_malfunction_scopes_fields.items()
                                           for scope in after_malfunction_scopes
                                       ] + [
                                           (f'{prefix}_{scope}', type_)
                                           for prefix, (type_, _) in speedup_scopes_fields.items()
                                           for scope in speed_up_scopes
                                       ] + [
                                           (f'{prefix}_{scope}', type_)
                                           for prefix, type_ in prediction_scopes_fields.items()
                                           for scope in prediction_scopes
                                       ]
                                       )

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


def convert_list_of_experiment_results_analysis_to_data_frame(l: List[ExperimentResultsAnalysis]) -> DataFrame:
    return pd.DataFrame(columns=COLUMNS_ANALYSIS, data=[r._asdict() for r in l])


def filter_experiment_results_analysis_data_frame(experiment_data: pd.DataFrame) -> pd.DataFrame:
    time_full_after_malfunction = experiment_data['solver_statistics_times_total_full_after_malfunction']
    return experiment_data[
        (time_full_after_malfunction > 10) &
        (time_full_after_malfunction <= time_full_after_malfunction.quantile(.97))
        ]


def expand_experiment_results_for_analysis(
        experiment_results: ExperimentResults,
        nonify_all_structured_fields: bool = False
) -> ExperimentResultsAnalysis:
    """

    Parameters
    ----------
    experiment_results:
        experiment_results to expand into to experiment_results_analysis
        TODO SIM-418 cleanup of this workaround: what would be a good compromise between typing and memory usage?
    debug: bool
    nonify_all_structured_fields: bool
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
    nb_resource_conflicts_delta_perfect_after_malfunction = experiment_results.results_delta_perfect_after_malfunction.nb_conflicts
    nb_resource_conflicts_full_after_malfunction = experiment_results.results_full_after_malfunction.nb_conflicts
    # search space indiciators
    factor_resource_conflicts = -1
    try:
        factor_resource_conflicts = \
            nb_resource_conflicts_delta_perfect_after_malfunction / \
            nb_resource_conflicts_full_after_malfunction
    except ZeroDivisionError:
        warnings.warn(f"no resource conflicts for experiment {experiment_id} -> set ratio to -1")

    nb_agents = experiment_parameters.infra_parameters.number_of_agents

    ground_truth_positive_changed_agents = {
        agent_id
        for agent_id, trainrun_full_after_malfunction in experiment_results.results_full_after_malfunction.trainruns_dict.items()
        if set(experiment_results.results_full.trainruns_dict[agent_id]) != set(trainrun_full_after_malfunction)
    }

    ground_truth_negative_changed_agents = \
        (set(range(nb_agents)).difference(ground_truth_positive_changed_agents))

    def predicted_dict(predicted_changed_agents: Set[int], scope: str):
        predicted_changed_agents_false_positives = \
            (predicted_changed_agents.intersection(ground_truth_negative_changed_agents))
        predicted_changed_agents_false_negatives = \
            (set(range(nb_agents)).difference(predicted_changed_agents)).intersection(ground_truth_positive_changed_agents)
        return {
            f'predicted_changed_agents_number_{scope}': len(predicted_changed_agents),
            f'predicted_changed_agents_false_positives_{scope}': len(predicted_changed_agents_false_positives),
            f'predicted_changed_agents_false_negatives_{scope}': len(predicted_changed_agents_false_negatives),
            f'predicted_changed_agents_percentage_{scope}': (len(predicted_changed_agents) / nb_agents),
            f'predicted_changed_agents_false_positives_percentage_{scope}': (len(predicted_changed_agents_false_positives) /
                                                                             len(ground_truth_negative_changed_agents)),
            f'predicted_changed_agents_false_negatives_percentage_{scope}': (len(predicted_changed_agents_false_negatives) /
                                                                             len(ground_truth_positive_changed_agents))
        }

    online_predicted_dict = predicted_dict(experiment_results.predicted_changed_agents_online_after_malfunction, 'delta_online_after_malfunction')
    random_predicted_dict = predicted_dict(experiment_results.predicted_changed_agents_random_after_malfunction, 'delta_random_after_malfunction')
    d = dict(
        **experiment_results._asdict(),
        **{
            f'{prefix}_{scope}': results_extractor(experiment_results._asdict()[f'results_{scope}'], experiment_results._asdict()[f'problem_{scope}']) for
            prefix, (_, results_extractor) in
            experiment_results_analysis_all_scopes_fields.items() for scope in all_scopes
        },
        **{
            f'{prefix}_{scope}': results_extractor(
                results_schedule=experiment_results._asdict()[f'results_full'],
                results_reschedule=experiment_results._asdict()[f'results_{scope}'],
                problem_reschedule=experiment_results._asdict()[f'problem_{scope}'],
            )
            for prefix, (_, results_extractor) in experiment_results_analysis_after_malfunction_scopes_fields.items()
            for scope in after_malfunction_scopes
        },
        **{
            f'{prefix}_{scope}': results_extractor(results_full_reschedule=experiment_results._asdict()[f'results_full_after_malfunction'],
                                                   results_other_reschedule=experiment_results._asdict()[f'results_{scope}'],
                                                   experiment_results=experiment_results
                                                   )
            for prefix, (_, results_extractor) in speedup_scopes_fields.items()
            for scope in speed_up_scopes
        }

    )

    def _to_experiment_results_analysis():
        return ExperimentResultsAnalysis(
            experiment_id=experiment_parameters.experiment_id,
            grid_id=experiment_parameters.grid_id,
            size=experiment_parameters.infra_parameters.width,
            n_agents=experiment_parameters.infra_parameters.number_of_agents,
            max_num_cities=experiment_parameters.infra_parameters.max_num_cities,
            max_rail_between_cities=experiment_parameters.infra_parameters.max_rail_between_cities,
            max_rail_in_city=experiment_parameters.infra_parameters.max_rail_in_city,

            infra_id=experiment_parameters.infra_parameters.infra_id,
            schedule_id=experiment_parameters.schedule_parameters.schedule_id,
            earliest_malfunction=experiment_parameters.earliest_malfunction,
            malfunction_duration=experiment_parameters.malfunction_duration,
            malfunction_agent_id=experiment_parameters.malfunction_agent_id,
            weight_route_change=experiment_parameters.weight_route_change,
            weight_lateness_seconds=experiment_parameters.weight_lateness_seconds,
            max_window_size_from_earliest=experiment_parameters.max_window_size_from_earliest,

            factor_resource_conflicts=factor_resource_conflicts,

            **d,
            **online_predicted_dict,
            **random_predicted_dict,
        )

    plausibility_check_experiment_results_analysis(experiment_results_analysis=_to_experiment_results_analysis())

    # nonify all non-float fields
    if nonify_all_structured_fields:
        d.update({f"problem_{scope}": None for scope in all_scopes})
        d.update({f"results_{scope}": None for scope in all_scopes})
        d.update({f"solution_{scope}": None for scope in all_scopes})
        d.update({f"lateness_per_agent_{scope}": None for scope in after_malfunction_scopes})
        d.update({f"costs_from_route_section_penalties_per_agent_{scope}": None for scope in after_malfunction_scopes})
        d.update({f"vertex_lateness_{scope}": None for scope in after_malfunction_scopes})
        d.update({f"costs_from_route_section_penalties_per_agent_and_edge_{scope}": None for scope in after_malfunction_scopes})
        d.update({
            'experiment_parameters': None,
            'malfunction': None,
            'predicted_changed_agents_online_after_malfunction': None,
            'predicted_changed_agents_random_after_malfunction': None,
        })

    return _to_experiment_results_analysis()


def plausibility_check_experiment_results(experiment_results: ExperimentResults):
    """Verify the following experiment expectations:

    1. a) same waypoint in schedule and re-schedule -> waypoint also in scope perfect re-schedule
       b) same waypoint and time in schedule and re-schedule -> same waypoint and ant time also in re-schedule delta perfect
    2. number of routing alternatives should be decreasing from full to delta
    """
    route_dag_constraints_delta_perfect_after_malfunction = experiment_results.results_delta_perfect_after_malfunction.route_dag_constraints

    # 1.
    for agent_id in route_dag_constraints_delta_perfect_after_malfunction:
        # b) S0[x] == S[x]) ==> S'[x]: path and time
        schedule: Set[TrainrunWaypoint] = frozenset(experiment_results.results_full.trainruns_dict[agent_id])
        reschedule_full: Set[TrainrunWaypoint] = frozenset(
            experiment_results.results_full_after_malfunction.trainruns_dict[agent_id])
        reschedule_delta_perfect: Set[TrainrunWaypoint] = frozenset(
            experiment_results.results_delta_perfect_after_malfunction.trainruns_dict[agent_id])
        assert schedule.intersection(reschedule_full).issubset(reschedule_delta_perfect)

        # a) S0[x] == S[x]) ==> S'[x]: path
        waypoints_schedule: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.results_full.trainruns_dict[agent_id]
        }
        waypoints_reschedule_full: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.results_full_after_malfunction.trainruns_dict[agent_id]
        }
        waypoints_reschedule_delta_perfect: Set[Waypoint] = {
            trainrun_waypoint.waypoint
            for trainrun_waypoint in experiment_results.results_delta_perfect_after_malfunction.trainruns_dict[agent_id]
        }
        assert waypoints_schedule.intersection(waypoints_reschedule_full).issubset(waypoints_reschedule_delta_perfect)

    # 2. plausibility test: number of alternatives should be decreasing from full to delta
    for agent_id in route_dag_constraints_delta_perfect_after_malfunction:
        node_set_full_after_malfunction = set(experiment_results.problem_full_after_malfunction.topo_dict[agent_id].nodes)
        node_set_delta_perfect_after_malfunction = set(experiment_results.problem_delta_perfect_after_malfunction.topo_dict[agent_id].nodes)
        if not node_set_full_after_malfunction.issuperset(node_set_delta_perfect_after_malfunction):
            assert node_set_full_after_malfunction.issuperset(node_set_delta_perfect_after_malfunction), \
                f"{agent_id}: not all nodes from delta perfect are also in full: only delta perfect " \
                f"{node_set_delta_perfect_after_malfunction.difference(node_set_full_after_malfunction)}, " \
                f"malfunction={experiment_results.malfunction}"


def plausibility_check_experiment_results_analysis(experiment_results_analysis: ExperimentResultsAnalysis):
    experiment_id = experiment_results_analysis.experiment_id
    plausibility_check_experiment_results(experiment_results=experiment_results_analysis)

    # sanity check costs
    for scope in after_malfunction_scopes:
        costs = experiment_results_analysis._asdict()[f'costs_{scope}']
        costs_from_route_section_penalties = experiment_results_analysis._asdict()[f'costs_from_route_section_penalties_{scope}']
        costs_from_lateness = experiment_results_analysis._asdict()[f'costs_from_lateness_{scope}']
        # TODO make hard assert again
        try:
            assert costs == (
                    costs_from_lateness + costs_from_route_section_penalties), \
                f"experiment {experiment_id}: " \
                f"costs_{scope}={costs}, " \
                f"costs_from_lateness_{scope}={costs_from_lateness}, " \
                f"costs_from_route_section_penalties_{scope}={costs_from_route_section_penalties} "
        except AssertionError as e:
            rsp_logger.warn(str(e))
    for scope in ['trivially_perfect', 'perfect']:
        costs = experiment_results_analysis._asdict()[f'costs_delta_{scope}_after_malfunction']
        assert costs == experiment_results_analysis.costs_full_after_malfunction
    for scope in ['no_rerouting', 'random', 'online']:
        costs = experiment_results_analysis._asdict()[f'costs_delta_{scope}_after_malfunction']
        assert costs >= experiment_results_analysis.costs_full_after_malfunction


def lateness_to_effective_cost(
        weight_lateness_seconds: int,
        lateness_dict: Dict[int, int]) -> Dict[int, int]:
    """Map lateness per agent to costs for lateness.

    Parameters
    ----------
    weight_lateness_seconds
    lateness_dict

    Returns
    -------
    """
    penalty_leap_at = DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY
    penalty_leap = 5000000 + penalty_leap_at * weight_lateness_seconds
    return sum([(penalty_leap
                 if lateness > penalty_leap_at
                 else (lateness // DELAY_MODEL_RESOLUTION) * DELAY_MODEL_RESOLUTION * weight_lateness_seconds)
                for agent_id, lateness in lateness_dict.items()])


_pp = pprint.PrettyPrinter(indent=4)


def experiment_freeze_dict_pretty_print(d: RouteDAGConstraintsDict):
    for agent_id, route_dag_constraints in d.items():
        prefix = f"agent {agent_id} "
        experiment_freeze_pretty_print(route_dag_constraints, prefix)


def experiment_freeze_pretty_print(route_dag_constraints: RouteDAGConstraints, prefix: str = ""):
    print(f"{prefix}earliest={_pp.pformat(route_dag_constraints.earliest)}")
    print(f"{prefix}latest={_pp.pformat(route_dag_constraints.latest)}")
