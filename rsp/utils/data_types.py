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
from flatland.envs.rail_trainrun_data_structures import Waypoint
from numpy import inf
from pandas import DataFrame

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraintsDict
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.utils.general_helpers import catch_zero_division_error_as_minus_one

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

    ('malfunction_agend_id', int),

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
    ('malfunction_agend_id', int),
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
    ('problem_delta_perfect_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_naive_after_malfunction', ScheduleProblemDescription),
    ('results_full', SchedulingExperimentResult),
    ('results_full_after_malfunction', SchedulingExperimentResult),
    ('results_delta_perfect_after_malfunction', SchedulingExperimentResult),
    ('results_delta_naive_after_malfunction', SchedulingExperimentResult),
])

# TODO SIM-672 naming???
speed_up_scopes = ['delta_perfect_after_malfunction', 'delta_naive_after_malfunction']

after_malfunction_scopes = ['full_after_malfunction', ] + speed_up_scopes
all_scopes = ['full'] + after_malfunction_scopes


def time_from_experiment_results(results: SchedulingExperimentResult) -> float:
    return results.solver_statistics["summary"]["times"]["total"]


def solve_time_from_experiment_results(results: SchedulingExperimentResult) -> float:
    return results.solver_statistics["summary"]["times"]["solve"]


def total_delay_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription
) -> float:
    # TODO SIM-672 duplicate code
    def get_delay_trainruns_dict(trainruns_dict_schedule: TrainrunDict, trainruns_dict_reschedule: TrainrunDict):
        return sum([
            max(trainruns_dict_reschedule[agent_id][-1].scheduled_at - trainruns_dict_schedule[agent_id][-1].scheduled_at,
                0)
            for agent_id in trainruns_dict_reschedule])

    return get_delay_trainruns_dict(
        trainruns_dict_schedule=results_schedule.trainruns_dict,
        trainruns_dict_reschedule=results_reschedule.trainruns_dict
    )


def trainrun_dict_from_results(results: SchedulingExperimentResult) -> TrainrunDict:
    return results.trainruns_dict


def costs_from_results(results: SchedulingExperimentResult) -> float:
    return results.solver_statistics["summary"]["costs"][0] \
        if len(results.solver_statistics["summary"]["costs"]) > 0 \
        else -1


def nb_resource_conflicts_from_results(results: SchedulingExperimentResult) -> float:
    return results.nb_conflicts


def solve_total_ratio_from_results(r: SchedulingExperimentResult) -> float:
    return catch_zero_division_error_as_minus_one(
        lambda:
        r.solver_statistics["summary"]["times"]["solve"] /
        r.solver_statistics["summary"]["times"]["total"])


def choice_conflict_ratio_from_results(r: SchedulingExperimentResult) -> float:
    return catch_zero_division_error_as_minus_one(
        lambda:
        r.solver_statistics["solving"]["solvers"]["choices"] /
        r.solver_statistics["solving"]["solvers"]["conflicts"]
    )


def choices_from_results(r: SchedulingExperimentResult) -> float:
    return r.solver_statistics["solving"]["solvers"]["choices"]


def conflicts_from_results(r: SchedulingExperimentResult) -> float:
    return r.solver_statistics["solving"]["solvers"]["conflicts"]


def user_accu_propagations_from_results(r: SchedulingExperimentResult) -> float:
    return sum(map(lambda d: d["Propagation(s)"],
                   r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])) / len(r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])


def user_step_propagations_from_results(r: SchedulingExperimentResult) -> float:
    return sum(map(lambda d: d["Propagation(s)"],
                   r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])) / len(r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])


def lateness_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription):
    return {
        agent_id: max(
            results_reschedule.trainruns_dict[agent_id][-1].scheduled_at -
            problem_reschedule.route_dag_constraints_dict[agent_id].earliest[results_schedule.trainruns_dict[agent_id][-1].waypoint],
            0)
        for agent_id in results_reschedule.trainruns_dict.keys()
    }


def changed_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription):
    return sum([
        1 if set(results_schedule.trainruns_dict[agent_id]) != set(results_reschedule.trainruns_dict[agent_id]) else 0
        for agent_id in results_reschedule.trainruns_dict.keys()
    ]) / len(results_reschedule.trainruns_dict)


def vertex_eff_lateness_from_results(
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


def edge_eff_route_penalties_from_results(
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


def sum_route_section_penalties_from_results(
        results_schedule: SchedulingExperimentResult,
        results_reschedule: SchedulingExperimentResult,
        problem_reschedule: ScheduleProblemDescription) -> Dict[int, float]:
    # TODO we should avoid redundant computations?
    edge_penalties = edge_eff_route_penalties_from_results(
        results_schedule=results_schedule,
        results_reschedule=results_reschedule,
        problem_reschedule=problem_reschedule)
    return {agent_id: sum(edge_penalties.values()) for agent_id, edge_penalties in edge_penalties.items()}


def speed_up_from_results(results_full_schedule: SchedulingExperimentResult,
                          results_other_reschedule: SchedulingExperimentResult) -> float:
    return sum([
        1
        for agent_id, edge_penalties in results_other_reschedule.trainruns_dict.items()
        if set(results_full_schedule.trainruns_dict[agent_id]) != set(results_other_reschedule.trainruns_dict[agent_id])
    ]) / len(results_other_reschedule.trainruns_dict)


experiment_results_analysis_all_scopes_fields = {
    'time': (float, time_from_experiment_results),
    'solve_time': (float, solve_time_from_experiment_results),
    'solution': (TrainrunDict, trainrun_dict_from_results),
    'costs': (float, costs_from_results),
    'nb_resource_conflicts': (float, nb_resource_conflicts_from_results),
    'solve_total_ratio': (float, solve_total_ratio_from_results),
    'choice_conflict_ratio': (float, choice_conflict_ratio_from_results),
    'choices': (float, choices_from_results),
    'conflicts': (float, conflicts_from_results),
    'user_accu_propagations': (float, user_accu_propagations_from_results),
    'user_step_propagations': (float, user_step_propagations_from_results),
}

experiment_results_analysis_after_malfunction_scopes_fields = {
    'total_delay': (float, total_delay_from_results),
    'lateness': (Dict[int, int], lateness_from_results),
    'changed_agents_percentage': (float, changed_from_results),
    'sum_route_section_penalties': (Dict[int, float], sum_route_section_penalties_from_results),
    'vertex_eff_lateness': (Dict[int, Dict[Waypoint, float]], vertex_eff_lateness_from_results),
    'edge_eff_route_penalties': (Dict[int, Dict[Tuple[Waypoint, Waypoint], float]], edge_eff_route_penalties_from_results)
}

speedup_scopes_fields = {
    'speed_up': (float, speed_up_from_results),
}

ExperimentResultsAnalysis = NamedTuple('ExperimentResultsAnalysis', [
    ('experiment_parameters', ExperimentParameters),
    ('malfunction', ExperimentMalfunction),
    ('problem_full', ScheduleProblemDescription),
    ('problem_full_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_perfect_after_malfunction', ScheduleProblemDescription),
    ('problem_delta_naive_after_malfunction', ScheduleProblemDescription),
    ('results_full', SchedulingExperimentResult),
    ('results_full_after_malfunction', SchedulingExperimentResult),
    ('results_delta_perfect_after_malfunction', SchedulingExperimentResult),
    ('results_delta_naive_after_malfunction', SchedulingExperimentResult),
    ('experiment_id', int),
    ('grid_id', int),
    ('size', int),
    ('n_agents', int),
    ('max_num_cities', int),
    ('max_rail_between_cities', int),
    ('max_rail_in_city', int),

    # TODO SIM-672 should it be separated?
    ('size_used', int),

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


def expand_experiment_results_for_analysis(
        experiment_results: ExperimentResults,
        # TODO SIM-672 rename: nonify_all_structured_fields?
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
    used_cells: Set[Waypoint] = {
        waypoint for agent_id, topo in
        experiment_results.problem_full.topo_dict.items()
        for waypoint in topo.nodes
    }
    d = dict(
        **experiment_results._asdict(),
        **{
            f'{prefix}_{scope}': results_extractor(experiment_results._asdict()[f'results_{scope}']) for prefix, (_, results_extractor) in
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
            f'{prefix}_{scope}': results_extractor(results_full_schedule=experiment_results._asdict()[f'results_full'],
                                                   results_other_reschedule=experiment_results._asdict()[f'results_{scope}'],
                                                   )
            for prefix, (_, results_extractor) in speedup_scopes_fields.items()
            for scope in speed_up_scopes
        }
    )
    # nonify all non-float fields
    d.update({
                 'problem_full': None,
                 'problem_full_after_malfunction': None,
                 'problem_delta_perfect_after_malfunction': None,
                 'problem_delta_naive_after_malfunction': None,

                 'results_full': None,
                 'results_full_after_malfunction': None,
                 'results_delta_perfect_after_malfunction': None,
                 'results_delta_naive_after_malfunction': None,

                 'solution_full': None,
                 'solution_full_after_malfunction': None,
                 'solution_delta_perfect_after_malfunction': None,
                 'solution_delta_naive_after_malfunction': None,

                 'lateness_full_after_malfunction': None,
                 'lateness_delta_perfect_after_malfunction': None,
                 'lateness_delta_naive_after_malfunction': None,

                 'sum_route_section_penalties_full_after_malfunction': None,
                 'sum_route_section_penalties_delta_perfect_after_malfunction': None,
                 'sum_route_section_penalties_delta_naive_after_malfunction': None,

                 'vertex_eff_lateness_full_after_malfunction': None,
                 'vertex_eff_lateness_delta_perfect_after_malfunction': None,
                 'vertex_eff_lateness_delta_naive_after_malfunction': None,

                 'edge_eff_route_penalties_full_after_malfunction': None,
                 'edge_eff_route_penalties_delta_perfect_after_malfunction': None,
                 'edge_eff_route_penalties_delta_naive_after_malfunction': None,

                 'experiment_parameters': None,

                 'malfunction': None,
             } if nonify_problem_and_results else {})

    return ExperimentResultsAnalysis(
        experiment_id=experiment_parameters.experiment_id,
        grid_id=experiment_parameters.grid_id,
        size=experiment_parameters.infra_parameters.width,
        n_agents=experiment_parameters.infra_parameters.number_of_agents,
        max_num_cities=experiment_parameters.infra_parameters.max_num_cities,
        max_rail_between_cities=experiment_parameters.infra_parameters.max_rail_between_cities,
        max_rail_in_city=experiment_parameters.infra_parameters.max_rail_in_city,
        factor_resource_conflicts=factor_resource_conflicts,
        size_used=len(used_cells),
        **d
    )


_pp = pprint.PrettyPrinter(indent=4)


def experiment_freeze_dict_pretty_print(d: RouteDAGConstraintsDict):
    for agent_id, route_dag_constraints in d.items():
        prefix = f"agent {agent_id} "
        experiment_freeze_pretty_print(route_dag_constraints, prefix)


def experiment_freeze_pretty_print(route_dag_constraints: RouteDAGConstraints, prefix: str = ""):
    print(f"{prefix}earliest={_pp.pformat(route_dag_constraints.earliest)}")
    print(f"{prefix}latest={_pp.pformat(route_dag_constraints.latest)}")
