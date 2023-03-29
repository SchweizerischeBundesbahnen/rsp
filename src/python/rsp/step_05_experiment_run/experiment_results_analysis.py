"""`ExperimentResultsAnalysis` contains data structures for analysis,
with/without raw `ExperimentResults`.

Data structure should be mostly be flat (fields should be numbers) and
only some agent dicts that are often re-used.
"""
# TODO cleaner data structures without optionals?
import warnings
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Set
from typing import Tuple

import numpy as np
import pandas as pd
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from numpy import inf
from pandas import DataFrame

from rsp.scheduling.schedule import SchedulingExperimentResult
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_agenda_expansion.global_constants import GLOBAL_CONSTANTS
from rsp.step_05_experiment_run.experiment_malfunction import ExperimentMalfunction
from rsp.step_05_experiment_run.experiment_results import ExperimentResults
from rsp.step_05_experiment_run.experiment_results import plausibility_check_experiment_results
from rsp.utils.catch_zero_division import catch_zero_division_error_as_minus_one
from rsp.utils.rsp_logger import rsp_logger

speed_up_scopes = [
    "offline_fully_restricted",
    "offline_delta",
    "offline_delta_weak",
    # "online_route_restricted",
    # "online_transmission_chains_fully_restricted",
    "online_transmission_chains_route_restricted",
] + [f"online_random_{i}" for i in range(GLOBAL_CONSTANTS.NB_RANDOM)]

prediction_scopes = ["online_transmission_chains_route_restricted"] + [f"online_random_{i}" for i in range(GLOBAL_CONSTANTS.NB_RANDOM)]
rescheduling_scopes = ["online_unrestricted"] + speed_up_scopes
all_scopes = ["schedule"] + rescheduling_scopes


def _extract_visulization(l: List[str]):
    return [scope for scope in l if not scope.startswith("online_random_")] + ["online_random_average"]


all_scopes_visualization = _extract_visulization(all_scopes)
rescheduling_scopes_visualization = _extract_visulization(rescheduling_scopes)
prediction_scopes_visualization = _extract_visulization(prediction_scopes)
speed_up_scopes_visualization = _extract_visulization(speed_up_scopes)


def solver_statistics_costs_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["costs"][0]


def solver_statistics_times_total_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["times"]["total"]


def solver_statistics_times_solve_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["times"]["solve"]


def solver_statistics_times_unsat_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["times"]["unsat"]


def solver_statistics_times_sat_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["times"]["sat"]


def solver_statistics_times_total_without_solve_from_experiment_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.solver_statistics["summary"]["times"]["total"] - results.solver_statistics["summary"]["times"]["solve"]


def trainrun_dict_from_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> TrainrunDict:
    return results.trainruns_dict


# TODO duplicate with solver_statistics_costs
def costs_from_results(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
) -> float:
    return results_reschedule.solver_statistics["summary"]["costs"][0]


def nb_resource_conflicts_from_results(results: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return results.nb_conflicts


def solve_total_ratio_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return catch_zero_division_error_as_minus_one(lambda: r.solver_statistics["summary"]["times"]["solve"] / r.solver_statistics["summary"]["times"]["total"])


def choice_conflict_ratio_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return catch_zero_division_error_as_minus_one(
        lambda: r.solver_statistics["solving"]["solvers"]["choices"] / r.solver_statistics["solving"]["solvers"]["conflicts"]
    )


def size_used_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> int:
    used_cells: Set[Waypoint] = {waypoint for agent_id, topo in p.topo_dict.items() for waypoint in topo.nodes}
    return len(used_cells)


def solver_statistics_choices_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return r.solver_statistics["solving"]["solvers"]["choices"]


def solver_statistics_conflicts_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return r.solver_statistics["solving"]["solvers"]["conflicts"]


def summed_user_accu_propagations_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return sum(map(lambda d: d["Propagation(s)"], r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"])) / len(
        r.solver_statistics["user_accu"]["DifferenceLogic"]["Thread"]
    )


def summed_user_step_propagations_from_results(r: SchedulingExperimentResult, p: ScheduleProblemDescription) -> float:
    return sum(map(lambda d: d["Propagation(s)"], r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"])) / len(
        r.solver_statistics["user_step"]["DifferenceLogic"]["Thread"]
    )


def lateness_per_agent_from_results(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
):
    return {
        agent_id: results_reschedule.trainruns_dict[agent_id][-1].scheduled_at - results_schedule.trainruns_dict[agent_id][-1].scheduled_at
        for agent_id in results_reschedule.trainruns_dict
    }


def costs_from_lateness_from_results(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
):
    # TODO SIM-672 runs twice, optimize (extractors returning multiple fields?)
    lateness_dict = lateness_per_agent_from_results(
        results_schedule=results_schedule, results_reschedule=results_reschedule, problem_reschedule=problem_reschedule
    )
    return lateness_to_effective_cost(weight_lateness_seconds=problem_reschedule.weight_lateness_seconds, lateness_dict=lateness_dict)


def lateness_from_results(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
) -> float:
    # TODO SIM-672 runs twice, optimize (extractors returning multiple fields?)
    return sum(lateness_per_agent_from_results(results_schedule, results_reschedule, problem_reschedule).values())


def changed_from_results(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
):
    return len(
        [
            agent_id
            for agent_id in results_reschedule.trainruns_dict.keys()
            if set(results_schedule.trainruns_dict[agent_id]) != set(results_reschedule.trainruns_dict[agent_id])
        ]
    )


def changed_percentage_from_results(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
):
    return sum(
        [
            1 if set(results_schedule.trainruns_dict[agent_id]) != set(results_reschedule.trainruns_dict[agent_id]) else 0
            for agent_id in results_reschedule.trainruns_dict.keys()
        ]
    ) / len(results_reschedule.trainruns_dict)


def vertex_lateness_from_results(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
) -> Dict[int, Dict[Waypoint, float]]:
    schedule = {
        agent_id: {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in trainrun}
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
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
) -> Dict[int, Dict[Tuple[Waypoint, Waypoint], int]]:
    edges_rescheduling = {
        agent_id: {(wp1.waypoint, wp2.waypoint) for wp1, wp2 in zip(reschedule_trainrun, reschedule_trainrun[1:])}
        for agent_id, reschedule_trainrun in results_reschedule.trainruns_dict.items()
    }

    return {
        agent_id: {edge: penalty for edge, penalty in problem_reschedule.route_section_penalties[agent_id].items() if edge in edges_rescheduling[agent_id]}
        for agent_id in results_reschedule.trainruns_dict.keys()
    }


def costs_from_route_section_penalties_per_agent_from_results(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
) -> Dict[int, float]:
    # TODO SIM-672 runs twice, optimize (extractors returning multiple fields?)
    edge_penalties = costs_from_route_section_penalties_per_agent_and_edge_from_results(
        results_schedule=results_schedule, results_reschedule=results_reschedule, problem_reschedule=problem_reschedule
    )
    return {agent_id: sum(edge_penalties.values()) for agent_id, edge_penalties in edge_penalties.items()}


def costs_from_route_section_penalties(
    results_schedule: SchedulingExperimentResult, results_reschedule: SchedulingExperimentResult, problem_reschedule: ScheduleProblemDescription
) -> float:
    # TODO SIM-672 runs twice, optimize (extractors returning multiple fields?)
    edge_penalties = costs_from_route_section_penalties_per_agent_and_edge_from_results(
        results_schedule=results_schedule, results_reschedule=results_reschedule, problem_reschedule=problem_reschedule
    )
    return sum([sum(edge_penalties.values()) for agent_id, edge_penalties in edge_penalties.items()])


def lateness_to_effective_cost(weight_lateness_seconds: int, lateness_dict: Dict[int, int]) -> Dict[int, int]:
    """Map lateness per agent to costs for lateness.

    Parameters
    ----------
    weight_lateness_seconds
    lateness_dict

    Returns
    -------
    """
    penalty_leap_at = GLOBAL_CONSTANTS.DELAY_MODEL_UPPER_BOUND_LINEAR_PENALTY
    penalty_leap = 5000000 + penalty_leap_at * weight_lateness_seconds
    return sum(
        [
            (
                penalty_leap
                if lateness > penalty_leap_at
                else (lateness // GLOBAL_CONSTANTS.DELAY_MODEL_RESOLUTION) * GLOBAL_CONSTANTS.DELAY_MODEL_RESOLUTION * weight_lateness_seconds
            )
            for agent_id, lateness in lateness_dict.items()
        ]
    )


experiment_results_analysis_all_scopes_fields = {
    "solver_statistics_costs": (float, solver_statistics_costs_from_experiment_results),
    "solver_statistics_times_total": (float, solver_statistics_times_total_from_experiment_results),
    "solver_statistics_times_solve": (float, solver_statistics_times_solve_from_experiment_results),
    "solver_statistics_times_sat": (float, solver_statistics_times_sat_from_experiment_results),
    "solver_statistics_times_unsat": (float, solver_statistics_times_unsat_from_experiment_results),
    "solver_statistics_times_total_without_solve": (float, solver_statistics_times_total_without_solve_from_experiment_results),
    "solver_statistics_choices": (float, solver_statistics_choices_from_results),
    "solver_statistics_conflicts": (float, solver_statistics_conflicts_from_results),
    "summed_user_accu_propagations": (float, summed_user_accu_propagations_from_results),
    "summed_user_step_propagations": (float, summed_user_step_propagations_from_results),
    "solution": (TrainrunDict, trainrun_dict_from_results),
    "nb_resource_conflicts": (float, nb_resource_conflicts_from_results),
    "solve_total_ratio": (float, solve_total_ratio_from_results),
    "choice_conflict_ratio": (float, choice_conflict_ratio_from_results),
    "size_used": (int, size_used_from_results),
}

experiment_results_analysis_rescheduling_scopes_fields = {
    "costs": (float, costs_from_results),
    "costs_from_route_section_penalties": (float, costs_from_route_section_penalties),
    "costs_from_route_section_penalties_per_agent": (Dict[int, float], costs_from_route_section_penalties_per_agent_from_results),
    "costs_from_route_section_penalties_per_agent_and_edge": (
        Dict[int, Dict[Tuple[Waypoint, Waypoint], float]],
        costs_from_route_section_penalties_per_agent_and_edge_from_results,
    ),
    "costs_from_lateness": (float, costs_from_lateness_from_results),
    # lateness unweighted
    "lateness": (float, lateness_from_results),
    "lateness_per_agent": (Dict[int, int], lateness_per_agent_from_results),
    "vertex_lateness": (Dict[int, Dict[Waypoint, float]], vertex_lateness_from_results),
    "changed_agents": (float, changed_from_results),
    "changed_agents_percentage": (float, changed_percentage_from_results),
}

# TODO we could make this more systematic by calling fields ratio_XXX (e.g."speed_up" should become "ratio_solver_statistics_times_total")
# TODO streamline "non_solve_time" and "times_total_without_solve"!
# ratio to online_unrestricted
speedup_scopes_ratio_fields = {
    "speed_up": "solver_statistics_times_total",
    "costs_ratio": "solver_statistics_costs",
    "speed_up_solve_time": "solver_statistics_times_solve",
    "speed_up_non_solve_time": "solver_statistics_times_total_without_solve",
    "nb_resource_conflicts_ratio": "nb_resource_conflicts",
    "solver_statistics_conflicts_ratio": "solver_statistics_conflicts",
    "solver_statistics_choices_ratio": "solver_statistics_choices",
}
# difference to online_unrestricted
speedup_scopes_additional_fields = ["changed_agents", "costs", "lateness", "costs_from_route_section_penalties"]

prediction_scopes_fields = {
    "predicted_changed_agents_number": int,
    "predicted_changed_agents_percentage": float,
    "predicted_changed_agents_false_positives_percentage": float,
    "predicted_changed_agents_false_negatives_percentage": float,
    "predicted_changed_agents_false_positives": int,
    "predicted_changed_agents_false_negatives": int,
    "prediction_quality": int,
}

online_random_average_fields = [
    prefix
    for prefix in (
        list(speedup_scopes_ratio_fields.keys())
        + [f"additional_{f}" for f in speedup_scopes_additional_fields]
        + list(experiment_results_analysis_all_scopes_fields.keys())
        + list(experiment_results_analysis_rescheduling_scopes_fields.keys())
        + list(prediction_scopes_fields.keys())
    )
    if prefix != "solution" and "_per_" not in prefix and not prefix.startswith("vertex_")
]

ExperimentResultsAnalysis = NamedTuple(
    "ExperimentResultsAnalysis",
    [
        ("experiment_id", int),
        ("grid_id", int),
        ("infra_id", int),
        ("schedule_id", int),
        ("infra_id_schedule_id", int),
        ("size", int),
        ("n_agents", int),
        ("max_num_cities", int),
        ("max_rail_between_cities", int),
        ("max_rail_in_city", int),
        ("earliest_malfunction", int),
        ("malfunction_duration", int),
        ("malfunction_agent_id", int),
        ("weight_route_change", int),
        ("weight_lateness_seconds", int),
        ("max_window_size_from_earliest", int),
        ("n_agents_running", int),
        ("rescheduling_horizon", int),
        ("factor_resource_conflicts", int),
    ]
    + [(f"{prefix}_{scope}", type_) for prefix, (type_, _) in experiment_results_analysis_all_scopes_fields.items() for scope in all_scopes]
    + [(f"{prefix}_{scope}", type_) for prefix, (type_, _) in experiment_results_analysis_rescheduling_scopes_fields.items() for scope in rescheduling_scopes]
    + [(f"{prefix}_{scope}", float) for prefix in speedup_scopes_ratio_fields for scope in speed_up_scopes]
    + [(f"additional_{prefix}_{scope}", float) for prefix in speedup_scopes_additional_fields for scope in speed_up_scopes]
    + [(f"{prefix}_{scope}", type_) for prefix, type_ in prediction_scopes_fields.items() for scope in prediction_scopes]
    + [(f"{prefix}_online_random_average", float) for prefix in online_random_average_fields],
)


def plausibility_check_experiment_results_analysis(experiment_results_analysis: ExperimentResultsAnalysis):
    experiment_id = experiment_results_analysis.experiment_id

    # sanity check costs
    for scope in rescheduling_scopes:
        costs = experiment_results_analysis._asdict()[f"costs_{scope}"]
        costs_from_route_section_penalties = experiment_results_analysis._asdict()[f"costs_from_route_section_penalties_{scope}"]
        costs_from_lateness = experiment_results_analysis._asdict()[f"costs_from_lateness_{scope}"]
        # TODO make hard assert again
        try:
            assert costs == (costs_from_lateness + costs_from_route_section_penalties), (
                f"experiment {experiment_id}: "
                f"costs_{scope}={costs}, "
                f"costs_from_lateness_{scope}={costs_from_lateness}, "
                f"costs_from_route_section_penalties_{scope}={costs_from_route_section_penalties} "
            )
        except AssertionError as e:
            rsp_logger.warn(str(e))
        assert costs >= experiment_results_analysis.costs_online_unrestricted
        assert costs >= experiment_results_analysis.malfunction_duration

    for scope in ["offline_fully_restricted", "offline_delta"]:
        costs = experiment_results_analysis._asdict()[f"costs_{scope}"]
        assert costs == experiment_results_analysis.costs_online_unrestricted
    for scope in ["online_route_restricted", "online_transmission_chains_fully_restricted", "online_transmission_chains_fully_restricted"] + [
        f"online_random_{i}" for i in range(GLOBAL_CONSTANTS.NB_RANDOM)
    ]:
        costs = experiment_results_analysis._asdict()[f"costs_{scope}"]

    assert experiment_results_analysis.costs_online_unrestricted >= experiment_results_analysis.malfunction_duration, (
        f"costs_online_unrestricted {experiment_results_analysis.costs_online_unrestricted} should be greater than malfunction duration, "
        f"{experiment_results_analysis.malfunction_duration} {experiment_results_analysis}"
    )


def convert_list_of_experiment_results_analysis_to_data_frame(l: List[ExperimentResultsAnalysis]) -> DataFrame:
    df = pd.DataFrame(columns=ExperimentResultsAnalysis._fields, data=[r._asdict() for r in l])
    df = df.select_dtypes(exclude=["object"])
    return df


def filter_experiment_results_analysis_data_frame(
    experiment_data: pd.DataFrame,
    min_time_online_unrestricted: int = 60,
    max_time_online_unrestricted_q: float = 0.97,
    max_time_online_unrestricted: int = 2000,
) -> pd.DataFrame:
    time_online_unrestricted = experiment_data["solver_statistics_times_total_online_unrestricted"]
    return experiment_data[
        (time_online_unrestricted >= min_time_online_unrestricted)
        & (time_online_unrestricted <= max_time_online_unrestricted)
        & (time_online_unrestricted <= time_online_unrestricted.quantile(max_time_online_unrestricted_q))
    ]


def expand_experiment_results_for_analysis(experiment_results: ExperimentResults) -> ExperimentResultsAnalysis:  # noqa: C901
    """

    Parameters
    ----------
    experiment_results:
        experiment_results to expand into to experiment_results_analysis
    Returns
    -------

    """

    if not isinstance(experiment_results, ExperimentResults):
        experiment_results_as_dict = dict(experiment_results[0])
        experiment_results = ExperimentResults(**experiment_results_as_dict)
    experiment_parameters: ExperimentParameters = experiment_results.experiment_parameters
    experiment_id = experiment_parameters.experiment_id

    # derive speed up
    nb_resource_conflicts_offline_delta = experiment_results.results_offline_delta.nb_conflicts
    nb_resource_conflicts_online_unrestricted = experiment_results.results_online_unrestricted.nb_conflicts
    # search space indiciators
    factor_resource_conflicts = -1
    try:
        factor_resource_conflicts = nb_resource_conflicts_offline_delta / nb_resource_conflicts_online_unrestricted
    except ZeroDivisionError:
        warnings.warn(f"no resource conflicts for experiment {experiment_id} -> set ratio to -1", stacklevel=2)

    nb_agents = experiment_parameters.infra_parameters.number_of_agents

    ground_truth_positive_changed_agents = {
        agent_id
        for agent_id, trainrun_online_unrestricted in experiment_results.results_online_unrestricted.trainruns_dict.items()
        if set(experiment_results.results_schedule.trainruns_dict[agent_id]) != set(trainrun_online_unrestricted)
    }

    ground_truth_negative_changed_agents = set(range(nb_agents)).difference(ground_truth_positive_changed_agents)

    def predicted_dict(predicted_changed_agents: Set[int], scope: str):
        predicted_changed_agents_false_positives = predicted_changed_agents.intersection(ground_truth_negative_changed_agents)
        predicted_changed_agents_false_negatives = (set(range(nb_agents)).difference(predicted_changed_agents)).intersection(
            ground_truth_positive_changed_agents
        )
        return {
            f"predicted_changed_agents_number_{scope}": len(predicted_changed_agents),
            f"predicted_changed_agents_false_positives_{scope}": len(predicted_changed_agents_false_positives),
            f"predicted_changed_agents_false_negatives_{scope}": len(predicted_changed_agents_false_negatives),
            f"predicted_changed_agents_percentage_{scope}": (len(predicted_changed_agents) / nb_agents),
            f"predicted_changed_agents_false_positives_percentage_{scope}": catch_zero_division_error_as_minus_one(
                lambda: len(predicted_changed_agents_false_positives) / len(ground_truth_negative_changed_agents)
            ),
            f"predicted_changed_agents_false_negatives_percentage_{scope}": catch_zero_division_error_as_minus_one(
                lambda: len(predicted_changed_agents_false_negatives) / len(ground_truth_positive_changed_agents)
            ),
            f"prediction_quality_{scope}": len(predicted_changed_agents_false_positives) + len(predicted_changed_agents_false_negatives),
        }

    # retain no fields from experiment results!
    d = dict()

    # extract predicted numbers
    d.update(
        **predicted_dict(experiment_results.predicted_changed_agents_online_transmission_chains_fully_restricted, "online_transmission_chains_fully_restricted")
    )
    d.update(
        **predicted_dict(experiment_results.predicted_changed_agents_online_transmission_chains_fully_restricted, "online_transmission_chains_route_restricted")
    )

    for i in range(GLOBAL_CONSTANTS.NB_RANDOM):
        d.update(**predicted_dict(experiment_results._asdict()[f"predicted_changed_agents_online_random_{i}"], f"online_random_{i}"))

    # extract other fields by configuration
    d.update(
        **extract_all_scopes_fields(experiment_results=experiment_results, all_scopes=all_scopes),
        **extract_rescheduling_scopes_fields(experiment_results=experiment_results, rescheduling_scopes=rescheduling_scopes),
    )
    for ratio_field, from_field in speedup_scopes_ratio_fields.items():
        for speed_up_scope in speed_up_scopes:
            try:
                d[f"{ratio_field}_{speed_up_scope}"] = d[f"{from_field}_online_unrestricted"] / d[f"{from_field}_{speed_up_scope}"]
            except ZeroDivisionError:
                if d[f"{from_field}_online_unrestricted"] == d[f"{from_field}_{speed_up_scope}"]:
                    d[f"{ratio_field}_{speed_up_scope}"] = 1.0
                else:
                    d[f"{ratio_field}_{speed_up_scope}"] = np.inf
    for additional_field in speedup_scopes_additional_fields:
        for speed_up_scope in speed_up_scopes:
            d[f"additional_{additional_field}_{speed_up_scope}"] = d[f"{additional_field}_{speed_up_scope}"] - d[f"{additional_field}_online_unrestricted"]

    # add online_random_average by averaging over online_random_{i}
    d = dict(
        **d,
        **{
            f"{prefix}_online_random_average": np.mean([d[f"{prefix}_online_random_{i}"] for i in range(GLOBAL_CONSTANTS.NB_RANDOM)])
            for prefix in online_random_average_fields
        },
    )

    # plausibility check with fields not nonified
    plausibility_check_experiment_results(experiment_results=experiment_results)
    plausibility_check_experiment_results_analysis(
        experiment_results_analysis=_to_experiment_results_analysis(
            d=d,
            experiment_parameters=experiment_results.experiment_parameters,
            problem_online_unrestricted=experiment_results.problem_online_unrestricted,
            malfunction=experiment_results.malfunction,
            results_schedule=experiment_results.results_schedule,
            factor_resource_conflicts=factor_resource_conflicts,
        )
    )

    return _to_experiment_results_analysis(
        d=d,
        experiment_parameters=experiment_results.experiment_parameters,
        problem_online_unrestricted=experiment_results.problem_online_unrestricted,
        malfunction=experiment_results.malfunction,
        results_schedule=experiment_results.results_schedule,
        factor_resource_conflicts=factor_resource_conflicts,
    )


def extract_rescheduling_scopes_fields(experiment_results: ExperimentResults, rescheduling_scopes: List[str]):
    return {
        f"{prefix}_{scope}": results_extractor(
            results_schedule=experiment_results._asdict()[f"results_schedule"],
            results_reschedule=experiment_results._asdict()[f"results_{scope}"],
            problem_reschedule=experiment_results._asdict()[f"problem_{scope}"],
        )
        for prefix, (_, results_extractor) in experiment_results_analysis_rescheduling_scopes_fields.items()
        for scope in rescheduling_scopes
    }


def extract_all_scopes_fields(experiment_results: ExperimentResults, all_scopes: List[str]):
    return {
        f"{prefix}_{scope}": results_extractor(experiment_results._asdict()[f"results_{scope}"], experiment_results._asdict()[f"problem_{scope}"])
        for prefix, (_, results_extractor) in experiment_results_analysis_all_scopes_fields.items()
        for scope in all_scopes
    }


def _to_experiment_results_analysis(
    d: dict,
    experiment_parameters: ExperimentParameters,
    results_schedule: ExperimentResults,
    malfunction: ExperimentMalfunction,
    problem_online_unrestricted: ScheduleProblemDescription,
    factor_resource_conflicts: int,
):
    return ExperimentResultsAnalysis(
        **extract_base_fields(
            experiment_parameters=experiment_parameters,
            results_schedule=results_schedule,
            malfunction=malfunction,
            problem_online_unrestricted=problem_online_unrestricted,
        ),
        **d,
        factor_resource_conflicts=factor_resource_conflicts,
    )


def extract_base_fields(
    experiment_parameters: ExperimentParameters,
    results_schedule: ExperimentResults,
    malfunction: ExperimentMalfunction,
    problem_online_unrestricted: ScheduleProblemDescription,
):
    return dict(
        experiment_id=experiment_parameters.experiment_id,
        grid_id=experiment_parameters.grid_id,
        size=experiment_parameters.infra_parameters.width,
        n_agents=experiment_parameters.infra_parameters.number_of_agents,
        n_agents_running=len(
            [agent_id for agent_id, schedule_trainrun in results_schedule.trainruns_dict.items() if schedule_trainrun[-1].scheduled_at >= malfunction.time_step]
        ),
        rescheduling_horizon=problem_online_unrestricted.max_episode_steps + malfunction.malfunction_duration - malfunction.time_step,
        max_num_cities=experiment_parameters.infra_parameters.max_num_cities,
        max_rail_between_cities=experiment_parameters.infra_parameters.max_rail_between_cities,
        max_rail_in_city=experiment_parameters.infra_parameters.max_rail_in_city,
        infra_id=experiment_parameters.infra_parameters.infra_id,
        schedule_id=experiment_parameters.schedule_parameters.schedule_id,
        infra_id_schedule_id=experiment_parameters.infra_id_schedule_id,
        earliest_malfunction=experiment_parameters.re_schedule_parameters.earliest_malfunction,
        malfunction_duration=experiment_parameters.re_schedule_parameters.malfunction_duration,
        malfunction_agent_id=experiment_parameters.re_schedule_parameters.malfunction_agent_id,
        weight_route_change=experiment_parameters.re_schedule_parameters.weight_route_change,
        weight_lateness_seconds=experiment_parameters.re_schedule_parameters.weight_lateness_seconds,
        max_window_size_from_earliest=experiment_parameters.re_schedule_parameters.max_window_size_from_earliest,
    )
