import pprint

from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.utils.data_types import convert_data_frame_row_to_experiment_results
from rsp.utils.data_types import expand_experiment_results_for_analysis
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import extract_path_search_space

_pp = pprint.PrettyPrinter(indent=4)


def _analyze_times(experiment_results: ExperimentResults, experiment_results_analysis: ExperimentResultsAnalysis):
    time_delta_after_m = experiment_results_analysis.time_delta_after_malfunction
    time_full_after_m = experiment_results_analysis.time_full_after_malfunction
    # Delta is all train run way points in the re-schedule that are not also in the schedule
    schedule_trainrunwaypoints = experiment_results_analysis.solution_full
    full_reschedule_trainrunwaypoints_dict = experiment_results_analysis.solution_full_after_malfunction
    schedule_full_reschedule_delta: TrainrunDict = {
        agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).difference(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at)
        for agent_id in schedule_trainrunwaypoints.keys()
    }
    schedule_full_reschedule_delta_percentage = \
        100 * sum([len(schedule_full_reschedule_delta[agent_id])
                   for agent_id in schedule_full_reschedule_delta.keys()]) / \
        sum([len(full_reschedule_trainrunwaypoints_dict[agent_id])
             for agent_id in full_reschedule_trainrunwaypoints_dict.keys()])
    # Freeze is all train run way points in the schedule that are also in the re-schedule
    schedule_full_reschedule_freeze: TrainrunDict = \
        {agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).intersection(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at) for agent_id in schedule_full_reschedule_delta.keys()}
    schedule_full_reschedule_freeze_percentage = 100 * sum(
        [len(schedule_full_reschedule_freeze[agent_id]) for agent_id in schedule_full_reschedule_freeze.keys()]) / sum(
        [len(schedule_trainrunwaypoints[agent_id]) for agent_id in schedule_trainrunwaypoints.keys()])

    # TODO SIM-151 do we need absolute counts as well as below?
    print(
        f"**** full schedule -> full re-schedule: {schedule_full_reschedule_freeze_percentage}%"
        " of trainrun waypoints in the full schedule stay the same in the full re-schedule")
    print(
        f"**** full schedule -> full re-schedule: {schedule_full_reschedule_delta_percentage}% "
        "of trainrun waypoints in the full re-schedule are different from the initial full schedule")
    all_full_reschedule_trainrunwaypoints = {
        full_reschedule_trainrunwaypoint
        for full_reschedule_trainrunwaypoints in full_reschedule_trainrunwaypoints_dict.values()
        for full_reschedule_trainrunwaypoint in full_reschedule_trainrunwaypoints
    }
    all_delta_reschedule_trainrunwaypoints = {
        full_reschedule_trainrunwaypoint
        for full_reschedule_trainrunwaypoints in
        experiment_results_analysis.solution_full_after_malfunction.values()
        for full_reschedule_trainrunwaypoint in full_reschedule_trainrunwaypoints
    }
    full_delta_same_counts = len(
        all_full_reschedule_trainrunwaypoints.intersection(all_delta_reschedule_trainrunwaypoints))
    full_delta_same_percentage = 100 * full_delta_same_counts / len(all_full_reschedule_trainrunwaypoints)
    full_delta_new_counts = len(
        all_delta_reschedule_trainrunwaypoints.difference(all_full_reschedule_trainrunwaypoints))
    full_delta_stale_counts = len(
        all_full_reschedule_trainrunwaypoints.difference(all_delta_reschedule_trainrunwaypoints))
    print(
        f"**** full re-schedule -> delta re-schedule: "
        f"same {full_delta_same_percentage}% ({full_delta_same_counts})"
        f"(+{full_delta_new_counts}, -{full_delta_stale_counts}) waypoints")
    time_rescheduling_speedup_factor = time_full_after_m / time_delta_after_m
    print(f"**** full re-schedule -> delta re-schedule: "
          f"time speed-up factor {time_rescheduling_speedup_factor:4.1f} "
          f"{time_full_after_m}s -> {time_delta_after_m}s")


def _analyze_paths(experiment_results: ExperimentResults,
                   experiment_results_analysis: ExperimentResultsAnalysis,
                   experiment_id: int):
    _rsp_delta, _rsp_full, _schedule = extract_path_search_space(
        experiment_results=experiment_results)
    print(f"**** (experiment {experiment_id}) path search space: "
          f"path_search_space_schedule={_schedule:.2E}, "
          f"path_search_space_rsp_full={_rsp_full:.2E}, "
          f"path_search_space_rsp_delta={_rsp_delta:.2E}")
    resource_conflicts_search_space_schedule = experiment_results_analysis.nb_resource_conflicts_full
    resource_conflicts_search_space_rsp_full = experiment_results_analysis.nb_resource_conflicts_full_after_malfunction
    resource_conflicts_search_space_rsp_delta = experiment_results_analysis.nb_resource_conflicts_delta_after_malfunction
    print(f"**** (experiment {experiment_id}) resource conflicts search space: "
          f"resource_conflicts_search_space_schedule={resource_conflicts_search_space_schedule :.2E}, "
          f"resource_conflicts_search_space_rsp_full={resource_conflicts_search_space_rsp_full :.2E}, "
          f"resource_conflicts_search_space_rsp_delta={resource_conflicts_search_space_rsp_delta :.2E}")


def analyze_experiment(experiment: ExperimentParameters,
                       data_frame: DataFrame):
    # find first row for this experiment (iloc[0]
    rows = data_frame.loc[data_frame['experiment_id'] == experiment.experiment_id]

    experiment_results: ExperimentResults = convert_data_frame_row_to_experiment_results(rows)

    experiment_results_analysis = expand_experiment_results_for_analysis(
        experiment_id=experiment.experiment_id,
        experiment_results=experiment_results)
    _analyze_times(experiment_results=experiment_results, experiment_results_analysis=experiment_results_analysis)
    _analyze_paths(experiment_results=experiment_results,
                   experiment_results_analysis=experiment_results_analysis,
                   experiment_id=experiment.experiment_id)
