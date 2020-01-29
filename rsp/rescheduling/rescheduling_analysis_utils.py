import pprint
from functools import reduce
from operator import mul
from typing import List

from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.utils.data_types import AgentsPathsDict
from rsp.utils.data_types import convert_data_frame_row_to_experiment_results
from rsp.utils.data_types import ExperimentFreezeDict
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.route_graph_analysis import get_number_of_paths_for_experiment_freeze
from rsp.utils.route_graph_analysis import visualize_experiment_freeze

_pp = pprint.PrettyPrinter(indent=4)


# TODO SIM-151: use in plots
def _analyze_times(current_results: ExperimentResults):
    time_delta_after_m = current_results.time_delta_after_malfunction
    time_full_after_m = current_results.time_full_after_malfunction
    # Delta is all train run way points in the re-schedule that are not also in the schedule
    schedule_trainrunwaypoints = current_results.solution_full
    full_reschedule_trainrunwaypoints_dict = current_results.solution_full_after_malfunction
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
        " of waypoints in the full schedule stay the same in the full re-schedule")
    print(
        f"**** full schedule -> full re-schedule: {schedule_full_reschedule_delta_percentage}% "
        "of waypoints in the full re-schedule are different from the initial full schedule")
    all_full_reschedule_trainrunwaypoints = {
        full_reschedule_trainrunwaypoint
        for full_reschedule_trainrunwaypoints in full_reschedule_trainrunwaypoints_dict.values()
        for full_reschedule_trainrunwaypoint in full_reschedule_trainrunwaypoints
    }
    all_delta_reschedule_trainrunwaypoints = {
        full_reschedule_trainrunwaypoint
        for full_reschedule_trainrunwaypoints in current_results.solution_delta_after_malfunction.values()
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


# numpy produces overflow -> python ints are unbounded, see https://stackoverflow.com/questions/2104782/returning-the-product-of-a-list
def _prod(l: List[int]):
    return reduce(mul, l, 1)


# TODO SIM-151: use in plots instead of only printing
def _analyze_paths(experiment_results: ExperimentResults, debug: bool = False):
    experiment_freeze_delta_afer_malfunction = experiment_results.experiment_freeze_delta_after_malfunction
    experiment_freeze_full_after_malfunction = experiment_results.experiment_freeze_full_after_malfunction
    all_nb_alternatives_schedule = []
    all_nb_alternatives_rsp_full = []
    all_nb_alternatives_rsp_delta = []

    agents_paths_dict = experiment_results.agents_paths_dict
    for agent_id in experiment_freeze_delta_afer_malfunction:
        agent_paths = agents_paths_dict[agent_id]
        # TODO SIM-239 this is not correct, there may not be enough time for all paths; use experiment_freeze for scheduling
        nb_alternatives_schedule = get_number_of_paths_for_experiment_freeze(
            agent_paths,
            None
        )
        nb_alternatives_rsp_full = get_number_of_paths_for_experiment_freeze(
            agent_paths,
            experiment_freeze_full_after_malfunction[agent_id]
        )
        nb_alternatives_rsp_delta = get_number_of_paths_for_experiment_freeze(
            agent_paths,
            experiment_freeze_delta_afer_malfunction[agent_id]
        )
        assert nb_alternatives_schedule >= nb_alternatives_rsp_full
        assert nb_alternatives_rsp_full >= nb_alternatives_rsp_delta

        all_nb_alternatives_schedule.append(nb_alternatives_schedule)
        all_nb_alternatives_rsp_full.append(nb_alternatives_rsp_full)
        all_nb_alternatives_rsp_delta.append(nb_alternatives_rsp_delta)
        if debug:
            print(f"  agent {agent_id}: "
                  f"nb_alternatives_schedule={nb_alternatives_schedule}, "
                  f"nb_alternatives_rsp_full={nb_alternatives_rsp_full}",
                  f"nb_alternatives_rsp_delta={nb_alternatives_rsp_delta}, "
                  )
    assert len(all_nb_alternatives_schedule) == len(agents_paths_dict.keys())
    assert len(all_nb_alternatives_rsp_full) == len(agents_paths_dict.keys())
    assert len(all_nb_alternatives_rsp_delta) == len(agents_paths_dict.keys())
    search_space_space_schedule = _prod(all_nb_alternatives_schedule)
    search_space_space_rsp_full = _prod(all_nb_alternatives_rsp_full)
    search_space_space_rsp_delta = _prod(all_nb_alternatives_rsp_delta)
    assert search_space_space_schedule >= search_space_space_rsp_full
    assert search_space_space_rsp_full >= search_space_space_rsp_delta
    # TODO SIM-252 plot analysis
    print("**** search space"
          f"search_space_space_schedule={search_space_space_schedule:.2E}, "
          f"search_space_space_rsp_full={search_space_space_rsp_full:.2E}",
          f"search_space_space_rsp_delta={search_space_space_rsp_delta:.2E}, "
          )


def analyze_experiment(experiment: ExperimentParameters,
                       data_frame: DataFrame):
    # find first row for this experiment (iloc[0]
    rows = data_frame.loc[data_frame['experiment_id'] == experiment.experiment_id]
    print('malfunction_time_step')
    print(rows['malfunction_time_step'].iloc[0])
    n_agents: int = int(rows['n_agents'].iloc[0])

    experiment_results: ExperimentResults = convert_data_frame_row_to_experiment_results(rows)
    train_runs_input: TrainrunDict = experiment_results.solution_full
    train_runs_full_after_malfunction: TrainrunDict = experiment_results.solution_full_after_malfunction
    train_runs_delta_after_malfunction: TrainrunDict = experiment_results.solution_delta_after_malfunction
    experiment_freeze_delta_afer_malfunction: ExperimentFreezeDict = experiment_results.experiment_freeze_delta_after_malfunction
    malfunction: ExperimentMalfunction = experiment_results.malfunction
    agents_paths_dict: AgentsPathsDict = experiment_results.agents_paths_dict

    _analyze_times(experiment_results)
    _analyze_paths(experiment_results)

    for agent_id in experiment_freeze_delta_afer_malfunction:
        visualize_experiment_freeze(
            agent_paths=agents_paths_dict[agent_id],
            train_run_input=train_runs_input[agent_id],
            train_run_full_after_malfunction=train_runs_full_after_malfunction[agent_id],
            train_run_delta_after_malfunction=train_runs_delta_after_malfunction[agent_id],
            f=experiment_freeze_delta_afer_malfunction[agent_id],
            title=f"experiment {experiment.experiment_id}\nagent {agent_id}/{n_agents}\n{malfunction}"
        )
