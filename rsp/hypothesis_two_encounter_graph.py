import pprint
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.compute_time_analysis.compute_time_analysis import _get_difference_in_time_space
from rsp.compute_time_analysis.compute_time_analysis import plot_time_resource_data
from rsp.compute_time_analysis.compute_time_analysis import resource_time_2d
from rsp.encounter_graph.encounter_graph import symmetric_distance_between_trains_dummy_Euclidean
from rsp.encounter_graph.encounter_graph import symmetric_distance_between_trains_sum_of_time_window_overlaps
from rsp.encounter_graph.encounter_graph import symmetric_temporal_distance_between_trains
from rsp.encounter_graph.encounter_graph_visualization import _plot_encounter_graph_directed
from rsp.encounter_graph.encounter_graph_visualization import plot_encounter_graphs_for_experiment_result
from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder
from rsp.utils.flatland_replay_utils import convert_trainrundict_to_positions_after_flatland_timestep

_pp = pprint.PrettyPrinter(indent=4)

LeftClosedInterval = NamedTuple('LeftClosedInterval', [
    ('from_incl', int),
    ('to_excl', int)])
Resource = NamedTuple('Resource', [
    ('row', int),
    ('column', int)])
ResourceOccupation = NamedTuple('ResourceOccupation', [
    ('interval', LeftClosedInterval),
    ('resource', Resource),
    ('agent_id', int)])
RELEASE_TIME = 1
SortedResourceOccupationsPerResourceDict = Dict[Resource, List[ResourceOccupation]]
SortedResourceOccupationsPerAgentDict = Dict[int, List[ResourceOccupation]]
ResourceSorting = Dict[Resource, int]
Trajectories = List[List[Tuple[int, int]]]


def hypothesis_two_encounter_graph_directed(
        experiment_base_directory: str,
        experiment_ids: List[int] = None,
        width: int = 400
):
    experiment_analysis_directory = f'{experiment_base_directory}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}/'

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_data_directory,
        experiment_ids=experiment_ids)
    experiment_data: pd.DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)

    for i in list(range(len(experiment_results_list))):
        experiment_output_folder = f"{experiment_analysis_directory}/experiment_{experiment_ids[i]:04d}_analysis"
        encounter_graph_folder = f"{experiment_output_folder}/encounter_graphs"

        # Check and create the folders
        check_create_folder(experiment_output_folder)
        check_create_folder(encounter_graph_folder)

        experiment_result: ExperimentResultsAnalysis = experiment_results_list[i]

        # Get full schedule Time-resource-Data
        schedule = experiment_result.results_full.trainruns_dict
        malfunction = experiment_result.malfunction
        malfunction_agent_id = malfunction.agent_id
        reschedule = experiment_result.results_full_after_malfunction.trainruns_dict
        number_of_trains = len(schedule)

        # tweak the trainrun_dict: merge the initial synchronization step of length 1 in the first cell with the remainder in the cell
        # for agent_id, trainrun in schedule.items():
        #     trainrun.pop(1)
        # for agent_id, trainrun in reschedule.items():
        #     trainrun.pop(1)

        # 0. data preparation
        rolled_out_schedule = convert_trainrundict_to_positions_after_flatland_timestep(schedule)
        rolled_out_reschedule = convert_trainrundict_to_positions_after_flatland_timestep(reschedule)

        # aggregate occupations of each resource
        schedule_resource_occupations_per_resource, schedule_resource_occupations_per_agent = extract_resource_occupations(schedule)
        reschedule_resource_occupations_per_resource, reschedule_resource_occupations_per_agent = extract_resource_occupations(reschedule)

        # 1. compute the forward-only wave of the malfunction
        # TODO does not work yet!
        # - forward-only: only agents running at or after the wave hitting are considere, i.e. agents do not decelerate ahed of the wave!

        resource_reached_at: Dict[Resource, int] = {}
        resource_reached_by: Dict[Resource, int] = {}

        queue: List[Tuple[Resource, int]] = []
        queue.append((rolled_out_schedule[malfunction_agent_id][malfunction.time_step].position, malfunction.time_step))
        while len(queue) > 0:
            wave_front = queue.pop()
            # print(wave_front)
            wave_front_resource, wave_front_time = wave_front

            if wave_front_resource in resource_reached_at and wave_front_time >= resource_reached_at[wave_front_resource]:
                # already dealt with
                # non-optimized: resource already reached, we might re-enque impacted trains!
                continue

            resource_reached_at.setdefault(wave_front_resource, wave_front_time)
            resource_reached_at[wave_front_resource] = min(resource_reached_at[wave_front_resource], wave_front_time)
            impacted_trains = [
                (resource_occupation.agent_id, resource_occupation.interval.from_incl)
                for resource_occupation in schedule_resource_occupations_per_resource[wave_front_resource]
                # TODO this is forward_only, trains only wait if wave hits them during their run
                if (resource_occupation.interval.to_excl >= wave_front_time)
            ]
            # print(impacted_trains)

            for agent_id, time_from in impacted_trains:
                # print(agent_id)
                rolled_out_schedule_agent = rolled_out_schedule[agent_id]
                for t in range(time_from, len(rolled_out_schedule_agent)):
                    if rolled_out_schedule_agent[t] is not None:
                        queue.append((rolled_out_schedule_agent[t].position, t))

        # 2. visualize the wave in resource-time diagram
        # hack wave as additional agent
        wave_agent_id = number_of_trains
        fake_resource_occupations = [
            ResourceOccupation(interval=LeftClosedInterval(from_incl=reached_at, to_excl=reached_at + 1),
                               resource=resource,
                               agent_id=wave_agent_id)
            for resource, reached_at in resource_reached_at.items()
        ]
        schedule_resource_occupations_per_agent[wave_agent_id] = fake_resource_occupations
        reschedule_resource_occupations_per_agent[wave_agent_id] = fake_resource_occupations

        # sort and plot
        _, resource_sorting = resource_time_2d(schedule=schedule,
                                               width=width,
                                               malfunction_agent_id=malfunction_agent_id,
                                               sorting=None)
        _plot_resource_time_diagram(malfunction=malfunction,
                                    nb_agents=number_of_trains,
                                    resource_sorting=resource_sorting,
                                    resource_occupations_schedule=schedule_resource_occupations_per_agent,
                                    resource_occupations_reschedule=reschedule_resource_occupations_per_agent,
                                    width=width)

        # 3. non-symmetric distance matrix of primary, secondary etc. effects
        # TODO take only encounters after wave reaches the resource
        # non-symmetric distance_matrix: insert distance between two consecutive trains at a resource; if no direct encounter, distance is inf
        distance_matrix = np.zeros((number_of_trains, number_of_trains))
        distance_matrix.fill(np.inf)

        debug_info = {}

        for resource, occupations in schedule_resource_occupations_per_resource.items():
            for leaving_train_occupation, entering_train_occupation in zip(occupations, occupations[1:]):
                # skip exit events before malfunction starts (cannot have any influence any more)
                # TODO we could even deduce RELEASE_TIME, since the agent is already in the next cell?
                if leaving_train_occupation.interval.to_excl <= malfunction.time_step:
                    continue
                distance = entering_train_occupation.interval.from_incl - leaving_train_occupation.interval.to_excl
                if distance < 0:
                    print(f"{resource} {leaving_train_occupation} - {entering_train_occupation}")
                    print(f"{resource}: {occupations}")
                    print(f"leaving train {leaving_train_occupation.agent_id}: {_pp.pformat(schedule[leaving_train_occupation.agent_id])}")
                    print(f"leaving train ({entering_train_occupation.agent_id}): {_pp.pformat(schedule[entering_train_occupation.agent_id])}")
                assert distance >= 0
                distance_matrix[leaving_train_occupation.agent_id, entering_train_occupation.agent_id] = min(distance, distance_matrix[
                    leaving_train_occupation.agent_id, entering_train_occupation.agent_id])
                debug_info.setdefault((leaving_train_occupation.agent_id, entering_train_occupation.agent_id), []).append(
                    (leaving_train_occupation, entering_train_occupation))

        weights_matrix = 1 / (distance_matrix + 0.000001)
        np_max = np.max(weights_matrix)
        print(np_max)
        weights_matrix /= np_max
        print(weights_matrix)


        # 4. visualize
        # TODO
        pos = {}
        pos[malfunction.agent_id] = (0, 0)
        agents_at_depth = {malfunction.agent_id}
        depth = 1
        done_at_depth = set()
        edges = set()
        while len(agents_at_depth) > 0:

            done_at_depth.update(agents_at_depth)

            # TODO replace by time!
            # TODO filter out edges not dependent! How to do that exactly?
            round_index = 0
            next_round = set()
            for agent_id in agents_at_depth:
                for neighbour in range(number_of_trains):
                    if neighbour in done_at_depth:
                        continue
                    weight = weights_matrix[agent_id, neighbour]
                    if weight > 0:
                        pos[neighbour] = (round_index * 5, depth)
                        round_index += 1
                        next_round.add(neighbour)
            print(f"depth {depth}: {next_round}")
            print(pos)

            agents_at_depth = next_round
            depth += 1

        # TODO visualiz removed ones
        staengeli_index = 0
        for agent_id in range(number_of_trains):
            if agent_id not in pos:
                pos[agent_id] = (staengeli_index * 2, depth + 5)
                staengeli_index += 1
        _debug_directed((6, 2), debug_info, distance_matrix, weights_matrix)
        _plot_encounter_graph_directed(
            weights_matrix=weights_matrix,
            title=f"Encounter Graph for experiment {experiment_result.experiment_id}, {malfunction}",
            # file_name=f"encounter_graph_{experiment_result.experiment_id:04d}.pdf",
            pos=pos)


def extract_resource_occupations(schedule: TrainrunDict) -> Tuple[SortedResourceOccupationsPerResourceDict, SortedResourceOccupationsPerAgentDict]:
    # TODO refactor rolloing out resource occupations
    rolled_out_resource_occupations: Dict[Tuple[Resource, int], int] = {}
    resource_occupations_per_resource: SortedResourceOccupationsPerResourceDict = {}
    resource_occupations_per_agent: SortedResourceOccupationsPerAgentDict = {}
    for agent_id, trainrun in schedule.items():
        resource_occupations_per_agent[agent_id] = []
        for entry_event, exit_event in zip(trainrun, trainrun[1:]):
            resource = entry_event.waypoint.position
            from_incl = entry_event.scheduled_at
            to_excl = exit_event.scheduled_at + RELEASE_TIME
            ro = ResourceOccupation(interval=LeftClosedInterval(from_incl, to_excl), agent_id=agent_id, resource=resource)
            resource_occupations_per_resource.setdefault(resource, []).append(ro)
            resource_occupations_per_agent[agent_id].append(ro)
            for t in range(from_incl, to_excl):
                rolled_out_resource_occupations[(resource, t)] = agent_id
    # sort occupations by interval lower bound
    for resource, occupations in resource_occupations_per_resource.items():
        occupations.sort(key=lambda ro: ro.interval.from_incl)
    return resource_occupations_per_resource, resource_occupations_per_agent


def _plot_resource_time_diagram(malfunction: ExperimentMalfunction,
                                resource_sorting: ResourceSorting,
                                nb_agents: int,
                                resource_occupations_schedule: SortedResourceOccupationsPerAgentDict,
                                resource_occupations_reschedule: SortedResourceOccupationsPerAgentDict,
                                width: int):
    malfunction_agent_id = malfunction.agent_id
    # TODO extract sorting from  resource_time_2d
    schedule_trajectories = trajectories_from_resource_occupations_per_agent(resource_occupations_schedule, resource_sorting, width)
    reschedule_trajectories = trajectories_from_resource_occupations_per_agent(resource_occupations_reschedule, resource_sorting, width)

    # # Get full reschedule Time-resource-Data
    # time_resource_reschedule_full, _ = resource_time_2d(schedule=reschedule,
    #                                                     width=width,
    #                                                     malfunction_agent_id=malfunction_agent_id,
    #                                                     sorting=ressource_sorting)
    # Plotting the graphs
    ranges = (len(resource_sorting),
              max(max([ro[-1].interval.to_excl for ro in resource_occupations_schedule.values()]),
                  max([ro[-1].interval.to_excl for ro in resource_occupations_reschedule.values()])))
    # Plot Schedule
    plot_time_resource_data(trajectories=schedule_trajectories, title='Schedule', ranges=ranges)
    # Plot Reschedule Full
    plot_time_resource_data(trajectories=reschedule_trajectories, title='Full Reschedule', ranges=ranges)
    # Compute the difference between schedules and return traces for plotting
    traces_influenced_agents, changed_agents_list = _get_difference_in_time_space(
        time_resource_matrix_a=schedule_trajectories,
        time_resource_matrix_b=reschedule_trajectories)
    # Printing situation overview
    print(
        "Agent nr.{} has a malfunction at time {} for {} s and influenced {} other agents of {}. Total delay = {}.".format(
            malfunction_agent_id, malfunction.time_step,
            malfunction.malfunction_duration,
            len(traces_influenced_agents),
            nb_agents,
            "TODO"))
    # Plot difference
    plot_time_resource_data(trajectories=traces_influenced_agents, title='Changed Agents',
                            ranges=ranges)


def trajectories_from_resource_occupations_per_agent(resource_occupations_schedule: SortedResourceOccupationsPerAgentDict,
                                                     resource_sorting: ResourceSorting,
                                                     width: int) -> Trajectories:
    schedule_trajectories: Trajectories = []
    print(resource_sorting)
    for agent_id, resource_ocupations in resource_occupations_schedule.items():
        train_time_path = []
        for resource_ocupation in resource_ocupations:
            print(f" {agent_id} {resource_ocupation}")
            position = coordinate_to_position(width, [resource_ocupation.resource])[0]
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.from_incl))
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.to_excl))
            train_time_path.append((None, None))
        schedule_trajectories.append(train_time_path)
    return schedule_trajectories


def _debug_directed(pair, debug_info, distance_matrix, weights_matrix):
    print(f"debug_info[{pair}]={debug_info[pair]}")
    # print(f"debug_info[(2,6)]={debug_info[(2,6)]}")
    print(f"weights_matrix[{pair}]={weights_matrix[pair]}")
    print(f"distance_matrix[{pair}]={distance_matrix[pair]}")


def hypothesis_two_encounter_graph_undirected(
        experiment_base_directory: str,
        experiment_ids: List[int] = None,
        debug_pair: Optional[Tuple[int, int]] = None
):
    """This method computes the encounter graphs of the specified experiments.
    Within this first approach, the distance measure within the encounter
    graphs is undirected.

    Parameters
    ----------
    experiment_base_directory
    experiment_ids
    debug_pair
    """
    experiment_analysis_directory = f'{experiment_base_directory}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}/'

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_data_directory,
        experiment_ids=experiment_ids)

    for i in list(range(len(experiment_results_list))):
        experiment_output_folder = f"{experiment_analysis_directory}/experiment_{experiment_ids[i]:04d}_analysis"
        encounter_graph_folder = f"{experiment_output_folder}/encounter_graphs"

        # Check and create the folders
        check_create_folder(experiment_output_folder)
        check_create_folder(encounter_graph_folder)

        experiment_result = experiment_results_list[i]

        for metric_function in [
            symmetric_distance_between_trains_dummy_Euclidean,
            symmetric_distance_between_trains_sum_of_time_window_overlaps,
            symmetric_temporal_distance_between_trains
        ]:
            plot_encounter_graphs_for_experiment_result(
                experiment_result=experiment_result,
                encounter_graph_folder=encounter_graph_folder,
                metric_function=metric_function,
                debug_pair=debug_pair
            )


if __name__ == '__main__':
    hypothesis_two_encounter_graph_directed(
        experiment_base_directory='./res/exp_hypothesis_one_2020_03_31T07_11_03',
        experiment_ids=[79],
    )
