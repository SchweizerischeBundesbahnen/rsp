import pprint
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
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
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder

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
TransmissionLeg = NamedTuple('TransmissionLeg', [
    ('hop_on', ResourceOccupation),
    ('hop_off', ResourceOccupation)
])
TransmissionChain = List[TransmissionLeg]


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

    for i in list(range(len(experiment_results_list))):
        experiment_output_folder = f"{experiment_analysis_directory}/experiment_{experiment_ids[i]:04d}_analysis"
        encounter_graph_folder = f"{experiment_output_folder}/encounter_graphs"

        # Check and create the folders
        check_create_folder(experiment_output_folder)
        check_create_folder(encounter_graph_folder)

        experiment_result: ExperimentResultsAnalysis = experiment_results_list[i]

        disturbance_propagation_graph_visualization(experiment_result, width)


def disturbance_propagation_graph_visualization(experiment_result: ExperimentResultsAnalysis, width: int = 400):
    # Get full schedule Time-resource-Data
    schedule = experiment_result.results_full.trainruns_dict
    malfunction = experiment_result.malfunction
    malfunction_agent_id = malfunction.agent_id
    reschedule = experiment_result.results_full_after_malfunction.trainruns_dict
    number_of_trains = len(schedule)
    max_time_schedule = np.max([trainrun[-1].scheduled_at for trainrun in schedule.values()])
    # 0. data preparation
    # aggregate occupations of each resource
    schedule_resource_occupations_per_resource, schedule_resource_occupations_per_agent = extract_resource_occupations(schedule)
    reschedule_resource_occupations_per_resource, reschedule_resource_occupations_per_agent = extract_resource_occupations(reschedule)
    for resource, occupations in schedule_resource_occupations_per_resource.items():
        for ro in occupations:
            assert ro.resource == resource
    for agent_id, occupations in schedule_resource_occupations_per_agent.items():
        for ro in occupations:
            assert ro.agent_id == agent_id
    # 1. compute the forward-only wave of the malfunction
    # "forward-only" means only agents running at or after the wave hitting them are considered,
    # i.e. agents do not decelerate ahead of the wave!
    open_wave_front: List[Tuple[ResourceOccupation, TransmissionChain]] = []

    transmission_chains: List[TransmissionChain] = []
    closed_wave_front: List[ResourceOccupation] = []
    malfunction_occupation = next(ro for ro in schedule_resource_occupations_per_agent[malfunction_agent_id] if malfunction.time_step < ro.interval.to_excl)
    for ro in schedule_resource_occupations_per_agent[malfunction_agent_id]:
        # N.B. intervals include release times, therefore we can be strict at upper bound!
        if malfunction.time_step < ro.interval.to_excl:
            # TODO should the interval extended by the malfunction duration?
            open_wave_front.append((ro, [TransmissionLeg(malfunction_occupation, ro)]))
    assert len(open_wave_front) > 0
    while len(open_wave_front) > 0:
        wave_front, history = open_wave_front.pop()
        wave_front_resource = wave_front.resource
        if wave_front in closed_wave_front:
            continue
        closed_wave_front.append(wave_front)

        # the next scheduled train may be impacted!
        # TODO we do not consider decelerate to let pass yet! search backward in radius? would this be another type of chain?
        for ro in schedule_resource_occupations_per_resource[wave_front_resource]:
            if ro.interval.from_incl >= wave_front.interval.to_excl:
                for subsequent_ro in schedule_resource_occupations_per_agent[ro.agent_id]:
                    # hop_on and hop_off may be at the same resource
                    if subsequent_ro.interval.from_incl >= ro.interval.from_incl:
                        chain = history + [TransmissionLeg(ro, subsequent_ro)]
                        el = (subsequent_ro, chain)
                        assert subsequent_ro not in history
                        if ro in closed_wave_front:
                            continue
                        open_wave_front.append(el)
                        transmission_chains.append(chain)
                break
    for transmission_chain in transmission_chains:
        for transmission in transmission_chain:
            assert transmission.hop_on.agent_id == transmission.hop_off.agent_id, transmission
            # N.B: hop_on and hop_off may be at the same resource occupation!
            assert transmission.hop_off.interval.from_incl >= transmission.hop_on.interval.from_incl, transmission
            assert transmission.hop_off.interval.to_excl >= transmission.hop_on.interval.to_excl, transmission
        for tr1, tr2 in zip(transmission_chain, transmission_chain[1:]):
            assert tr1.hop_off.agent_id != tr2.hop_off.agent_id, (tr1, tr2)
            # transmission via a resource
            assert tr2.hop_on.resource == tr1.hop_off.resource, (tr1, tr2)
            assert tr2.hop_on.interval.from_incl >= tr1.hop_off.interval.to_excl, (tr1, tr2)

    # 2. visualize the wave in resource-time diagram
    # TODO dirty hack: visualize wave as additional agent
    wave_agent_id = number_of_trains
    fake_resource_occupations = [
        ResourceOccupation(interval=ro.interval,
                           resource=ro.resource,
                           agent_id=wave_agent_id)
        for ro in closed_wave_front
    ]
    schedule_resource_occupations_per_agent[wave_agent_id] = fake_resource_occupations
    reschedule_resource_occupations_per_agent[wave_agent_id] = []
    # sort and plot
    _, resource_sorting = resource_time_2d(schedule=schedule,
                                           width=width,
                                           malfunction_agent_id=malfunction_agent_id,
                                           sorting=None)
    changed_agents: Dict[int, bool] = _plot_resource_time_diagram(
        malfunction=malfunction,
        nb_agents=number_of_trains,
        resource_sorting=resource_sorting,
        resource_occupations_schedule=schedule_resource_occupations_per_agent,
        resource_occupations_reschedule=reschedule_resource_occupations_per_agent,
        width=width)
    # 3. non-symmetric distance matrix of primary, secondary etc. effects
    # non-symmetric distance_matrix: insert distance between two consecutive trains at a resource; if no direct encounter, distance is inf
    distance_matrix = np.zeros((number_of_trains, number_of_trains))
    distance_first_reaching = np.zeros((number_of_trains, number_of_trains))
    distance_matrix.fill(np.inf)
    distance_first_reaching.fill(np.inf)
    debug_info = {}
    hop_on_at_depth = {agent_id: {} for agent_id in range(number_of_trains)}
    minimal_depth = {}
    for transmission_chain in transmission_chains:
        if False:
            # TODO weighting/damping: take into account number of intermediate steps and their distance!
            # cutoff at 2 * distance
            distances = [to_leg.hop_on.interval.from_incl - from_leg.hop_off.interval.to_excl for from_leg, to_leg in
                         zip(transmission_chain, transmission_chain[1:])]
            for distance in distances:
                assert distance >= 0
            max_distance = np.max(distances)
            if max_distance >= malfunction.malfunction_duration:
                continue
        from_ro = transmission_chain[-2].hop_off
        to_ro = transmission_chain[-1].hop_on
        hop_on_depth = len(transmission_chain) - 1
        hop_on_at_depth[to_ro.agent_id].setdefault(hop_on_depth, []).append(to_ro)
        minimal_depth.setdefault(to_ro.agent_id, hop_on_depth)
        minimal_depth[to_ro.agent_id] = min(minimal_depth[to_ro.agent_id], hop_on_depth)

        distance = to_ro.interval.from_incl - from_ro.interval.to_excl
        assert distance >= 0
        from_agent_id = from_ro.agent_id
        to_agent_id = to_ro.agent_id
        debug_info.setdefault((from_agent_id, to_agent_id), []).append(transmission_chain)

        distance_before = distance_matrix[from_agent_id, to_agent_id]

        if distance_before > distance:
            distance_matrix[from_agent_id, to_agent_id] = distance
            distance_first_reaching[from_agent_id, to_agent_id] = to_ro.interval.from_incl
        elif distance_before == distance:
            distance_first_reaching[from_agent_id, to_agent_id] = min(distance_first_reaching[from_agent_id, to_agent_id], to_ro.interval.from_incl)

    # almost-inverse: distance -> weights
    weights_matrix = 1 / (distance_matrix + 0.000001)
    # normalize
    np_max = np.max(weights_matrix)
    print(np_max)
    weights_matrix /= np_max
    print(weights_matrix)
    # 4. visualize
    # take minimum depth of transmission between and the time reaching the second agent as coordinates
    # TODO is it a good idea to take minimum depth, should it not rather be cumulated distance from the malfunction train?
    pos = {}
    pos[malfunction.agent_id] = (0, 0)
    max_depth = 0

    for to_agent_id in range(number_of_trains):
        if to_agent_id == malfunction_agent_id or to_agent_id not in minimal_depth:
            continue

        #  take minimum depth as row
        d = minimal_depth[to_agent_id]

        # take earliest hop on at minimum depth, might not correspond to minimum distance!!
        hop_ons = hop_on_at_depth[to_agent_id][d]
        hop_ons.sort(key=lambda t: t.interval.from_incl)

        pos[to_agent_id] = (hop_ons[0].interval.from_incl, d)
        max_depth = max(max_depth, d)
    # TODO explantion for upward arrows?
    # TODO why do we have bidirectional arrays? is this a bug?
    # TODO why jumping over from malfunction agent
    staengeli_index = 0
    nb_not_affected = len([agent_id for agent_id in range(number_of_trains) if agent_id not in pos])
    for agent_id in range(number_of_trains):
        if agent_id not in pos:
            # distribute evenly over diagram
            pos[agent_id] = (staengeli_index * max_time_schedule / nb_not_affected, max_depth + 5)
            staengeli_index += 1
    _plot_encounter_graph_directed(
        weights_matrix=weights_matrix,
        changed_agents=changed_agents,
        title=f"Encounter Graph for experiment {experiment_result.experiment_id}, {malfunction}",
        file_name=f"encounter_graph_{experiment_result.experiment_id:04d}.pdf",
        pos=pos)

    return debug_info, distance_matrix, weights_matrix


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
    for _, occupations in resource_occupations_per_resource.items():
        occupations.sort(key=lambda ro: ro.interval.from_incl)
    return resource_occupations_per_resource, resource_occupations_per_agent


def _plot_resource_time_diagram(malfunction: ExperimentMalfunction,
                                resource_sorting: ResourceSorting,
                                nb_agents: int,
                                resource_occupations_schedule: SortedResourceOccupationsPerAgentDict,
                                resource_occupations_reschedule: SortedResourceOccupationsPerAgentDict,
                                width: int) -> Dict[int, bool]:
    malfunction_agent_id = malfunction.agent_id
    # TODO extract sorting from  resource_time_2d
    schedule_trajectories = trajectories_from_resource_occupations_per_agent(resource_occupations_schedule, resource_sorting, width)
    reschedule_trajectories = trajectories_from_resource_occupations_per_agent(resource_occupations_reschedule, resource_sorting, width)

    # Plotting the graphs
    ranges = (len(resource_sorting),
              max(max([ro[-1].interval.to_excl for ro in resource_occupations_schedule.values() if len(ro) > 0]),
                  max([ro[-1].interval.to_excl for ro in resource_occupations_reschedule.values() if len(ro) > 0])))
    # Plot Schedule
    if False:
        plot_time_resource_data(trajectories=schedule_trajectories, title='Schedule', ranges=ranges)
    # Plot Reschedule Full
    if False:
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
            len([agent_id for agent_id, changed in changed_agents_list.items() if changed]),
            nb_agents,
            "TODO"))
    # Plot difference
    if False:
        plot_time_resource_data(trajectories=traces_influenced_agents, title='Changed Agents',
                                ranges=ranges)
    return changed_agents_list


def trajectories_from_resource_occupations_per_agent(resource_occupations_schedule: SortedResourceOccupationsPerAgentDict,
                                                     resource_sorting: ResourceSorting,
                                                     width: int) -> Trajectories:
    schedule_trajectories: Trajectories = []
    for _, resource_ocupations in resource_occupations_schedule.items():
        train_time_path = []
        for resource_ocupation in resource_ocupations:
            position = coordinate_to_position(width, [resource_ocupation.resource])[0]
            # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
            if position not in resource_sorting:
                resource_sorting[position] = len(resource_sorting)
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.from_incl))
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.to_excl))
            train_time_path.append((None, None))
        schedule_trajectories.append(train_time_path)
    return schedule_trajectories


def _debug_directed(pair, debug_info, distance_matrix, weights_matrix):
    print(f"debug_info[{pair}]={debug_info[pair]}")
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
        experiment_base_directory='./agent_0_malfunction_2020_05_18T11_56_31',
        experiment_ids=list(range(1)),
    )
