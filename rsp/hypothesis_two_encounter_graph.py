import pprint
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import tqdm
from flatland.core.grid.grid_utils import coordinate_to_position

from rsp.compute_time_analysis.compute_time_analysis import _get_difference_in_time_space_trajectories
from rsp.compute_time_analysis.compute_time_analysis import extract_schedule_plotting
from rsp.compute_time_analysis.compute_time_analysis import plot_resource_time_diagram
from rsp.compute_time_analysis.compute_time_analysis import plot_time_resource_trajectories
from rsp.compute_time_analysis.compute_time_analysis import plot_time_window_resource_trajectories
from rsp.compute_time_analysis.compute_time_analysis import Trajectories
from rsp.compute_time_analysis.compute_time_analysis import trajectories_from_resource_occupations_per_agent
from rsp.encounter_graph.encounter_graph_visualization import _plot_encounter_graph_directed
from rsp.logger import rsp_logger
from rsp.transmission_chains.transmission_chains import distance_matrix_from_tranmission_chains
from rsp.transmission_chains.transmission_chains import extract_transmission_chains
from rsp.transmission_chains.transmission_chains import extract_transmission_chains_from_time_windows
from rsp.transmission_chains.transmission_chains import TransmissionChain
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import SchedulingProblemInTimeWindows
from rsp.utils.data_types_converters_and_validators import extract_time_windows
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder
from rsp.utils.global_constants import RELEASE_TIME

_pp = pprint.PrettyPrinter(indent=4)


def hypothesis_two_disturbance_propagation_graph(
        experiment_base_directory: str,
        experiment_ids: List[int] = None,
        width: int = 400,
        show: bool = True
):
    """

    Parameters
    ----------
    experiment_base_directory
    experiment_ids
    width
    show
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

        experiment_result: ExperimentResultsAnalysis = experiment_results_list[i]

        disturbance_propagation_graph_visualization(experiment_result, show=show)


def disturbance_propagation_graph_visualization(
        experiment_result: ExperimentResultsAnalysis,
        show: bool = True
) -> Tuple[List[TransmissionChain], np.ndarray, np.ndarray, Dict[int, int]]:
    """

    Parameters
    ----------
    show
    experiment_result

    Returns
    -------
    transmission_chains, distance_matrix, weights_matrix, minimal_depth

    """
    # Get full schedule Time-resource-Data
    schedule = experiment_result.results_full.trainruns_dict
    malfunction = experiment_result.malfunction
    malfunction_agent_id = malfunction.agent_id
    number_of_trains = len(schedule)
    max_time_schedule = np.max([trainrun[-1].scheduled_at for trainrun in schedule.values()])

    # 0. data preparation
    # aggregate occupations of each resource
    schedule_plotting = extract_schedule_plotting(experiment_result=experiment_result)

    schedule_resource_occupations_per_agent = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent
    schedule_resource_occupations_per_resource = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_resource
    reschedule_full_resource_occupations_per_agent = schedule_plotting.reschedule_full_as_resource_occupations.sorted_resource_occupations_per_agent

    # 1. compute the forward-only wave of the malfunction
    # "forward-only" means only agents running at or after the wave hitting them are considered,
    # i.e. agents do not decelerate ahead of the wave!
    transmission_chains = extract_transmission_chains(malfunction=malfunction,
                                                      resource_occupations_per_agent=schedule_resource_occupations_per_agent,
                                                      resource_occupations_per_resource=schedule_resource_occupations_per_resource)
    # TODO SIM-549
    if False:
        # 2. visualize the wave in resource-time diagram
        # TODO remove dirty hack of visualizing wave as additional agent
        wave_agent_id = number_of_trains
        fake_resource_occupations = [
            ResourceOccupation(interval=transmission_chain[-1].hop_off.interval,
                               resource=transmission_chain[-1].hop_off.resource,
                               direction=transmission_chain[-1].hop_off.direction,
                               agent_id=wave_agent_id)
            for transmission_chain in transmission_chains
        ]

        schedule_resource_occupations_per_agent[wave_agent_id] = fake_resource_occupations
        reschedule_full_resource_occupations_per_agent[wave_agent_id] = []

        # get resource
        changed_agents: Dict[int, bool] = plot_resource_time_diagram(schedule_plotting)

        # 3. non-symmetric distance matrix of primary, secondary etc. effects
        distance_matrix, weights_matrix, minimal_depth, wave_fronts_reaching_other_agent = distance_matrix_from_tranmission_chains(
            number_of_trains=number_of_trains, transmission_chains=transmission_chains)
        # 4. visualize
        _plot_delay_propagation_graph(changed_agents, experiment_result, malfunction, malfunction_agent_id, max_time_schedule, minimal_depth, number_of_trains,
                                      wave_fronts_reaching_other_agent, weights_matrix)

    plot_time_window_resource_trajectories(
        experiment_result=experiment_result,
        plotting_information=schedule_plotting.plotting_information
    )

    # 5. transmission chains re-scheduling problem
    rsp_logger.info("start extract_time_windows")
    re_schedule_full_time_windows: SchedulingProblemInTimeWindows = extract_time_windows(
        route_dag_constraints_dict=experiment_result.problem_full_after_malfunction.route_dag_constraints_dict,
        minimum_travel_time_dict=experiment_result.problem_full_after_malfunction.minimum_travel_time_dict,
        release_time=RELEASE_TIME)
    rsp_logger.info("end extract_time_windows")
    rsp_logger.info("start extract_transmission_chains_from_time_windows")
    transmission_chains_time_window: List[TransmissionChain] = extract_transmission_chains_from_time_windows(
        time_windows=re_schedule_full_time_windows,
        malfunction=malfunction)
    rsp_logger.info("end extract_transmission_chains_from_time_windows")

    trajectories_from_transmission_chains_time_window: Trajectories = [[] for _ in schedule]
    plotting_information = schedule_plotting.plotting_information
    for transmission_chain in tqdm.tqdm(transmission_chains_time_window):
        last_time_window = transmission_chain[-1].hop_off
        position = coordinate_to_position(plotting_information.grid_width, [last_time_window.resource])[0]
        # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
        if position not in plotting_information.sorting:
            plotting_information.sorting[position] = len(plotting_information.sorting)

        agent_id = last_time_window.agent_id
        train_time_path = trajectories_from_transmission_chains_time_window[agent_id]
        train_time_path.append((plotting_information.sorting[position], last_time_window.interval.from_incl))
        train_time_path.append((plotting_information.sorting[position], last_time_window.interval.to_excl))
        train_time_path.append((None, None))

    plot_time_resource_trajectories(
        trajectories=trajectories_from_transmission_chains_time_window,
        title='Time Window Propagation',
        malfunction=malfunction,
        ranges=plotting_information.dimensions
    )

    # Plot difference of reschedule_full with prediciton
    trajectories_reschedule_full: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=reschedule_full_resource_occupations_per_agent,
        plotting_information=plotting_information)
    trajectories_influenced_agents, changed_agents_list = _get_difference_in_time_space_trajectories(
        trajectories_a=trajectories_reschedule_full,
        trajectories_b=trajectories_from_transmission_chains_time_window)
    plot_time_resource_trajectories(
        trajectories=trajectories_influenced_agents,
        title='Reduction by prediction',
        malfunction=malfunction,
        ranges=plotting_information.dimensions
    )

    return transmission_chains, distance_matrix, weights_matrix, minimal_depth


def _plot_delay_propagation_graph(
        changed_agents,
        experiment_result,
        malfunction,
        malfunction_agent_id,
        max_time_schedule,
        minimal_depth,
        number_of_trains,
        wave_fronts_reaching_other_agent,
        weights_matrix,
        file_name: str = None,
):
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
        hop_ons = wave_fronts_reaching_other_agent[to_agent_id][d]
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
        file_name=file_name,
        pos=pos)


if __name__ == '__main__':
    hypothesis_two_disturbance_propagation_graph(
        experiment_base_directory='../rsp-data/agent_0_malfunction_2020_05_27T19_45_49',
        experiment_ids=[0]
    )
