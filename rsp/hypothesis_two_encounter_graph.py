import pprint
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from rsp.compute_time_analysis.compute_time_analysis import extract_schedule_plotting
from rsp.compute_time_analysis.compute_time_analysis import plot_resource_time_diagram
from rsp.encounter_graph.encounter_graph_visualization import _plot_encounter_graph_directed
from rsp.transmission_chains.transmission_chains import distance_matrix_from_tranmission_chains
from rsp.transmission_chains.transmission_chains import extract_transmission_chains
from rsp.transmission_chains.transmission_chains import TransmissionChain
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder

_pp = pprint.PrettyPrinter(indent=4)


def hypothesis_two_disturbance_propagation_graph(
        experiment_base_directory: str,
        experiment_ids: List[int] = None,
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
        experiment_base_directory='../rsp-data/agent_0_malfunction_2020_05_18T11_56_31',
        experiment_ids=list(range(1))
    )
