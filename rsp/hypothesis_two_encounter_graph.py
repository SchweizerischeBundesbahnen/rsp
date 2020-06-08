import pprint
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from rsp.encounter_graph.encounter_graph_visualization import _plot_encounter_graph_directed
from rsp.transmission_chains.transmission_chains import distance_matrix_from_tranmission_chains
from rsp.transmission_chains.transmission_chains import extract_transmission_chains_from_schedule
from rsp.transmission_chains.transmission_chains import TransmissionChain
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import SortedResourceOccupationsPerAgent
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder
from rsp.utils.plotting_data_types import SchedulePlotting

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

        # Todo -> This currently does nothing by itself. do we need this at all?
        compute_disturbance_propagation_graph(experiment_result)


def compute_disturbance_propagation_graph(schedule_plotting: SchedulePlotting) \
        -> Tuple[List[TransmissionChain], np.ndarray, np.ndarray, Dict[int, int]]:
    """Method to Compute the disturbance propagation in the schedule when there
    is no dispatching done. This method will return more changed agents than
    will actually change.

    Parameters
    ----------
    experiment_result

    Returns
    -------
    transmission_chains, distance_matrix, weights_matrix, minimal_depth
    """

    # 1. compute the forward-only wave of the malfunction
    transmission_chains = extract_transmission_chains_from_schedule(schedule_plotting=schedule_plotting)

    # 2. non-symmetric distance matrix of primary, secondary etc. effects
    number_of_trains = len(schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent)
    distance_matrix, weights_matrix, minimal_depth, wave_fronts_reaching_other_agent = distance_matrix_from_tranmission_chains(
        number_of_trains=number_of_trains, transmission_chains=transmission_chains)

    return transmission_chains, distance_matrix, weights_matrix, minimal_depth


def resource_occpuation_from_transmission_chains(transmission_chains: List[TransmissionChain]) -> SortedResourceOccupationsPerAgent:
    """Method to construct Ressource Occupation from transmition chains. Used
    to plot the transmission in the resource-time-diagram.

    Parameters
    ----------
    transmission_chains

    Returns
    -------
    Ressource Occupation of a given Transmission Chain
    """
    wave_resource_occupations: SortedResourceOccupationsPerAgent = {}
    wave_plotting_id = -1
    time_resource_malfunction_wave = [
        ResourceOccupation(interval=transmission_chain[-1].hop_off.interval,
                           resource=transmission_chain[-1].hop_off.resource,
                           direction=transmission_chain[-1].hop_off.direction,
                           agent_id=wave_plotting_id)
        for transmission_chain in transmission_chains]
    wave_resource_occupations[wave_plotting_id] = time_resource_malfunction_wave
    return wave_resource_occupations


def plot_delay_propagation_graph(
        changed_agents,
        max_time_schedule,
        minimal_depth,
        wave_fronts_reaching_other_agent,
        weights_matrix,
        file_name: Optional[str] = None,
):
    """

    Parameters
    ----------
    changed_agents
    max_time_schedule
    minimal_depth
    wave_fronts_reaching_other_agent
    weights_matrix
    file_name

    Returns
    -------

    """
    # TODO Fix this method!
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
        title=f"Delay Propagation Graph",
        file_name=file_name,
        pos=pos)


if __name__ == '__main__':
    hypothesis_two_disturbance_propagation_graph(
        experiment_base_directory='../rsp-data/agent_0_malfunction_2020_05_18T11_56_31',
        experiment_ids=list(range(1))
    )
