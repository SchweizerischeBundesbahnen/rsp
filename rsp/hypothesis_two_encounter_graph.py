import pprint
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import plotly.graph_objects as go

from rsp.transmission_chains.transmission_chains import distance_matrix_from_tranmission_chains
from rsp.transmission_chains.transmission_chains import extract_transmission_chains_from_schedule
from rsp.transmission_chains.transmission_chains import TransmissionChain
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ResourceOccupation
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
    distance_matrix, minimal_depth, wave_fronts_reaching_other_agent = distance_matrix_from_tranmission_chains(
        number_of_trains=number_of_trains, transmission_chains=transmission_chains)

    return transmission_chains, distance_matrix, minimal_depth


def resource_occpuation_from_transmission_chains(transmission_chains: List[TransmissionChain], changed_agents: Dict) -> List[ResourceOccupation]:
    """Method to construct Ressource Occupation from transmition chains. Used
    to plot the transmission in the resource-time-diagram.

    Parameters
    ----------
    transmission_chains

    Returns
    -------
    Ressource Occupation of a given Transmission Chain
    """
    wave_plotting_id = -1
    time_resource_malfunction_wave = [
        ResourceOccupation(interval=transmission_chain[-1].hop_off.interval,
                           resource=transmission_chain[-1].hop_off.resource,
                           direction=transmission_chain[-1].hop_off.direction,
                           agent_id=wave_plotting_id)
        for transmission_chain in transmission_chains
        if changed_agents[transmission_chain[-1].hop_off.agent_id]]
    wave_resource_occupations: List[ResourceOccupation] = time_resource_malfunction_wave
    return wave_resource_occupations


def plot_delay_propagation_graph(
        changed_agents,
        minimal_depth,
        distance_matrix):
    """

    Parameters
    ----------
    changed_agents
    minimal_depth
    transmission_chains
    ToDO: This function is not working properly yet
    Returns
    -------

    """
    layout = go.Layout(
        plot_bgcolor='rgba(46,49,49,1)'
    )
    fig = go.Figure(layout=layout)
    max_depth = max(list(minimal_depth.values()))
    agents_per_depth = {}
    for i in range(max_depth + 1):
        agents_per_depth[i] = 0
    num_agents = len(distance_matrix[:, 0])
    for from_agent in range(num_agents):
        node_positions = {}
        node_line = []
        if from_agent in minimal_depth.keys():
            from_agent_depth = minimal_depth[from_agent]
            if from_agent not in list(node_positions.keys()):
                node_positions[from_agent] = (from_agent_depth, agents_per_depth[from_agent_depth])
                agents_per_depth[from_agent_depth] += 10
            for to_agent in range(num_agents):
                if to_agent in minimal_depth.keys():
                    to_agent_depth = minimal_depth[to_agent]
                    if 1. / distance_matrix[from_agent, to_agent] > 0.001 and from_agent_depth < to_agent_depth:
                        if to_agent not in list(node_positions.keys()):
                            node_positions[to_agent] = (to_agent_depth, agents_per_depth[to_agent_depth])
                            agents_per_depth[to_agent_depth] += 10
                        node_line.append(node_positions[from_agent])
                        node_line.append(node_positions[to_agent])
                        node_line.append((None, None))



        x = []
        y = []
        for pos in node_line:
            x.append(pos[1])
            y.append(pos[0])
        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='lines+markers',
            marker=dict(size=5)
        ))
    fig.update_yaxes(zeroline=False, showgrid=True, range=[max_depth, 0], tick0=-0.5, dtick=1, gridcolor='Grey')

    fig.show()


if __name__ == '__main__':
    hypothesis_two_disturbance_propagation_graph(
        experiment_base_directory='../rsp-data/agent_0_malfunction_2020_05_18T11_56_31',
        experiment_ids=list(range(1))
    )
