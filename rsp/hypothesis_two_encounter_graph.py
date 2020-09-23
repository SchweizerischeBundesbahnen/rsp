import pprint
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import tqdm
from flatland.core.grid.grid_utils import coordinate_to_position

from rsp.analysis.compute_time_analysis import extract_schedule_plotting
from rsp.analysis.compute_time_analysis import get_difference_in_time_space_trajectories
from rsp.analysis.compute_time_analysis import plot_time_resource_trajectories
from rsp.analysis.compute_time_analysis import time_windows_as_resource_occupations_per_agent
from rsp.analysis.compute_time_analysis import Trajectories
from rsp.analysis.compute_time_analysis import trajectories_from_resource_occupations_per_agent
from rsp.transmission_chains.transmission_chains import distance_matrix_from_tranmission_chains
from rsp.transmission_chains.transmission_chains import extract_transmission_chains_from_schedule
from rsp.transmission_chains.transmission_chains import extract_transmission_chains_from_time_windows
from rsp.transmission_chains.transmission_chains import TransmissionChain
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import SchedulingProblemInTimeWindows
from rsp.utils.data_types_converters_and_validators import extract_time_windows
from rsp.utils.global_constants import RELEASE_TIME
from rsp.utils.plotting_data_types import SchedulePlotting
from rsp.utils.rsp_logger import rsp_logger

_pp = pprint.PrettyPrinter(indent=4)


def compute_disturbance_propagation_graph(schedule_plotting: SchedulePlotting) \
        -> Tuple[List[TransmissionChain], np.ndarray, np.ndarray, Dict[int, int]]:
    """Method to Compute the disturbance propagation in the schedule when there
    is no dispatching done. This method will return more changed agents than
    will actually change.

    Parameters
    ----------
    schedule_plotting

    Returns
    -------
    transmission_chains, distance_matrix, weights_matrix, minimal_depth
    """

    # 1. compute the forward-only wave of the malfunction
    transmission_chains = extract_transmission_chains_from_schedule(
        malfunction=schedule_plotting.malfunction,
        occupations=schedule_plotting.schedule_as_resource_occupations)

    # 2. non-symmetric distance matrix of primary, secondary etc. effects
    number_of_trains = len(schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent)
    distance_matrix, minimal_depth, _ = distance_matrix_from_tranmission_chains(
        number_of_trains=number_of_trains, transmission_chains=transmission_chains)

    return transmission_chains, distance_matrix, minimal_depth


# TODO SIM-672 remove noqa
def resource_occpuation_from_transmission_chains(  # noqa
        transmission_chains: List[TransmissionChain],
        changed_agents: Dict) -> List[ResourceOccupation]:
    """Method to construct Ressource Occupation from transmition chains. Used
    to plot the transmission in the resource-time-diagram.

    Parameters
    ----------
    changed_agents
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


def plot_transmission_chains_time_window(
        experiment_result: ExperimentResultsAnalysis,
        transmission_chains_time_window: List[TransmissionChain],
        output_folder: Optional[str] = None
):
    """

    Parameters
    ----------
    experiment_result
    transmission_chains_time_window

    Returns
    -------

    """
    schedule_plotting = extract_schedule_plotting(experiment_result=experiment_result)
    reschedule_full_resource_occupations_per_agent = schedule_plotting.reschedule_full_as_resource_occupations.sorted_resource_occupations_per_agent
    num_agents = len(experiment_result.results_full.trainruns_dict.keys())
    plotting_information = schedule_plotting.plotting_information

    prediction = extract_trajectories_from_transmission_chains_time_window(num_agents, plotting_information, transmission_chains_time_window)
    plot_time_resource_trajectories(
        trajectories=prediction,
        title='Time Window Prediction',
        schedule_plotting=schedule_plotting,
        output_folder=output_folder
    )

    # TODO sanity check:
    trajectories_reschedule_full_time_windows = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=time_windows_as_resource_occupations_per_agent(problem=experiment_result.problem_full_after_malfunction),
        plotting_information=schedule_plotting.plotting_information)
    sanity_false_positives, _ = get_difference_in_time_space_trajectories(
        base_trajectories=prediction,
        target_trajectories=trajectories_reschedule_full_time_windows)
    plot_time_resource_trajectories(
        trajectories=sanity_false_positives,
        title='Sanity false positives (in prediction, but not in re-schedule full time windows): should be empty',
        schedule_plotting=schedule_plotting,
        output_folder=output_folder
    )
    sanity_false_negatives, _ = get_difference_in_time_space_trajectories(
        target_trajectories=prediction,
        base_trajectories=trajectories_reschedule_full_time_windows)
    plot_time_resource_trajectories(
        trajectories=sanity_false_negatives,
        title='Sanity false negatives (not in prediction but in re-schedule full time windows): reduction by prediction - any?',
        schedule_plotting=schedule_plotting,
        output_folder=output_folder
    )
    # Get trajectories for reschedule full
    trajectories_reschedule_full: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=reschedule_full_resource_occupations_per_agent,
        plotting_information=plotting_information)

    # Plot difference with prediction
    false_negatives, _ = get_difference_in_time_space_trajectories(
        base_trajectories=trajectories_reschedule_full,
        target_trajectories=prediction)
    plot_time_resource_trajectories(
        trajectories=false_negatives,
        # TODO SIM-549 is there something wrong because release times are not contained in time windows?
        title='False negatives (in re-schedule full but not in prediction)',
        schedule_plotting=schedule_plotting,
        output_folder=output_folder
    )
    false_positives, _ = get_difference_in_time_space_trajectories(
        base_trajectories=prediction,
        target_trajectories=trajectories_reschedule_full)
    plot_time_resource_trajectories(
        trajectories=false_positives,
        title='False positives (in prediction but not in re-schedule full)',
        schedule_plotting=schedule_plotting,
        output_folder=output_folder
    )
    # TODO SIM-549 damping: probabilistic delay propagation?


def extract_trajectories_from_transmission_chains_time_window(
        num_agents,
        plotting_information,
        transmission_chains_time_window,
        release_time: int = RELEASE_TIME):
    trajectories_from_transmission_chains_time_window: Trajectories = {agent_id: [] for agent_id in range(num_agents)}
    for transmission_chain in tqdm.tqdm(transmission_chains_time_window):
        last_time_window = transmission_chain[-1].hop_off
        position = coordinate_to_position(plotting_information.grid_width, [last_time_window.resource])[0]
        # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
        if position not in plotting_information.sorting:
            plotting_information.sorting[position] = len(plotting_information.sorting)

        agent_id = last_time_window.agent_id
        train_time_path = trajectories_from_transmission_chains_time_window[agent_id]
        train_time_path.append((plotting_information.sorting[position], last_time_window.interval.from_incl))
        train_time_path.append((plotting_information.sorting[position], last_time_window.interval.to_excl + release_time))
        train_time_path.append((None, None))
    return trajectories_from_transmission_chains_time_window


def extract_time_windows_and_transmission_chains(experiment_result: ExperimentResultsAnalysis) -> List[TransmissionChain]:
    """Extract time windows from scheduling problem and derive transmission
    chains from them.

    Parameters
    ----------
    experiment_result

    Returns
    -------
    """
    rsp_logger.info("start extract_time_windows")
    malfunction = experiment_result.malfunction
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
    return transmission_chains_time_window


def plot_delay_propagation_graph(  # noqa: C901
        minimal_depth: dict,
        distance_matrix,
        changed_agents: dict,
        file_name: Optional[str] = None
):
    """

    Parameters
    ----------
    distance_matrix
    minimal_depth
    Returns
    -------

    """
    layout = go.Layout(
        plot_bgcolor='rgba(46,49,49,1)'
    )
    x_scaling = 5
    fig = go.Figure(layout=layout)
    max_depth = max(list(minimal_depth.values()))
    agents_per_depth = [[] for _ in range(max_depth + 1)]
    agent_counter_per_depth = [0 for _ in range(max_depth + 1)]

    # see changed agents that are not in the propagation
    changed_agents_ids = dict(filter(lambda elem: elem[1], changed_agents.items()))
    true_positives = set.intersection(set(changed_agents_ids.keys()), set(minimal_depth.keys()))
    false_negatives = set(changed_agents_ids.keys()) - true_positives
    print("Agents not shown but affected \n", list(false_negatives))
    # get agents for each depth
    for agent, depth in minimal_depth.items():
        agents_per_depth[depth].append(agent)

    num_agents = len(distance_matrix[:, 0])
    node_positions = {}
    for depth in range(max_depth + 1):
        for from_agent in agents_per_depth[depth]:
            node_line = []
            from_agent_depth = depth
            if from_agent not in list(node_positions.keys()):
                node_positions[from_agent] = (from_agent_depth, x_scaling * (agent_counter_per_depth[depth] - 0.5 * len(agents_per_depth[depth])))
                agent_counter_per_depth[depth] += 1
            for to_agent in range(num_agents):
                if from_agent == to_agent:
                    continue
                if to_agent in minimal_depth.keys():
                    to_agent_depth = minimal_depth[to_agent]
                    # Check if the agents are connected and only draw lines from lower to deeper influence depth
                    if 1. / distance_matrix[from_agent, to_agent] > 0.001 and from_agent_depth < to_agent_depth:
                        if to_agent not in list(node_positions.keys()):
                            rel_pos = node_positions[from_agent][1]
                            node_positions[to_agent] = (
                                to_agent_depth, rel_pos + x_scaling * (agent_counter_per_depth[to_agent_depth] - 0.5 * len(agents_per_depth[to_agent_depth])))
                            agents_per_depth[to_agent_depth][agent_counter_per_depth[to_agent_depth]] = to_agent
                            agent_counter_per_depth[to_agent_depth] += 1

                        node_line.append(node_positions[from_agent])
                        node_line.append(node_positions[to_agent])
                        node_line.append((None, None))
            x = []
            y = []
            for pos in node_line:
                x.append(pos[1])
                y.append(pos[0])
            if from_agent in true_positives:
                color = "red"
            else:
                color = "yellow"
            fig.add_trace(go.Scattergl(
                x=x,
                y=y,
                mode='lines+markers',
                name="Agent {}".format(from_agent),
                marker=dict(size=5, color=color)
            ))

    fig.update_yaxes(zeroline=False, showgrid=True, range=[max_depth, 0], tick0=0, dtick=1, gridcolor='Grey', title="Influence Depth")
    fig.update_xaxes(zeroline=False, showgrid=False, ticks=None, visible=False)
    if file_name is None:
        fig.show()
    else:
        fig.write_image(file_name)
