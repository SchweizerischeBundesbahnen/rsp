from rsp.hypothesis_two_encounter_graph import compute_disturbance_propagation_graph
from rsp.hypothesis_two_encounter_graph import plot_delay_propagation_graph
from rsp.step_03_run.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.step_04_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_delay_propagation_2d
from rsp.step_04_analysis.detailed_experiment_analysis.schedule_plotting import extract_schedule_plotting
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import get_difference_in_time_space_trajectories
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import trajectories_from_resource_occupations_per_agent

if __name__ == "__main__":
    experiment_base_directory = "../src.python.rsp-data/agent_0_malfunction_2020_06_22T11_48_47/"
    agent_of_interest = 26

    experiment_data_directory = f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}"

    # ========================================================
    # Plotting differnt figures for malfunciton investigation
    # #========================================================

    for experiment_id in range(48):
        experiment_of_interest = experiment_id
        file_name = "../src.python.rsp-data/Call_Emma/delay_propagation_{}.png".format(str(experiment_id).zfill(5))
        file_name_2d = "../src.python.rsp-data/Call_Emma/spacial_delay_propagation_{}.png".format(str(experiment_id).zfill(5))
        exp_results_of_experiment_of_interest = load_and_expand_experiment_results_from_data_folder(
            experiment_data_folder_name=experiment_data_directory, experiment_ids=[experiment_of_interest], nonify_all_structured_fields=False
        )[0]
        plotting_data = extract_schedule_plotting(experiment_result=exp_results_of_experiment_of_interest, sorting_agent_id=agent_of_interest)

        transmission_chains, distance_matrix, minimal_depth = compute_disturbance_propagation_graph(schedule_plotting=plotting_data)
        schedule_resource_occupations = plotting_data.schedule_as_resource_occupations.sorted_resource_occupations_per_agent
        schedule_trajectories = trajectories_from_resource_occupations_per_agent(schedule_resource_occupations, plotting_data.plotting_information)

        reschedule_resource_occupations = plotting_data.reschedule_delta_perfect_as_resource_occupations.sorted_resource_occupations_per_agent
        reschedule_trajectories = trajectories_from_resource_occupations_per_agent(reschedule_resource_occupations, plotting_data.plotting_information)

        _, changed_agents_dict = get_difference_in_time_space_trajectories(base_trajectories=schedule_trajectories, target_trajectories=reschedule_trajectories)

        # Get resource occupation and time-space trajectories for full reschedule

        plot_delay_propagation_graph(minimal_depth=minimal_depth, distance_matrix=distance_matrix, file_name=file_name, changed_agents=changed_agents_dict)

        # Compute actual changes and plot the effects of this

        plot_delay_propagation_2d(
            plotting_data=plotting_data,
            delay_information=exp_results_of_experiment_of_interest.lateness_delta_perfect_after_malfunction,
            depth_dict=minimal_depth,
            changed_agents=changed_agents_dict,
            file_name=file_name_2d,
        )
